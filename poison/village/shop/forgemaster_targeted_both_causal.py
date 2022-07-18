"""Main class, holding information about models and training/testing routines."""

import torch
import time
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster
from ..consts import NON_BLOCKING, BENCHMARK

class ForgemasterTargeted_both_causal(_Forgemaster):

    def _initialize_forge(self, client, furnace):
        """Implement common initialization operations for forgeing."""
        client.eval(dropout=True)
        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            #self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau

        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble

        if self.args.full_data:
            dataloader = furnace.trainloader
        else:
            dataloader = furnace.poisonloader

        #update class_mu for causal_criterion
        mu = [torch.zeros(1, client.causal_model.hid_channels) for _ in range(furnace.num_class)]
        num = [0 for _ in range(furnace.num_class)]
        for batch, example in enumerate(dataloader):
            inputs, labels, ids = example
            inputs = inputs.to(**self.setup)
            labels_cuda = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

            if 's' in self.args.causal_loss_type:
                with torch.no_grad():
                    rep = client.causal_model.get_causal_rep(inputs).cpu()
            else:
                x_s, x_v, x_v_att = client.causal_model.split_x(inputs, labels_cuda, eval=True)
                with torch.no_grad():
                    rep = client.causal_model.get_conf_rep(x_v).cpu()
            for i in range(inputs.size(0)):
                mu[labels[i]] += rep[i]
                num[labels[i]] += 1
        for j in range(furnace.num_class):
            mu[j] /= num[j]
        client.causal_criterion.update_mu(torch.cat(mu, dim=0).to(**self.setup))



    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, causal_model, causal_criterion):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)
            new_labels = self._label_map(outputs, labels)

            causal_criterion.loss_type = self.args.causal_loss_type
            causal_criterion.beta = 1
            s_loss = causal_criterion.run(causal_model, inputs, labels, new_labels)
            s_loss.backward(retain_graph=self.retrain)

            # add causal_loss
            causal_criterion.loss_type = 'perturb_v_output'
            causal_criterion.beta = self.args.causal_beta
            v_loss = causal_criterion.run(causal_model, inputs, labels, new_labels)
            #loss = s_loss + v_loss

            v_loss.backward(retain_graph=self.retain)
            prediction = (outputs.data.argmax(dim=1) == new_labels).sum()

            return loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _label_map(self, outputs, labels):
        # This is a naiive permutation on the label space. You can implement
        # any permutation you like here.
        new_labels = (labels + 1) % outputs.shape[1]
        return new_labels

    def _run_trial(self, client, furnace):
        """Run a single trial."""
        poison_delta = furnace.initialize_poison()
        if self.args.full_data:
            dataloader = furnace.trainloader
        else:
            dataloader = furnace.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = furnace.dm.to(device=torch.device('cpu')), furnace.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        for step in range(self.args.attackiter):
            if step % 10 == 0:
                print(f'Step {step}')
            if step == self.args.attackiter // 2 and self.args.causal_reverse:
                if self.args.causal_loss_type == "perturb_s_output":
                    client.causal_criterion.loss_type = "perturb_v_output_min"
                elif self.args.causal_loss_type == "perturb_v_output":
                    client.causal_criterion.loss_type = "perturb_s_output_min"
            target_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                if batch == 0:
                    start = time.time()
                elif batch % 100 == 0:
                    end = time.time()
                    avg = (end-start)/100
                    start = end
                    print(f'average time per epoch: {len(dataloader) * avg}')
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, client, furnace)
                target_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    client.step(furnace, None, self.targets, self.true_classes)
                else:
                    client.step(furnace, poison_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break

        return poison_delta, target_losses
