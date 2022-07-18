"""Main class, holding information about models and training/testing routines."""

import torch
import time
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK
import torch.nn.functional as F

from .forgemaster_base import _Forgemaster
from ..consts import NON_BLOCKING, BENCHMARK

class ForgemasterTargeted_w_causal(_Forgemaster):

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
        if not 'output' in self.args.causal_loss_type:
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



    def _define_objective(self, inputs, labels, n=0):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, causal_model, causal_criterion):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            new_labels = self._label_map(torch.zeros(1,10), labels)
            # add causal_loss
            if self.args.causal_beta != 0:
                if self.args.causal_loss_type == 'perturb_v_output_and_s_output':
                    if n==0:
                        causal_criterion.loss_type = 'perturb_s_output'
                        causal_loss = causal_criterion.run(causal_model, inputs, labels, new_labels)
                        causal_loss.backward(retain_graph=True)
                    else:
                        causal_criterion.loss_type = 'perturb_v_output'
                        causal_loss = causal_criterion.run(causal_model, inputs, labels, new_labels)
                        causal_loss.backward(retain_graph=True)
                        torch.cuda.empty_cache()
                        return causal_loss.detach().cpu()
                else:
                    causal_loss = causal_criterion.run(causal_model, inputs, labels, new_labels)
                    causal_loss.backward(retain_graph=True)


            if not self.args.only_causal:
                outputs = model(inputs)
                loss = criterion(outputs, new_labels)
                loss.backward()
                torch.cuda.empty_cache()
            else:
                loss = 0




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
                    client.causal_criterion.loss_type = "perturb_v_output"
                elif self.args.causal_loss_type == "perturb_v_output":
                    client.causal_criterion.loss_type = "perturb_s_output"
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

    def _batched_step(self, poison_delta, poison_bounds, example, client, furnace):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example
        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Add adversarial pattern
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = furnace.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        if len(batch_positions) > 0:
            '''if self.args.causal_loss_type == 'perturb_v_output_and_s_output':
                clean_inputs = inputs.clone().detach()'''
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()
            poison_images = inputs[batch_positions]
            if self.args.recipe == 'poison-frogs':
                self.targets = inputs.clone().detach()
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = furnace.augment(inputs, randgen=None)

            # Define the loss objective and compute gradients
            closure = self._define_objective(inputs, labels)
            loss, prediction = client.compute(closure)
            delta_slice = client.sync_gradients(delta_slice)
            #grad = delta_slice.grad

            '''if self.args.causal_loss_type == 'perturb_v_output_and_s_output':
                inputs_2 = clean_inputs.detach()
                delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
                if self.args.clean_grad:
                    delta_slice = torch.zeros_like(delta_slice)
                delta_slice.requires_grad_()
                inputs_2[batch_positions] += delta_slice
                if self.args.paugment:
                    inputs_2 = furnace.augment(inputs_2, randgen=None)

                closure = self._define_objective(inputs_2, labels, n=1)
                loss_2 = client.compute(closure)
                delta_slice.grad += grad'''

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, furnace.dm, furnace.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()