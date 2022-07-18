"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterTargeted_mc(_Forgemaster):

    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)
            new_labels = self._label_map(outputs, labels)
            loss = criterion(outputs, new_labels)
            loss.backward(retain_graph=self.retain)
            prediction = (outputs.data.argmax(dim=1) == new_labels).sum()
            return loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _label_map(self, outputs, labels):
        y_t = torch.zeros((labels.shape[0]), dtype=torch.int64).to(labels.device)
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        # import pdb; pdb.set_trace()
        ind_sorted = outputs.sort(dim=-1)[1]
        # index for those correctly classified.
        ind = ind_sorted[:, -1] == labels
        true_idcs = torch.nonzero(ind).squeeze(1)
        false_idcs = torch.nonzero(~ind).squeeze(1)
        y_t[true_idcs] = ind_sorted[:, -2][true_idcs]
        y_t[false_idcs] = ind_sorted[:, -1][false_idcs]
        # y_t = torch.from_numpy(y_t).type(torch.int64).to(y.device)
        return y_t
