import torch
import torch.nn as nn
import numpy as np
import time
from utils.config import cfg
import os
from PIL import Image

class ConfounderBank(nn.Module):
    def __init__(self):
        """ an API for Accessing Confounder Buffer
        Params:
            shape (int): feature shape (default: 128)
            K (int): bank size; number of confounder
            N (int): number of confounder for each sample
            decay (boolean): whether to decay the sampling probability or not
            confounder_queue (tensor): queue for confounder
            confounder_priority (tensor): priority of each confounder
            queue_ptr (tensor): pointer
        """
        super(ConfounderBank, self).__init__()
        K = cfg.MODEL.CONF_BUFFER_SIZE
        N = cfg.MODEL.CONF_SET_SIZE
        decay = cfg.MODEL.CONF_DECAY
        queue_shape = [K]
        if cfg.MODEL.CONF_MODE == 'input':
            shape = [cfg.DATASET.NUM_CHANNEL, cfg.DATASET.SIZE_W, cfg.DATASET.SIZE_H]
            queue_shape.extend(shape)
        elif cfg.MODEL.CONF_MODE == 'rep':
            shape = cfg.MODEL.HID_CHANNELS
            queue_shape.extend([shape])
        self.register_buffer("confounder_queue", torch.randn(queue_shape))
        self.register_buffer("confounder_priority", torch.ones(K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.confounder_priority = torch.ones(K, dtype=torch.float)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.confounder_queue = torch.clamp(self.confounder_queue, 0., 1.)

        assert N <= K
        self.K = K
        self.N = N
        self.decay = decay
        self.queue_shape = queue_shape
        self.B = cfg.batch_size

    @torch.no_grad()
    def reset(self):
        self.confounder_queue = torch.randn(self.queue_shape)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def add(self, x_v, conf_pred=None, target=None):
        bs = x_v.shape[0]
        shape = list(x_v.shape[1:])
        assert shape == self.queue_shape[1:]
        ptr = int(self.queue_ptr)
        max_prior = self.K // bs + 2

        # replace the keys at ptr (dequeue and enqueue)
        if ptr+bs > self.K:
            self.confounder_queue[ptr:, :] = x_v[:self.K-ptr].detach()
            self.confounder_queue[:(ptr + bs - self.K), :] = x_v[self.K-ptr:].detach()
            if conf_pred is None:
                if self.decay:
                    self.confounder_priority -= 1
                    self.confounder_priority = torch.clamp(self.confounder_priority, 1., max_prior+1)
                    self.confounder_priority[ptr:] = max_prior
                    self.confounder_priority[:(ptr + bs - self.K)] = max_prior
            else:
                assert target is not None
                if self.decay:
                    self.confounder_priority *= 0.95
                prob = torch.sigmoid(conf_pred)
                pred = torch.argmax(conf_pred, dim=1)
                # get the index of imgs that are incorrectly classfied
                incorrect = (pred != target).float()
                # get the second maximum probability
                mask = 1 - (prob == prob.max(dim=1, keepdim=True)[0]).float()
                prob = torch.mul(mask, prob)
                max_prob = prob.max(1)[0].float()

                priority = torch.where(pred == target, max_prob, incorrect)
                self.confounder_priority[ptr:] = priority[:self.K-ptr].detach()
                self.confounder_priority[:(ptr + bs - self.K)] = priority[self.K-ptr:].detach()
        else:
            self.confounder_queue[ptr:ptr + bs, :] = x_v.detach()
            if conf_pred is None:
                if self.decay:
                    self.confounder_priority -= 1
                    self.confounder_priority = torch.clamp(self.confounder_priority, 1., max_prior+1)
                    self.confounder_priority[ptr:ptr + bs] = max_prior
            else:
                assert target is not None
                if self.decay:
                    self.confounder_priority *= 0.95
                prob = torch.sigmoid(conf_pred)
                pred = torch.argmax(conf_pred,dim=1)
                # get the index of imgs that are incorrectly classfied
                incorrect = (pred != target).float()
                # get the second maximum probability
                mask = 1 - (prob == prob.max(dim=1, keepdim=True)[0]).float()
                prob = torch.mul(mask, prob)
                max_prob = prob.max(1)[0].float()

                priority = torch.where(pred == target, max_prob, incorrect)
                self.confounder_priority[ptr:ptr + bs] = priority.detach()

        ptr = (ptr + bs) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def batch_sample_set(self, x_s):
        bs_size = x_s.shape[0]
        conf_set = []
        index_list = [i for i in range(self.K)]
        prob = self.confounder_priority / torch.sum(self.confounder_priority) + 1e-3
        prob = np.array(prob.cpu())
        prob /= prob.sum()
        for _ in range(bs_size):
            selected = np.random.choice(index_list, self.N, replace=False, p = prob)
            conf_set.append(self.confounder_queue[selected].unsqueeze(0))
        conf_set = torch.cat(conf_set, dim=0)

        return conf_set

    def sample_and_save(self, dir, num=10):
        def tensor2np(x):
            x = x.permute(0, 2, 3, 1)
            x = x.detach().cpu().numpy() * 255.0
            x = np.clip(x, 0, 255.)
            x = x.astype(np.uint8)
            return x

        index_list = [ i for i in range(self.K)]
        prob = self.confounder_priority / torch.sum(self.confounder_priority) + 1e-3
        prob = np.array(prob.cpu())
        prob /= prob.sum()
        selected = np.random.choice(index_list, num, replace=False, p=prob)
        x_vs = self.confounder_queue[selected]
        x_vs = tensor2np(x_vs)
        for j in range(num):
            img = x_vs[j]
            Image.fromarray(img, mode='RGB').save(os.path.join(dir, f"{j}.png"))