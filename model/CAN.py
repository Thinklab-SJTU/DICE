from torch import nn
import torch.nn.functional as F
import torch
from utils.config import cfg
from utils.utils import clamp, normalize
import math

class ConfAttNetwork(nn.Module):
    """ an API for Backdoor Intervention Approximation
    Params:
        embedding_size (int): size of embedding
        representation_size (int): size of flattened input tensor
        dim_rep (int): num of dimensions of input tensor 
        prior (tensor): sampling distribution of confounders 
    """
    def __init__(self):
        super(ConfAttNetwork, self).__init__()
        self.embedding_size = int(cfg.MODEL.ATTN_SIZE)
        if cfg.MODEL.CONF_MODE == 'input':
            representation_size = int(cfg.DATASET.SIZE_H) * int(cfg.DATASET.SIZE_W) * int(cfg.DATASET.NUM_CHANNEL)
            dim_rep = 3
        elif cfg.MODEL.CONF_MODE == 'rep':
            representation_size = cfg.MODEL.HID_CHANNELS
            dim_rep = 1

        self.Ws = nn.Linear(representation_size, self.embedding_size)
        self.Wv = nn.Linear(representation_size, self.embedding_size)

        nn.init.normal_(self.Ws.weight, std=0.02)
        nn.init.normal_(self.Wv.weight, std=0.02)
        nn.init.constant_(self.Ws.bias, 0)
        nn.init.constant_(self.Wv.bias, 0)

        self.feature_size = representation_size
        self.dim_rep = dim_rep
        # prior: uniform distribution
        if cfg.MODEL.CONF_PRIOR_DIST == 'uniform':
            self.prior = 1. / self.embedding_size
        else:
            raise ValueError('')

        if cfg.TRAIN.GRAD_PRE_WAY == 'std':
            self.grad_pre_func = self.grad_std
        elif cfg.TRAIN.GRAD_PRE_WAY == 'linear':
            self.grad_pre_func = self.grad_linear
        else:
            raise ValueError('wrong arguments for gradient preprocessing way.')            

        if cfg.TRAIN.GRAD_MASK_WAY == 'hard':
            self.grad_mask_func = self.grad_hard_mask
        elif cfg.TRAIN.GRAD_PRE_WAY == 'soft':
            self.grad_mask_func = self.grad_soft_mask
        else:
            raise ValueError('wrong arguments for gradient-based masking')
        
    def forward(self, s, v_set, no_att=False, norm=False):
        v_att = self.v_dic(s, v_set, no_att, norm)
        return v_att
        
    def v_dic(self, s, v_set, no_att, norm=False):
        length = v_set.size(1)
        if length == 1:
            return v_set.sum(dim=1)
        if no_att:
            v = self.prior * v_set.sum(dim=1)
            return v

        if norm:
            attention = torch.einsum('bd,bcd->bc', self.Ws(normalize(s).flatten(1)), self.Wv(normalize(v_set).flatten(2))) / (self.embedding_size ** 0.5)
        else:
            attention = torch.einsum('bd,bcd->bc', self.Ws(s.flatten(1)), self.Wv(v_set.flatten(2))) / (self.embedding_size ** 0.5)
    
        attention = F.softmax(attention, 1)

        # first broadcast attention map to the confounder size before applying mask
        v_hat = attention[(...,) + (None,) * self.dim_rep] * v_set

        if self.prior is None:
            v = v_hat.sum(dim=1)
        else:
            v = self.prior * v_hat.sum(dim=1)
        return v

# ------------------------------------------------------------------------------
# Gradient-based Attention Module 
# ------------------------------------------------------------------------------
    @staticmethod
    def grad_linear(grad, abs=True):
        if abs:
            grad = grad.abs_()
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        return grad

    @staticmethod
    def grad_std(grad, abs=True, clip=False):
        if abs:
            grad = grad.abs_()
        # average gradients of pixels per channel per example
        # image shape: (B, C, H, W)
        grad_avg = torch.mean(grad, (2, 3), keepdim=True)
        grad = grad - grad_avg
        # cover ~99.73% of sample data points within 3 sigma
        std = 3 * torch.std(grad, (2, 3), keepdim=True)
        if clip:
            grad = clamp(grad, -std, std)
        grad = (1 + grad / std) * 0.5
        return grad
    
    @staticmethod
    def grad_hard_mask(grad, att_r=0.2, pooling='mean'):
        # average gradients across all channels
        if pooling == 'mean':
            grad = torch.mean(grad, 1)
        elif pooling == 'max':
            grad = torch.max(grad, 1)[0]
        else:
            raise ValueError('unrecoginized arguments for pooling gradients')
        grad_ = grad.flatten(start_dim=1).sort(dim=1)[0]
        size_g = grad.shape[1] * grad.shape[2] - 1
        threds = grad_[:, math.floor(att_r * size_g)]
        # import pdb; pdb.set_trace()
        mask = torch.where(grad > threds[..., None, None], 1., 0.)
        mask = mask.unsqueeze(1)
        mask = mask.detach()
        # return foreground mask, background mask
        return mask, 1 - mask

    @staticmethod
    def grad_soft_mask(grad, att_r=0.2, pooling='mean'):
        if pooling == 'mean':
            grad = torch.mean(grad, 1)
        elif pooling == 'max':
            grad = torch.max(grad, 1)
        else:
            raise ValueError('unrecoginized arguments for pooling gradients')
        grad = torch.mean(grad, 1)
        grad_ = grad.flatten(start_dim=1).sort(dim=1)[0]
        size_g = grad.shape[1] * grad.shape[2] - 1
        threds = grad_[:, math.floor(att_r * size_g)]
        mask = torch.where(grad > threds[..., None, None], grad, -1 * grad)
        mask = mask.unsqueeze(1)
        mask = mask.detach()
        return torch.sigmoid(mask), torch.sigmoid(-1 * mask)


        