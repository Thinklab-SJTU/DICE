import torch.nn as nn
import torch
from torch.autograd import grad
from utils.utils import normalize
from utils.config import cfg
from torch.nn import Linear, ReLU, CrossEntropyLoss
from model.CAN import ConfAttNetwork
from model.ConfBank import ConfounderBank


class CausalRLNetwork(nn.Module):
    def __init__(self, backbone=None):
        super(CausalRLNetwork, self).__init__()
        self.backbone = backbone
        self.can = ConfAttNetwork()
        self.erb = ConfounderBank()
        self.hid_channels = cfg.MODEL.HID_CHANNELS
        self.causal_mlp = torch.nn.Sequential(
            Linear(self.hid_channels, 2*self.hid_channels),
            ReLU(),
            Linear(2*self.hid_channels, cfg.DATASET.NUM_CLASSES)
        )
        self.conf_mlp = torch.nn.Sequential(
            Linear(self.hid_channels, 2*self.hid_channels),
            ReLU(),
            Linear(2*self.hid_channels, cfg.DATASET.NUM_CLASSES)
        )
        self.CELoss = CrossEntropyLoss(reduction="mean")
        self.EleCELoss = CrossEntropyLoss(reduction="none")
        self.x_s_whole = cfg.MODEL.X_S_WHOLE
        self.conf_clean = cfg.MODEL.CONF_CLEAN
        self.att_r = cfg.TRAIN.GRAD_ATT_R
        self.pooling = cfg.TRAIN.GRAD_POOLING
        self.mode = cfg.TRAIN.MODE
        self.att_adv = cfg.MODEL.CONF_ADV_ATT
        self.no_att = cfg.MODEL.CONF_NO_ATT

    def get_causal_rep(self, x_do, norm=False):
        if norm:
            return self.backbone.get_rep(normalize(x_do))
        else:
            return self.backbone.get_rep(x_do)

    def get_conf_rep(self, x_v, norm=False):
        if norm:
            return self.backbone.get_rep(normalize(x_v))
        else:
            return self.backbone.get_rep(x_v)

    def get_causal_pred(self, z_s):
        return self.causal_mlp(z_s)

    def get_conf_pred(self, z_v):
        return self.conf_mlp(z_v)

    def forward(self, x, norm=False):
        z_x = self.get_causal_rep(x, norm)
        preds = self.get_causal_pred(z_x)
        return preds

    def forward_m(self, x, norm=False):
        x = self.mask_x(x, norm)
        z_x = self.get_causal_rep(x, norm)
        preds = self.get_causal_pred(z_x)
        return preds

    def mask_x(self, x, norm=False):
        x_v_set = self.erb.batch_sample_set(x)
        x_v_att = self.can(x, x_v_set, self.no_att, norm)
        return x + x_v_att.detach()

    def get_delta_x(self, x, y, norm=False):
        x.requires_grad = True
        if norm:
            pred = self.forward(normalize(x))
        else:
            pred = self.forward(x)
        loss = self.CELoss(pred, y)
        delta_x = grad(loss, x)[0].detach()
        x.requires_grad = False
        return delta_x

    def get_mask_x(self, x, y):
        delta_x = self.get_delta_x(x, y)
        grad_norm = self.can.grad_pre_func(delta_x)
        mask_fg, mask_bg = self.can.grad_mask_func(grad_norm, att_r=self.att_r, pooling=self.pooling)
        return mask_fg, mask_bg

    def split_x(self, x, y, conf_prior=False, norm=False):
        # 1. obtain gradients
        delta_x = self.get_delta_x(x, y, norm)
        # 2. normalize gradients 
        grad_norm = self.can.grad_pre_func(delta_x)
        # 3. obtain gradient-based attention map
        mask_fg, mask_bg = self.can.grad_mask_func(grad_norm, att_r=self.att_r, pooling=self.pooling)

        if self.x_s_whole:
            x_s = x
        else:
            x_s = mask_fg * x
        
        x_v = mask_bg * x
        x_v = x_v.detach()

        # 4. add one batch of x_v into memory buffer
        if conf_prior: # set the sampling priority based on responses of model to background.
            conf_rep = self.get_conf_rep(x_v, norm)
            conf_pred = self.get_conf_pred(conf_rep)
            self.erb.add(x_v, conf_pred, y)
        elif self.conf_clean:
            self.erb.add(x_v)

        # 5. sample a batch of confounder sets, each x_s has a confounder set with size of N, so the returned result has the shape (B, N, ...)
        x_v_set = self.erb.batch_sample_set(x_s)

        # 6. approximate causal intervention using our causal attention network
        if self.att_adv:
            x_v_att = self.can(x_s+delta_x, x_v_set, self.no_att, norm)
        else:
            x_v_att = self.can(x_s, x_v_set, self.no_att, norm)

        # x_v_att = x_v_att.detach()

        if conf_prior:
            return x_s, x_v, x_v_att, conf_pred, delta_x
        else:
            return x_s, x_v, x_v_att, delta_x