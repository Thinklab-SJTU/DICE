import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from utils.config import cfg

mu = torch.tensor(cfg.cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cfg.cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1, 0
restarts = cfg.ATTACK.RESTARTS 
epsilon = int(cfg.ATTACK.EPSILON)/255.
attack_iters = cfg.ATTACK.STEP
alpha = int(cfg.ATTACK.ALPHA)/255.

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def CW_loss(outputs, y, reduction='mean'):
    x_sorted, ind_sorted = outputs.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    # max loss = - (y_gt - y_mc)
    loss_value = -(outputs[np.arange(outputs.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def CE_loss(outputs, y, reduction='mean'):
    loss_value = F.cross_entropy(outputs, y, reduction=reduction)
    return loss_value

def Trades_loss(adv_out, natural_out):
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss_kl = criterion_kl(F.log_softmax(adv_out, dim=1),
                           F.softmax(natural_out, dim=1))
    return loss_kl

MSE_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
KL_loss = nn.KLDivLoss()

def hard_label_mc_target(outputs, y):
    y_t = torch.zeros((y.shape[0]), dtype=torch.int64).to(y.device)
    if isinstance(outputs, dict):
        outputs = outputs['logits']
    # import pdb; pdb.set_trace()
    ind_sorted = outputs.sort(dim=-1)[1]
    # index for those correctly classified.
    ind = ind_sorted[:, -1] == y
    true_idcs = torch.nonzero(ind).squeeze(1)
    false_idcs = torch.nonzero(~ind).squeeze(1)
    y_t[true_idcs] = ind_sorted[:, -2][true_idcs]
    y_t[false_idcs] = ind_sorted[:, -1][false_idcs]
    # y_t = torch.from_numpy(y_t).type(torch.int64).to(y.device)
    return y_t

def hard_label_rand_target(outputs, y, num_cls=10):
    y_t = np.zeros((y.shape[0]))
    for i in np.arange(y.shape[0]):
        tmp = np.ones([num_cls]) / (num_cls-1)
        tmp[y[i]] = 0.0
        y_t[i] = np.random.choice(num_cls, p=tmp)
    y_t = torch.from_numpy(y_t).type(torch.int64).to(y.device)
    return y_t

def tar_loss(outputs, y, reduction='mean'):
    if cfg.ATTACK.CONFOUNDING_WAY == 'mc':
        y_tar = hard_label_mc_target(outputs, y)
    elif cfg.ATTACK.CONFOUNDING_WAY == 'random':
        y_tar = hard_label_rand_target(outputs, y, num_cls=cfg.DATASET.NUM_CLASSES)
    else:
        raise ValueError
    return -1 * F.cross_entropy(outputs, y_tar, reduction=reduction)

def adap_loss(outputs, outputs_do, y, reduction='mean'):
    loss_s = F.cross_entropy(outputs, y, reduction=reduction)
    loss_do = F.cross_entropy(outputs_do, y, reduction=reduction)
    return loss_s + cfg.TRAIN.CAUSAL_REG * loss_do

def tar_adap_loss(epoch):
    beta_ = cfg.ATTACK.GAMMA * epoch / cfg.epochs
    def cal_loss(outputs, y):
        loss = (1. - beta_) * F.cross_entropy(outputs, y) + beta_ * tar_loss(outputs, y)
        return loss
    return cal_loss

class att_object:
    def __init__(self, att_key=None, eps=None, bs=None, norm='l_inf', dtype=torch.float32, device=None):
        self.att_key = att_key
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.bs = bs
        self.delta = None
        self.max_delta = None
        self.max_loss = torch.zeros(bs, device=device)
        self.delta_shape = None
        self.delta_grads = None
        self.norm = norm

    def init_max_delta(self, input):
        self.delta_shape = input.shape
        self.max_delta = torch.zeros(self.delta_shape, device=self.device)

    def init_delta(self):
        self.delta = torch.zeros(self.delta_shape, device=self.device)
        if self.norm == 'l_inf':
            self.delta.uniform_(-self.eps, self.eps)
        elif self.norm == 'l_2':
            self.delta.normal_()
            d_flat = self.delta.view(self.delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(self.delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            self.delta *= r / n * self.eps
        else:
            raise ValueError
        self.delta.requires_grad = True 
        # self.delta = clamp(self.delta, lower_limit-input, upper_limit-input)

def attack_pgd(model, X, y, norm='l_inf', 
               early_stop=False, early_stop_pgd_max=1,
               BNeval=False, loss_type=None, eval=False, norm_input=False,
               alpha=None, attack_iters=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    if loss_type is None:
        loss_type = cfg.ATTACK.LOSS_TYPE

    restarts = cfg.ATTACK.RESTARTS 
    if alpha is None and attack_iters is None:
        if eval:
            epsilon = int(cfg.ATTACK.EVAL_EPSILON)/255.
            attack_iters = cfg.ATTACK.EVAL_STEP
            alpha = int(cfg.ATTACK.EVAL_ALPHA)/255.        
        else:
            epsilon = int(cfg.ATTACK.EPSILON)/255.
            attack_iters = cfg.ATTACK.STEP
            alpha = int(cfg.ATTACK.ALPHA)/255.
    else:
        alpha = alpha/255.        
        epsilon = int(cfg.ATTACK.EPSILON)/255.
        attack_iters = attack_iters

    if loss_type == 'untar':
        criterion = CE_loss
    elif loss_type == 'cw' :
        criterion = CW_loss
    elif loss_type == 'tar' :
        criterion = tar_loss
    elif loss_type == 'trades':
        criterion = Trades_loss
    elif loss_type == 'adap' or loss_type == 'adap_m':
        criterion = adap_loss
    else:
        raise ValueError("Invalid loss type")

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # early stop pgd counter for each x
        early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).cuda()

        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            if norm_input:
                output = model(normalize(X + delta))
            else:
                if loss_type == 'adap' or loss_type == 'adap_m':
                    x_s, x_v, x_v_att, _ = model.split_x(X, y)
                    x_do = x_s + x_v_att.detach()
                    x_do = torch.clamp(x_do, 0., 1.)
                    output_do = model(x_do+delta)
                    output = model(X+delta)
                else:
                    output = model(X + delta)
            # if use early stop pgd
            if early_stop:
                # calculate mask for early stop pgd
                if_success_fool = (output.max(1)[1] != y).to(dtype=torch.int32)
                early_stop_pgd_count = early_stop_pgd_count - if_success_fool
                index = torch.where(early_stop_pgd_count > 0)[0]
                iter_count[index] = iter_count[index] + 1
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            if loss_type == 'adap' or loss_type == 'adap_m':
                loss = criterion(output, output_do, y)
            else:
                loss = criterion(output, y)
            loss.backward()
            grad = delta.grad.detach()

            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if norm_input:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if BNeval:
        model.train()

    return max_delta

def attack_pgd_on_conf(model, X, y, norm='l_inf', BNeval=False, loss_type=None):
    if loss_type is None:
        loss_type = cfg.ATTACK.LOSS_TYPE

    restarts = cfg.ATTACK.RESTARTS 
    # import pdb; pdb.set_trace()
    def updateclip(delta, grad, x):
        d = delta.detach()
        g = grad.detach()
        # import pdb; pdb.set_trace()
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        # import pdb; pdb.set_trace()
        # delta.grad.zero_()
        grad.zero_()

    if loss_type == 'untar':
        criterion = CE_loss
    elif loss_type == 'cw':
        criterion = CW_loss
    elif loss_type == 'tar':
        criterion = tar_loss
    elif loss_type == 'trades':
        criterion = Trades_loss
    else:
        raise ValueError("Invalid loss type")

    device = next(model.parameters()).device
    att_obj_dict = OrderedDict()
    att_key_list = ['bg']
    for i, att_key in enumerate(att_key_list):
        att_obj_dict[att_key] = att_object(att_key, epsilon, X[att_key].shape[0], device=device)
        att_obj_dict[att_key].init_max_delta(X[att_key])

    if BNeval:
        model.eval()

    for _ in range(restarts):    
        # initialize perturbation
        for att_obj in att_obj_dict.values():
            att_obj.init_delta()

        # craft adversarial examples
        for _ in range(attack_iters):

            conf_rep = model.get_conf_rep(X['bg']+att_obj_dict['bg'].delta)
            conf_pred = model.get_conf_pred(conf_rep)
            # conf_loss = F.cross_entropy(conf_pred, y)
            if loss_type == 'trades':
                conf_rep_natural = model.get_conf_rep(X['bg'])
                conf_pred_natural = model.get_conf_pred(conf_rep_natural)
                conf_loss = criterion(conf_pred, conf_pred_natural)
            else:
                conf_loss = criterion(conf_pred, y)
            # do_pred = model(X['fg']+att_obj_dict['fg'].delta)
            # do_loss =  F.cross_entropy(do_pred, y)
            
            total_loss = conf_loss

            delta_grads = torch.autograd.grad(total_loss, [att_obj.delta for att_obj in att_obj_dict.values()], create_graph=False)

            for i, att_obj in enumerate(att_obj_dict.values()):
                att_obj.delta_grads = delta_grads[i]

            for att_key, att_obj in att_obj_dict.items():
                updateclip(att_obj.delta, att_obj.delta_grads, X[att_key])
                # print('key {} min {} max {}'.format(att_key, att_obj.delta.min().item(), att_obj.delta.max().item()))
        all_loss = F.cross_entropy(model(X['bg']+att_obj_dict['bg'].delta), y, reduction='none')
        
        for att_key, att_obj in att_obj_dict.items():
            att_obj.max_delta[all_loss >= att_obj.max_loss] = att_obj.delta.detach()[all_loss >= att_obj.max_loss]
            att_obj.max_loss = torch.max(att_obj.max_loss, all_loss)

    if BNeval:
        model.train()

    return {'bg': att_obj_dict['bg'].max_delta}

def attack_pgd_on_both(model, X, y, norm='l_inf', 
               early_stop=False, early_stop_pgd_max=1,
               BNeval=False, loss_type=None, eval=False, norm_input=False,
               alpha=None, attack_iters=None, epoch=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta_bg = torch.zeros_like(X['bg']).cuda()
    max_delta_fg = torch.zeros_like(X['fg']).cuda()

    if loss_type is None:
        loss_type = cfg.ATTACK.LOSS_TYPE

    restarts = cfg.ATTACK.RESTARTS 
    if alpha is None and attack_iters is None:
        if eval:
            epsilon = float(cfg.ATTACK.EVAL_EPSILON)/255.
            attack_iters = cfg.ATTACK.EVAL_STEP
            alpha = float(cfg.ATTACK.EVAL_ALPHA)/255.        
        elif loss_type == 'trades':
            epsilon = float(cfg.ATTACK.TRADES_EPSILON)/255.
            attack_iters = cfg.ATTACK.TRADES_STEP
            alpha = float(cfg.ATTACK.TRADES_ALPHA)/255.
        else:
            epsilon = float(cfg.ATTACK.EPSILON)/255.
            attack_iters = cfg.ATTACK.STEP
            alpha = float(cfg.ATTACK.ALPHA)/255.
    else:
        alpha = alpha/255.        
        epsilon = int(cfg.ATTACK.EPSILON)/255.
        attack_iters = attack_iters

    if loss_type == 'untar':
        criterion = CE_loss
    elif loss_type == 'cw' :
        criterion = CW_loss
    elif loss_type == 'tar':
        criterion = tar_loss
    elif loss_type == 'trades':
        criterion = Trades_loss
    elif loss_type == 'tar_adap':
        criterion = tar_adap_loss(epoch)

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # early stop pgd counter for each x
        early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).cuda()

        # initialize perturbation
        delta_bg = torch.zeros_like(X['bg']).cuda()
        delta_fg = torch.zeros_like(X['fg']).cuda()
        if norm == "l_inf":
            delta_bg.uniform_(-epsilon, epsilon)
            delta_fg.uniform_(-epsilon, epsilon)
        # elif norm == "l_2":
        #     delta.normal_()
        #     d_flat = delta.view(delta.size(0),-1)
        #     n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        #     r = torch.zeros_like(n).uniform_(0, 1)
        #     delta *= r/n*epsilon
        else:
            raise ValueError
        delta_bg = clamp(delta_bg, lower_limit-X['bg'], upper_limit-X['bg'])
        delta_bg.requires_grad = True
        delta_fg = clamp(delta_fg, lower_limit-X['fg'], upper_limit-X['fg'])
        delta_fg.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            conf_rep = model.get_conf_rep(X['bg']+delta_bg, norm=norm_input)
            conf_pred = model.get_conf_pred(conf_rep)
            if loss_type == 'trades':
                conf_rep_natural = model.get_conf_rep(X['bg'], norm=norm_input)
                conf_pred_natural = model.get_conf_pred(conf_rep_natural)
                conf_loss = criterion(conf_pred, conf_pred_natural)
            else:
                conf_loss = criterion(conf_pred, y)
            output = model(X['fg'] + delta_fg, norm=norm_input)
            if loss_type == 'trades':
                causal_loss = criterion(output, model(X['fg'], norm=norm_input))
            else:
                causal_loss = criterion(output, y)

            # if use early stop pgd
            if early_stop:
                # calculate mask for early stop pgd
                if_success_fool = (output.max(1)[1] != y).to(dtype=torch.int32)
                early_stop_pgd_count = early_stop_pgd_count - if_success_fool
                index = torch.where(early_stop_pgd_count > 0)[0]
                iter_count[index] = iter_count[index] + 1
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            # Whether use mixup criterion
            #loss = F.cross_entropy(output, y)
            # loss = criterion(output, y)
            loss = conf_loss + causal_loss
            loss.backward()
            # deal with bg
            grad = delta_bg.grad.detach()
            d = delta_bg[index, :, :, :]
            g = grad[index, :, :, :]
            x = X['bg'][index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta_bg.data[index, :, :, :] = d
            delta_bg.grad.zero_()

            # deal with fg
            grad = delta_fg.grad.detach()
            d = delta_fg[index, :, :, :]
            g = grad[index, :, :, :]
            x = X['fg'][index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta_fg.data[index, :, :, :] = d
            delta_fg.grad.zero_()

        all_loss = F.cross_entropy(model(X['fg']+delta_fg, norm=norm_input), y, reduction='none')
        max_delta_bg[all_loss >= max_loss] = delta_bg.detach()[all_loss >= max_loss]
        max_delta_fg[all_loss >= max_loss] = delta_fg.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if BNeval:
        model.train()

    return {'fg': delta_fg, 'bg': delta_bg}