import torch
from utils.config import cfg

mu = torch.tensor(cfg.cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cfg.cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
