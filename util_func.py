import numpy as np
import time
import os
import torch
import random 
from torchvision import transforms

from utils.config import cfg
from model.vgg import VGG16, VGG19
from model.resnet import ResNet18, ResNet34, ResNet50
from model.wideresnet import WideResNet
from model.densenet import DenseNet121
from model.CRLv0 import CausalRLNetwork
from model.madry_model import WideResNet_Madry

robust_models = ['MMA', 'TRADES', 'Madry', 'AWP', 'MART']

mu = torch.tensor(cfg.cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cfg.cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train_norm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cfg.cifar10_mean, cfg.cifar10_std)
])

transform_test_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.cifar10_mean, cfg.cifar10_std)
])

def set_seed():
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)    

def make_model(arch):
    if arch == 'VGG16':
        model = VGG16()
    elif arch == 'VGG19':
        model = VGG19()
    elif arch == 'ResNet18':
        model = ResNet18()
    elif arch == 'ResNet34':
        model = ResNet34()
    elif arch == 'ResNet50':
        model = ResNet50()
    elif arch == 'DenseNet121':
        model = DenseNet121()
    # elif arch == 'VAEv2':
    #     model = VAEv2(latent_dim=256)
    # elif arch == 'VAEv3':
    #     model = VAEv3(latent_dim=256)
    elif arch == 'Madry':
        model = WideResNet_Madry(depth=34, num_classes=cfg.DATASET.NUM_CLASSES, widen_factor=10, dropRate=0.0)
    elif arch == 'causal_v0':
        backbone = make_model(cfg.BACK_ARCH)
        model = CausalRLNetwork(backbone)
    elif arch == 'WideResNet':
        model = WideResNet(34, cfg.DATASET.NUM_CLASSES, widen_factor=10, dropRate=0.0)
    else:
        raise ValueError('unrecognized Arguments for model arch {}'.format(arch))
    # if resume_path is not None:
    #     print('\n=> Loading checkpoint {}'.format(resume_path))
    #     checkpoint = torch.load(resume_path)
    #     # info_keys = ['epoch', 'train_acc', 'cln_val_acc', 'cln_test_acc', 'adv_val_acc', 'adv_test_acc']
    #     # info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
    #     # info = '. '.join(info_vals)
    #     # print(info)
    #     # model.load_state_dict(checkpoint['model'])
    #     model.load_state_dict(checkpoint)
    
    # model = torch.nn.DataParallel(model)
    # model = model.cuda()
    return model

class StatLogger:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = []

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = []
            self.__data[k].append(v)

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]]
        else:
            v_list = [self.__data[k] for k in keys]
            return tuple(v_list)

    def _get_array(self, *keys):
        if len(keys) == 1:
            return np.array(self.__data[keys[0]])
        else:
            v_list = [np.array(self.__data[k])[np.newaxis, :] for k in keys]
            return np.concatenate(v_list, axis=0)
    
    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

    def save(self, key=None, prefix=None):
        if key is None:
            # save all arrays
            for k in self.__data.keys():
                v = self._get_array(k)
                if prefix is not None:
                    np.save(os.path.join(cfg.STAT_PATH, '{}_{}.npy'.format(prefix, k)), v)
                else:
                    np.save(os.path.join(cfg.STAT_PATH, '{}.npy'.format(k)), v)
        else:
            v = self._get_array(key)
            if prefix is not None:
                np.save(os.path.join(cfg.STAT_PATH, '{}_{}.npy'.format(prefix, key)), v)
            else:
                np.save(os.path.join(cfg.STAT_PATH, '{}.npy'.format(key)), v)

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out