'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import random
from torch.autograd import Variable

from opacus import PrivacyEngine
from tqdm.notebook import tqdm
import pandas as pd
import random
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np

from RL_DP_Project.experiment.train_single_model import Experiment

from models import *
#from utils import progress_bar
from PIL import Image

import sys
sys.path.append("../")
from dup_stdout_manager import DupStdoutFileManager

class CIFAR_load(torch.utils.data.Dataset):
    def __init__(self, root, baseset, dummy_root='~/data', split='train', download=False, **kwargs):

        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split('.')[0])
        true_img, label = self.baseset[true_index]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx]))), label, true_img

def set_seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

def get_model():
    return ResNet18()

set_seed()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--load_path', type=str)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

with DupStdoutFileManager(os.path.join(args.load_path, "eval_log_DPSGD.log")) as _:
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    baseset = torchvision.datasets.CIFAR10(
        root='~/data', train=True, download=False, transform=transform_train)
    trainset = CIFAR_load(root=args.load_path, baseset=baseset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    accs = []

    for run in range(args.runs):
        criterion = nn.CrossEntropyLoss()
        e = Experiment(get_model, criterion, trainset, testset)
        results = e.run_experiment(1, 0.001)
        print()
        print("RESULTS:")
        _ = [print(key + ":", round(item, 4)) for key, item in results.items()]
        accs.append(results['val_acc'])
    print(accs)
    print(f'Mean accuracy: {np.mean(np.array(accs))}, \
                Std_error: {np.std(np.array(accs)) / np.sqrt(args.runs)}')
