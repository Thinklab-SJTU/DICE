import os
import torch
import xlwt
from torch.nn import DataParallel
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from utils.parse_args import parse_args
from utils.print_easydict import print_easydict
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.config import cfg
from utils.model_sl import load_model, save_model

args = parse_args('Beginning.')

from util_func import StatLogger, set_seed, transform_train, transform_test, make_model, transform_train_norm, transform_test_norm
from train import train, train_causal_poison, train_causal_adv, train_causal_attack, train_adv, eval_adv

def main(model, device, train_loader, optimizer, scheduler, tfboardwriter, loss_logger, acc_logger):
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        model.train()
        if cfg.TRAIN.MODE == 'causal_poison':
            lossdict, accdict = train_causal_poison(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'causal_adv':
            lossdict, accdict = train_causal_adv(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'causal_attack':
            lossdict, accdict = train_causal_attack(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'adv':
            lossdict, accdict = train_adv(model, device, train_loader, optimizer, epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'baseline':
            lossdict, accdict = train(model, device, train_loader, optimizer, epoch, tfboardwriter)
        else:
            raise ValueError('unrecognized arguments for train mode.')

        if isinstance(scheduler, list):
            for sche in scheduler:
                sche.step()
        else:
            scheduler.step()

        # log current epoch-level statistics
        loss_logger.add(lossdict)
        acc_logger.add(accdict)

        # save checkpoints
        if epoch >= cfg.SAVE_EPOCH and epoch % cfg.SAVE_FREQ == 0:
            save_model(model, os.path.join(cfg.SAVE_PATH, 'epoch{}.pt'.format(epoch)))
            if isinstance(optimizer, list):
                opt_names = ['causal', 'conf']
                for name, opt in zip(opt_names, optimizer):
                    torch.save(opt.state_dict(),
                                os.path.join(cfg.SAVE_PATH, '{}_opt_epoch{}.tar'.format(name, epoch)))
            else:
                torch.save(optimizer.state_dict(),
                            os.path.join(cfg.SAVE_PATH, 'opt_epoch{}.tar'.format(epoch)))

        #sample and save image from conf bank
        if 'causal' in cfg.TRAIN.MODE and  (epoch == 1  or epoch % cfg.SAVE_FREQ == 0):
            banksample_path = os.path.join(cfg.OUTPUT_PATH, f'confbank_sample')
            if not os.path.exists(banksample_path):
                os.mkdir(banksample_path)
            banksample_path = os.path.join(banksample_path, f"epoch{epoch}")
            if not os.path.exists(banksample_path):
                os.mkdir(banksample_path)
            if isinstance(model, DataParallel):
                model.module.erb.sample_and_save(banksample_path, num=10)
            else:
                model.erb.sample_and_save(banksample_path, num=10)

    # post-training
    loss_logger.save()
    acc_logger.save()

def evaluation(model=None):
    eval_adv(model, device, test_loader, 0, tfboardwriter, 'Test', att='untar')
    eval_adv(model, device, test_loader, 0, tfboardwriter, 'Test', att='cw')


if __name__ == '__main__':
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wb = xlwt.Workbook()
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / (now_time + '.xls'))
    loss_logger = StatLogger()
    acc_logger = StatLogger()

    set_seed()
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    if cfg.DATASET.NORM:
        transform_train = transform_train_norm
        transform_test = transform_test_norm

    if cfg.DATASET.NAME == 'CIFAR_100':
        data_set = datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform_train)
    else:
        data_set = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform_train)

    train_set, val_set = random_split(data_set, [len(data_set) - cfg.val_num_examples, cfg.val_num_examples],
                                    generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **kwargs)

    if cfg.DATASET.NAME == 'CIFAR_100':
        testset = datasets.CIFAR100(root=cfg.test_root, train=False, download=True, transform=transform_test)
    else:
        testset = datasets.CIFAR10(root=cfg.test_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, **kwargs)

    model = make_model(cfg.ARCH)
    model = model.cuda()

    if cfg.ARCH == 'causal_v0':
        causal_opt = optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
        conf_opt = optim.SGD(params=model.conf_mlp.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
        optimizers = [causal_opt, conf_opt]
    else:
        optimizers = optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    optimizer = optimizers
    model = torch.nn.DataParallel(model)

    model_path, optim_path = '', ''
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)

    if cfg.lr_mode == 'cyclic':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_mode == 'piecewise':
        if isinstance(optimizer, list):
            scheduler = []
            for opt in optimizer:
                sche = optim.lr_scheduler.MultiStepLR(opt,
                                                        milestones=cfg.TRAIN.LR_STEP,
                                                        gamma=cfg.TRAIN.LR_DECAY,
                                                        last_epoch=cfg.TRAIN.START_EPOCH - 1)
                scheduler.append(sche)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.TRAIN.LR_STEP,
                                                        gamma=cfg.TRAIN.LR_DECAY,
                                                        last_epoch=cfg.TRAIN.START_EPOCH - 1,
                                                        verbose=cfg.TRAIN.VERBOSE)

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        if cfg.EVAL:
            evaluation(model)
        else:
            main(model, device, train_loader,
                optimizer, scheduler, tfboardwriter,
                loss_logger, acc_logger)