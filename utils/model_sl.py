import torch
from torch.nn import DataParallel
from utils.config import cfg
from model.madry_model import WideResNet_Madry
from robustbench.utils import load_model as load_off_model

def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)

def load_model(model, path, strict=True):
    if isinstance(model, DataParallel):
        module = model.module
    else:
        module = model
    # import pdb; pdb.set_trace()
    missing_keys, unexpected_keys = module.load_state_dict(torch.load(path), strict=strict)
    if len(unexpected_keys) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))

def load_backbone(backbone_name):
    # backbone_name = cfg.ARCH
    model_name_list = {'MMA': 'Ding2020MMA','TRADES': 'Zhang2019Theoretically', 'Madry': 'Madry', 'MART': 'Wang2020Improving', 'AWP': 'Wu2020Adversarial'}
    if backbone_name == 'Madry':
        model_madry = WideResNet_Madry(depth=34, num_classes=cfg.DATASET.NUM_CLASSES, widen_factor=10, dropRate=0.0)
        checkpoint_madry = torch.load(cfg.MASK_BACKBONE_PATH)
        model_madry.load_state_dict(checkpoint_madry)
        model_madry.float()
        # model_madry.eval()
        model_sur = model_madry.cuda()
    else:
        model_sur = load_off_model(model_name_list[backbone_name], dataset='cifar10', threat_model='Linf')
        model_sur = model_sur.cuda()
        # model_sur = model_sur.eval()
    return model_sur