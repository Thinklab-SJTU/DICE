"""The global config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.data_root = "./data"
__C.test_root = "./data"
__C.mask_data_root = "./data/mask_cifar_10"

# amp setting
__C.opt_level = 'O2'
__C.master_weights = 0
__C.loss_scale = 1.

# --------------------------------
# hyper-parameters for training (basic)
# --------------------------------

__C.batch_size = 128
__C.test_batch_size = 128
__C.val_num_examples = 1000
__C.start_epoch = 1
__C.epochs = 120
__C.weight_decay = 5e-4
__C.lr = 0.1
__C.momentum = 0.9
__C.no_cuda = False
# random seed
__C.seed = 1
# learning rate mode
__C.lr_mode = 'piecewise'

# dataset statistics
__C.cifar10_mean = (0.4914, 0.4822, 0.4465)
__C.cifar10_std = (0.2471, 0.2435, 0.2616)

__C.out_dir = 'results/'

# --------------------------------
# basic setting
# --------------------------------

__C.EVAL = False
__C.EVAL_MASK = False
__C.EVAL_BLACK = False

__C.DATASET_NAME = 'CIFAR_10'
# most ``PATH'' can be automatically initialized without being specified
__C.OUTPUT_PATH = ''
__C.SAVE_PATH = ''
__C.STAT_PATH = ''
__C.BACKBONE_PATH = ''
__C.PRETRAINED_PATH = ''
__C.ARCH = ''
# backone for DICE 
__C.BACK_ARCH = ''
__C.BLACK_ARCH = 'Standard'
__C.EVAL_ARCH = ''
__C.LOG_INTERVAL = 100
# frequency of evaluation
__C.EVAL_FREQ = 1
# when to start saving model
__C.SAVE_EPOCH = 60
# frequency of saving model 
__C.SAVE_FREQ = 10

# --------------------------------
# hyper-parameters for DATASET
# --------------------------------

__C.DATASET = edict()

__C.DATASET.NAME = 'CIFAR_10'
__C.DATASET.NUM_CLASSES = 10
__C.DATASET.SIZE_H = 32
__C.DATASET.SIZE_W = 32
__C.DATASET.NUM_CHANNEL = 3
__C.DATASET.MODE = 'clean'
# whether to normalize data or not 
__C.DATASET.NORM = False

# --------------------------------
# hyper-parameters for MODEL
# --------------------------------

__C.MODEL = edict()
# size of representation space
__C.MODEL.HID_CHANNELS = 512

__C.MODEL.DO_HID_CHANNELS = 256

# whether to mask foreground or not
__C.MODEL.X_S_WHOLE = True

# --------------------------------
# hyper-parameters for confounder bank 
# --------------------------------

# embedding size for attention
__C.MODEL.ATTN_SIZE = 20

# confounder bank size
__C.MODEL.CONF_BUFFER_SIZE = 4096

# confounder set size
__C.MODEL.CONF_SET_SIZE = 10

# whether to reduce the probability of samples with time 
__C.MODEL.CONF_DECAY = False

# which type of confounder to be used 
__C.MODEL.CONF_MODE = 'input'

# prior distribution type 
__C.MODEL.CONF_PRIOR_DIST = 'uniform'

# whether to store clean confounder or not 
__C.MODEL.CONF_CLEAN = True

# whether to store adv confounder or not 
__C.MODEL.CONF_ADV = True

# way of constituting x_do
__C.MODEL.CONF_ADV_ATT = False

# whether to do similarity-based reweighting of confounders or not 
__C.MODEL.CONF_NO_ATT = False

# --------------------------------
# hyper-parameters for Attack
# --------------------------------

__C.ATTACK = edict()

# perturbation budget for training
__C.ATTACK.EPSILON = 8.

# perturbation budget for evaluation
__C.ATTACK.EVAL_EPSILON = 8.

__C.ATTACK.TRADES_EPSILON = 8.

__C.ATTACK.Madry_EPSILON = 8.

# relative step size during iterative optimization
__C.ATTACK.ALPHA = 2.

__C.ATTACK.EVAL_ALPHA = 2.

__C.ATTACK.TRADES_ALPHA = 2.

__C.ATTACK.Madry_ALPHA = 2.

# number of steps 
__C.ATTACK.STEP = 10

__C.ATTACK.EVAL_STEP = 20

__C.ATTACK.TRADES_STEP = 10

__C.ATTACK.Madry_STEP = 10

# number of restarts 
__C.ATTACK.RESTARTS = 1

# which type of attack to be performed, optional: ['tar', 'tar_adap', 'untar']
__C.ATTACK.LOSS_TYPE = 'v_min_ce'

# how to select the target label for ``target'' attack, optional: ['mc', 'random']
__C.ATTACK.CONFOUNDING_WAY = 'mc'

# whether to normalize data to (-1, 1) or not 
__C.ATTACK.NORMALIZE = False

# which mode to perform random target attack
__C.ATTACK.RAND_MODE = 'mse'

# tunable parameter for the trade-off between ``target'' and ``untarget'' attack
__C.ATTACK.GAMMA = 0.8

# --------------------------------
# hyper-parameters for training (DICE)
# --------------------------------

__C.TRAIN = edict()

__C.TRAIN.NORM = False

__C.TRAIN.MODE = 'ST'

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [60, 90]

# regularization ratio of intervention loss 
__C.TRAIN.CAUSAL_REG = 1.

# how to build surrogate loss  
__C.TRAIN.CAUSAL_SURRO = 'adv'

# way of normalizing gradients 
__C.TRAIN.GRAD_PRE_WAY = 'std'

# way of building gradient based masks
__C.TRAIN.GRAD_MASK_WAY = 'hard'

# ratio of gradients chosen for background
__C.TRAIN.GRAD_ATT_R = 0.2

# which way to pool gradients 
__C.TRAIN.GRAD_POOLING = 'max'

# whether to set BN mode as eval when crafting adversarial examples 
__C.TRAIN.BNeval = True

__C.TRAIN.VERBOSE = False

# For mask dataset, the frequency that we generate mask
__C.TRAIN.MASK_FREQ = 5

# For prior type
__C.TRAIN.PRIOR_TYPE = 'basic'

# AT MODE: TRADES or AT
__C.TRAIN.AT_MODE = 'AT'

__C.TRAIN.CAUSAL_MODE = 'AT'

# beta for TRADES loss
__C.TRAIN.TRADES_BETA = 1.0

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            if type(b[k]) is float and type(v) is int:
                v = float(v)
            else:
                if not k in ['CLASS']:
                    raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

