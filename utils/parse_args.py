import argparse
from utils.config import cfg, cfg_from_file, cfg_from_list
from pathlib import Path
from datetime import datetime
import os 

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--arch', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--clean_data_path', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='start epoch number for resume training', default=None, type=int)
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha', default=None, type=float)
    parser.add_argument('--num_iter', dest='num_iter',
                        help='iteration number', default=None, type=int)
    parser.add_argument('--eval_num_iter', 
                        help='eval iteration number', default=None, type=int)
    parser.add_argument('--eps', default=None, type=float)
    parser.add_argument('--loss_type', default=None, type=str)
    parser.add_argument('--sim_type', default=None, type=str)
    parser.add_argument('--datetime', default=None, type=str)
    parser.add_argument('--eval_epoch', default=None, type=int)
    parser.add_argument('--eval_per_num_epoch', default=None, type=int)
    parser.add_argument('--vis_dataset', default=None, type=str, choices=['voc', 'imcpt', 'willow', 'cub'])
    parser.add_argument('--prefix', default=None, type=str, help='prefix to exp dir name')
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument('--mode', default=None, type=str, help='which mode to be used during train-time', choices=['ST', 'AT'])
    parser.add_argument('--constraint', default='Linf', type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.loss_type is not None:
        cfg_from_list(['ATTACK.LOSS_TYPE', args.loss_type])
    if args.sim_type is not None:
        cfg_from_list(['ATTACK.SIM_TYPE', args.sim_type])
    if args.arch is not None:
        cfg_from_list(['ARCH', args.arch])

    # update cfg from command arguments
    if args.batch_size is not None:
        cfg_from_list(['batch_size', args.batch_size])
    if args.start_epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.start_epoch])
    if args.eval_epoch is not None:
        cfg_from_list(['EVAL.EPOCH', args.eval_epoch])
    if args.eval_per_num_epoch is not None:
        cfg_from_list(['EVAL.NUM_EPOCH', args.eval_per_num_epoch])
    if args.alpha is not None:
        cfg_from_list(['ATTACK.ALPHA', args.alpha])
    if args.num_iter is not None:
        cfg_from_list(['ATTACK.STEP', args.num_iter])
    if args.eval_num_iter is not None:
        cfg_from_list(['ATTACK.EVAL_STEP', args.eval_num_iter])        
    if args.eps is not None:
        cfg_from_list(['ATTACK.EPSILON', args.eps])
        cfg_from_list(['ATTACK.EVAL_EPSILON', args.eps])
    if args.model_path is not None:
        cfg_from_list(['MODEL_PATH', args.model_path])
    if args.mode is not None:
        cfg_from_list(['TRAIN.MODE', args.mode])

    if args.vis_dataset is not None:
        sum2fullname_dict = {
        'voc': 'PascalVOC',
        'imcpt': 'IMC_PT_SparseGM',
        'willow': 'WillowObject',
        'cub': 'CUB2011'}
        cfg.DATASET_FULL_NAME = sum2fullname_dict[args.vis_dataset]
        cfg.MODULE = None
    
    if args.exp_name is None:
        if cfg.TRAIN.MODE == 'causal':
            causal_str = 'reg_{}'.format(cfg.TRAIN.CAUSAL_REG)
            exp_name = '{}_{}_on_{}_{}'.format(args.prefix, cfg.TRAIN.MODE, cfg.BACK_ARCH, causal_str)
        elif cfg.TRAIN.MODE == 'causal_attack' or cfg.TRAIN.MODE == 'causal_adv':
            causal_str = 'reg_{}'.format(cfg.TRAIN.CAUSAL_REG)
            att_str = '{}_eps_{}_alp_{}_s_{}'.format(cfg.ATTACK.LOSS_TYPE, cfg.ATTACK.EPSILON, cfg.ATTACK.ALPHA, cfg.ATTACK.STEP)
            exp_name = '{}_{}_on_{}_{}_{}'.format(args.prefix, cfg.TRAIN.MODE, cfg.BACK_ARCH, causal_str, att_str)
        elif cfg.TRAIN.MODE == 'adv':
            att_str = '{}_eps_{}_alp_{}_s_{}'.format(cfg.ATTACK.LOSS_TYPE, cfg.ATTACK.EPSILON, cfg.ATTACK.ALPHA, cfg.ATTACK.STEP)
            exp_name = '{}_{}_on_{}_{}'.format(args.prefix, cfg.TRAIN.MODE, cfg.ARCH, att_str)
        elif cfg.TRAIN.MODE == 'vis':
            causal_str = 'grad_{}_{}_{}_r_{}'.format(cfg.TRAIN.GRAD_PRE_WAY, cfg.TRAIN.GRAD_MASK_WAY, cfg.TRAIN.GRAD_POOLING, cfg.TRAIN.GRAD_ATT_R)
            exp_name = '{}_{}_on_{}_{}'.format(args.prefix, cfg.TRAIN.MODE, cfg.ARCH, causal_str)
        elif cfg.TRAIN.MODE == 'debug':
            exp_name = '{}_debug'.format(args.prefix)

    if args.datetime is None:
        now_day = datetime.now().strftime('%Y-%m-%d')

    outp_path = os.path.join('output', now_day, exp_name)
    save_path = os.path.join(outp_path, 'params')
    stat_path = os.path.join(outp_path, 'stats')
    cfg_from_list(['OUTPUT_PATH', outp_path])
    cfg_from_list(['SAVE_PATH', save_path])    
    cfg_from_list(['STAT_PATH', stat_path])
    if not Path(cfg.SAVE_PATH).exists():
        Path(cfg.SAVE_PATH).mkdir(parents=True)
    if not Path(cfg.STAT_PATH).exists():
        Path(cfg.STAT_PATH).mkdir(parents=True)
    return args