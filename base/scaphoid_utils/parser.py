import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,  help='yaml config file')
    parser.add_argument('--config_folder', type=str,
                        default='./cfgs', help='config folder')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic',
                        action='store_true',
                        default=False,
                        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument('--sync_bn',
                        action='store_true',
                        default=False,
                        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type=str,
                        default='default', help='experiment name')
    parser.add_argument('--start_ckpts', type=str,
                        default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None,
                        help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='autoresume training (interrupted by accident)')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='test mode for certain ckpt')
    parser.add_argument('--mode',
                        choices=['easy', 'median', 'hard', None],
                        default=None,
                        help='difficulty mode for shapenet')
    parser.add_argument('--pretrained', type=str,
                        default=None, help='pretrained model path')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--strict',
                        choices=['True', 'False'],
                        default='False',
                        help='strict mode for dataset loading')
    parser.add_argument('--partly',
                        choices=['dorsal', 'volar'],
                        default=None,
                        help='partly mode for dataset loading, only used in RotationDataset')
    parser.add_argument('--smac3', action='store_true',
                        default=False, help='smac3 mode for hyperparameter optimization')
    parser.add_argument('--wandb', type=str, default="Master Thesis", help='WandB project name')

    args, _ = parser.parse_known_args()

    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')


    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode

    if args.strict == 'True':
        args.strict = True
    elif args.strict == 'False':
        args.strict = False
    
    return args
