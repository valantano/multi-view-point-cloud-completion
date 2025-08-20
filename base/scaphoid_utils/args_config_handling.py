import os, sys
from pathlib import Path

from easydict import EasyDict
import torch

# Add the main directory to sys.path such that submodules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
poin_tr_dir = os.path.join(main_dir, 'submodules', 'PoinTr')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(poin_tr_dir)

from submodules.PoinTr.utils import misc

from base.scaphoid_utils import parser
from base.scaphoid_utils.config import ConfigHandler
from base.scaphoid_utils.logger import Logger, print_log, SimulLogger
from base.scaphoid_utils.scaphoid_utils import create_experiment_dir


def replace_config_with_priority(config: EasyDict, priority_config: dict, logger = None):
    """
    Replace values in an EasyDict with values from a priority dictionary.
    If a key in the priority dictionary exists in the EasyDict, its value will be replaced.
    If a key in the priority dictionary does not exist in the EasyDict, a KeyError will be raised.
    :param config: EasyDict to be updated.
    :param priority_config: Dictionary with priority values.
    :param logger: Logger for logging information.
    :return: Updated EasyDict.
    :raises KeyError: If a key in the priority dictionary does not exist in the EasyDict.
    """
    for key, value in priority_config.items():
        if key not in config:
            raise KeyError(f"Key '{key}' not found in EasyDict with keys: {list(config.keys())}")

        print_log(f"------{type(value)} {type(config[key])} Updating key '{key}' with value: {value}", 
                  logger, color='yellow')
        if isinstance(value, dict):
            print_log(f"---------Updating nested key '{key}' with dict", color='yellow', logger=logger)
            replace_config_with_priority(config[key], value)
        else:
            print_log(f"---------Replacing '{key}': {config[key]} → {value}", color='yellow', logger=logger)
            config[key] = value
    
    return config

    

def replace_args_with_priority(args, priority_args: dict):
    """
    Replace values in an EasyDict with values from a priority dictionary.
    If a key in the priority dictionary exists in the EasyDict, its value will be replaced.
    """
    for key, value in priority_args.items():
        if key in args:
            print_log(f"------Replacing {key} in EasyDict with value: {value}", color='yellow')
            setattr(args, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in EasyDict with keys: {args.keys()}")
    return args


def handle_args_and_config(priority_args=None, priority_config=None):
    # args
    print_log(f"Handling args and config... #####################", color='blue')
    print_log(f"---Priority args: {priority_args}, type={type(priority_args)}", color='blue')
    print_log(f"---Priority config: {priority_config}, type={type(priority_config)}", color='blue')
    args: EasyDict = parser.get_args()
    if priority_args is not None:
        args = replace_args_with_priority(args, priority_args)


    ###### Experiment Path ############################################################################################
    print_log(f"---args.config: {args.config}", color='blue')
    exp_batch_path = os.path.join('./mvpcc_experiments', Path(str(args.config)).parent.stem)
    args.experiment_path = os.path.join(exp_batch_path, Path(args.config).stem, args.exp_name)
    args.tfboard_path = os.path.join(exp_batch_path, Path(args.config).stem, 'TFBoard', args.exp_name)
    args.log_name = args.experiment_path

    create_experiment_dir(args)

    logger = Logger(args.experiment_path, args.log_name)

    # config
    print("*"*50)
    print_log(f"---Using config folder: {os.path.abspath(args.config_folder)}", logger = logger, color='blue')
    config_handler = ConfigHandler(args.config_folder, args.resume or args.test, logger = logger)
    config = config_handler.get_config(args) # if args.resume then ignore args.config and use config in experiment path

    if priority_config is not None:
        config = replace_config_with_priority(config, priority_config)
        print_log(f"---Updated config: {config.max_epoch}", logger=logger, color='yellow')
    ###################################################################################################################
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    metric_logger = SimulLogger(args, config)
    
    # set random seeds
    if args.seed is not None:
        print_log(f'---Set random seed to {args.seed}, \n deterministic: {args.deterministic}', logger, color='red')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    
    if args.debug:
        print_log("#"*20 + "Debug mode" + "#"*20, logger = logger, color='red')

    
    logger.log_args(args, 'args')

    
    logger.log_config(config, 'config')
    


    ################################# Loading Everything ##############################################################
    transforms = getattr(config.dataset, 'transforms', None)
    transform_with = getattr(config.dataset, 'transform_with', None)

    assert config.total_bs == 32, f"total_bs should be 32, but got {config.total_bs}"
    if args.debug:
        config.total_bs = 8

    return args, config, logger, metric_logger, transforms, transform_with

