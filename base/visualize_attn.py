import time
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display


current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the main directory to sys.path such that submodules can be imported
if main_dir not in sys.path:
    sys.path.append(main_dir)


from base.scaphoid_utils import parser
import base.scaphoid_utils.constants as const
from base.scaphoid_utils.config import ConfigHandler
from base.scaphoid_utils.misc import get_ptcloud_img, worker_init_fn
from base.scaphoid_utils.logger import SimulLogger, TensorboardLogger, Logger, print_log
from base.scaphoid_utils.AttnVisualizer import AttnVisualizer

from base.scaphoid_datasets.Transforms import ReverseTransforms
from base.scaphoid_datasets.ScaphoidDataset import ScaphoidDataset

from base.scaphoid_models.ScaphoidPointAttN import ScaphoidPointAttN
import base.scaphoid_models.ScaphoidPointAttN as module

import base.scaphoid_utils.builder as b


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' %
              args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


def main():
    # args
    args = parser.get_args()

    ###### Experiment Path ######
    exp_batch_path = os.path.join('./mvpcc_experiments', Path(args.config).parent.stem)
    args.experiment_path = os.path.join(exp_batch_path, Path(args.config).stem, args.exp_name)
    args.tfboard_path = os.path.join(exp_batch_path, Path(args.config).stem, 'TFBoard', args.exp_name)
    args.log_name = args.experiment_path

    ##############################
    print_log(f"Experiment Path: {args.config}", color='blue')
    print_log(f"Experiment Path: {args.experiment_path}", color='blue')

    # logger
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, name=args.log_name)

    # config
    config_handler = ConfigHandler(args.config_folder, args.resume)
    config = config_handler.get_config(args)        # if args.resume then ignore args.config and use config in experiment path
    config.dataset.train.others.bs = 1
    args.distributed = False

    # config
    logger = Logger(args.log_name)

    # build dataset
    transforms = None
    transform_with = None
    try:
        transforms = config.dataset.transforms
        transform_with = config.dataset.transform_with
    except:
        pass

    args.debug = True
    # t_dataset = EnrichedScaphoidDataset('train', config.dataset.train._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)
    v_dataset = ScaphoidDataset('val', config.dataset.val._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)

    # train_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=1, shuffle=True, drop_last=True, 
    #                                               num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    valid_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=1, shuffle=False, drop_last=False,
                                                  num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    
    print_log(f"################### Loading Model: {config.model_name} #####################", logger, color='blue')
    if config.model_name == 'ScaphoidPointAttN':
        base_model = ScaphoidPointAttN(config)
        base_model = torch.nn.DataParallel(base_model).cuda()
        if args.pretrained:
            b.load_model(base_model, args.pretrained, config.model_name, logger = logger)
            base_model.module.N_POINTS = 16384
    else:
        raise NotImplementedError(f'Train phase does not support {config.model_name}')
    
    print_log(f"################### Loading Optimizer and Scheduler #####################", logger, color='blue')
    if args.resume: # resume ckpts
        _, _ = b.resume_model(base_model, args, config, logger)
    # else:
    #     raise NotImplementedError(f'No resume model found for resume={args.resume}')
    # base_model.module.N_POINTS = 16384
    base_model.eval()  # set model to eval mode
    module.store_attn_weights = True
    collector = module.collector

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(valid_dataloader):

            dataset_name = config.dataset.val._base_.NAME
            if dataset_name != 'ScaphoidDataset':
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            
            net_input_volar = data[0].cuda()
            net_input_dorsal = data[1].cuda()
            gt = data[2].cuda()

            if config.model_name == 'ScaphoidPointAttN':

                dense1, dense, sparse, seeds = base_model(net_input_volar.transpose(2, 1).contiguous(), net_input_dorsal.transpose(2, 1).contiguous())
                
            else:
                raise NotImplementedError(f'Train phase does not support {config.model_name}')
            
            print(collector)

            vis= AttnVisualizer(collector, logger)
            print_log(f"{idx}")
            vis.plot()
            
            print("faf")
            del vis
            collector.clear()
            # plot_attn_interactive(199, 4, collector, base_model, logger)




    


if __name__ == '__main__':
    main()
