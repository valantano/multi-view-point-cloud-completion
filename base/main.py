import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
POINTR_DIR = os.path.join(MAIN_DIR, "submodules", "PoinTr")

# Add the main directory to sys.path such that submodules can be imported
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)
    sys.path.append(POINTR_DIR)

from submodules.PoinTr.tools import builder


import base.scaphoid_utils.builder as b
from base.scaphoid_utils.args_config_handling import handle_args_and_config
from base.scaphoid_utils.logger import print_log
from base.scaphoid_utils.misc import worker_init_fn
from base.runner_scaphoid import train, validate, test

from base.scaphoid_models.ScaphoidAdaPoinTr import ScaphoidAdaPoinTr
from base.scaphoid_models.ScaphoidPointAttN import ScaphoidPointAttN
from base.scaphoid_models.PoseEstPointAttN import CompletionPoseEstPointAttN
from base.scaphoid_datasets.ScaphoidDataset import ScaphoidDataset


def build_datasets(args, config, transforms, transform_with, logger):
    """
    Setup datasets and dataloader for training, validation, and testing.
    :param args: Arguments from command line.
    :param config: Configuration dictionary.
    :param transforms: Transformations to apply to the dataset.
    :param transform_with: Additional transformations.
    :param logger: Logger for logging information.
    :return: train_dataloader, valid_dataloader, test_dataloader
    """
    print_log(f"##################### Loading Datasets #####################", logger, color="blue")
    train_dataloader, valid_dataloader = None, None
    if not args.test:
        t_dataset = ScaphoidDataset("train", config.dataset.train._base_, transforms, args.strict, transform_with,
                                     args.debug, logger, config.model_name)
        v_dataset = ScaphoidDataset("val", config.dataset.val._base_, transforms, args.strict, transform_with,
                                     args.debug, logger, config.model_name)

        train_dataloader = DataLoader(t_dataset, batch_size=int(config.total_bs), shuffle=True, drop_last=True, 
                                      num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
        valid_dataloader = DataLoader(v_dataset, batch_size=1, shuffle=False, drop_last=False,
                                      num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)

    test_dataset = ScaphoidDataset("val", config.dataset.test._base_, transforms, False, transform_with, args.debug, 
                                   logger, config.model_name)  # not real test set
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, 
                                 num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)

    return train_dataloader, valid_dataloader, test_dataloader


def build_model(args, config, logger):
    """
    Setup the model for training.
    :param args: Arguments from command line.
    :param config: Configuration dictionary.
    :param logger: Logger for logging information.
    :return: base_model, start_epoch, best_cdl1_score
    """
    print_log(
        f"##################### Loading Model: {config.model_name} #####################", logger, color="blue")
    if config.model_name == "ScaphoidAdaPoinTr":
        base_model = ScaphoidAdaPoinTr(config.model)
        base_model = nn.DataParallel(base_model).cuda()
        if args.pretrained:
            b.load_model(base_model, args.pretrained, config.model_name, logger=logger)
        logger.log_model_info(base_model)

    elif config.model_name == "ScaphoidPointAttN":
        base_model = ScaphoidPointAttN(config, args.pretrained)
        base_model = torch.nn.DataParallel(base_model).cuda()
        if args.pretrained:
            b.load_model(base_model, args.pretrained, config.model_name, logger=logger)
            base_model.module.N_POINTS = 16384

    elif config.model_name == "CompletionPoseEstPointAttN":
        base_model = CompletionPoseEstPointAttN(config, args.pretrained)
        base_model = torch.nn.DataParallel(base_model).cuda()
        if args.pretrained:
            b.load_model(base_model, args.pretrained, config.model_name, logger=logger, arch=config.model.arch)
            base_model.module.N_POINTS = 16384
        b.init_pose_estimators(base_model, args, freeze=True, logger=logger)

    else:
        raise NotImplementedError(f"Train phase does not support {config.model_name}")

    # output model parameters
    num_params = sum(p.numel() for p in base_model.parameters())
    print_log(f"Total parameters: {num_params}", color="yellow")  # adapointr: 32.47M, pointattn: 31.41M

    print_log(f"Model: {config.model_name} loaded successfully", color="green")

    # Resume model
    start_epoch, best_cdl1_score = 0, None
    if args.resume or args.test:
        print_log(f"Resuming model from {args.resume}", color="red")
        start_epoch, best_cdl1_score = b.resume_model(base_model, args, config, logger)
        print_log(f"start_epoch: {start_epoch}, best_cdl1_score: {best_cdl1_score}", logger, color="red")

    return base_model, start_epoch, best_cdl1_score


def build_optimizer(base_model, args, config, logger):
    """
    Build optimizer for the model.
    :param base_model: The model to optimize.
    :param config: Configuration dictionary.
    :return: Optimizer instance.
    """
    print_log(f"##################### Loading Optimizer #####################", logger, color="blue")  
    optimizer = builder.build_optimizer(base_model, config)
    if args.resume or args.test:
        print_log(f"---Resuming optimizer from {args.resume}", color="red")
        b.resume_optimizer(optimizer, args, logger)
    print_log(f"---Optimizer loaded successfully", color="green")
    return optimizer


def main(prio_args=None, prio_config=None):
    """
    Main function to start one training run of the Scaphoid Model.
    Is either called using train_scaphoid_pointattn.sh or by SMAC3.
    :param prio_args: Dictionary with arguments that will override the default ones.
    :param prio_config: Dictionary with configuration that will override the default one.
    :return: The best cdl1 score achieved during the training.
    This function initializes the datasets, model, optimizer, and scheduler, and runs the training loop.
    It also handles validation and saving of the model checkpoints.
    If called by SMAC3, prio_args and prio_config will be provided.
    """

    args, config, logger, metric_logger, transforms, transform_with = handle_args_and_config(prio_args, prio_config)

    train_dataloader, valid_dataloader, test_dataloader = build_datasets(args, config, transforms, transform_with, 
                                                                         logger)

    start_epoch, best_cdl1_score, cdl1_score = 0, None, None

    base_model, start_epoch, best_cdl1_score = build_model(args, config, logger)
    optimizer = build_optimizer(base_model, args, config, logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch - 1)

    # b.partly_resume_model(base_model, args, logger)
    epoch = start_epoch

    ############################## Training Loop #####################################################################
    if args.test:
        cdl1_score = test(base_model, test_dataloader, epoch, args, config, logger, metric_logger)
    else:
        for epoch in range(start_epoch, config.max_epoch):
            train(base_model, train_dataloader, optimizer, scheduler, epoch, args, config, logger, metric_logger)

            if epoch % args.val_freq == 0 or (config.max_epoch - epoch) < 2:  # last epochs
                cdl1_score = validate(base_model, valid_dataloader, epoch, metric_logger, args, config, logger, 
                                      calc_emd=(epoch == config.max_epoch - 1))

                if best_cdl1_score is None or cdl1_score < best_cdl1_score:
                    best_cdl1_score = cdl1_score
                    b.save_checkpoint(base_model, optimizer, epoch, best_cdl1_score, best_cdl1_score, "ckpt-best", args,
                                      logger)
                if (config.max_epoch - epoch) < 2:
                    b.save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, 
                                      f"ckpt-epoch-{epoch:03d}", args, logger)

            b.save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, "ckpt-last", args, logger)

        if not args.smac3:  # do not evaluate on test set in SMAC3
            cdl1_score = test(base_model, test_dataloader, epoch, args, config, logger, metric_logger)

    print_log(f"################### Finished Training #####################", logger, color="blue")
    ##################################################################################################################

    metric_logger.close()
    return best_cdl1_score


if __name__ == "__main__":
    main()
