import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
poin_tr_dir = os.path.join(main_dir, 'submodules', 'PoinTr')

# Add the main directory to sys.path such that submodules can be imported
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(poin_tr_dir)


from base.scaphoid_utils.args_config_handling import handle_args_and_config
from submodules.PoinTr.tools import builder


from base.runner_pose_est import train, validate
from base.scaphoid_utils.logger import print_log
from base.scaphoid_utils.misc import worker_init_fn
import base.scaphoid_utils.builder as b
from base.scaphoid_models.PoseEstPointAttN import PoseEstPointAttN
from base.scaphoid_datasets.PoseEstDataset import PoseEstDataset


def main(prio_args=None, prio_config=None) -> torch.Tensor:
    """
    Main function to start one training run of the Pose Estimation Model.
    Is either called using train_pose_est_pointattn.sh or by SMAC3.
    :param prio_args: Dictionary with arguments that will override the default ones.
    :param prio_config: Dictionary with configuration that will override the default one.
    :return: The best cdl1 score achieved during the training.
    This function initializes the datasets, model, optimizer, and scheduler, and runs the training loop.
    It also handles validation and saving of the model checkpoints.
    If called by SMAC3, prio_args and prio_config will be provided.
    """
    args, config, logger, metric_logger, transforms, transform_with = handle_args_and_config(prio_args, prio_config)

    t_dataset = PoseEstDataset('train', config.dataset.train._base_, transforms, transform_with, args.debug, logger, 
                               args.strict)
    v_dataset = PoseEstDataset('val', config.dataset.val._base_, transforms, transform_with, args.debug, logger, 
                               args.strict)

    train_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=int(config.total_bs), shuffle=True, 
                                                   drop_last=True, num_workers=int(args.num_workers), 
                                                   worker_init_fn=worker_init_fn)
    valid_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=1, shuffle=False, 
                                                   drop_last=False, num_workers=int(args.num_workers), 
                                                   worker_init_fn=worker_init_fn)


    start_epoch, best_cdl1_score, metrics = 0, None, None

    print_log(f"################### Loading Model: {config.model_name} #####################", logger, color='blue')
    if config.model_name == 'PoseEstPointAttN':
        base_model = PoseEstPointAttN(config)
        base_model = torch.nn.DataParallel(base_model).cuda()
        if args.pretrained:
            b.load_model(base_model, args.pretrained, config.model_name, logger = logger)
            base_model.module.N_POINTS = 16384
    else:
        raise NotImplementedError(f'Train phase does not support {config.model_name}')
    
    print_log(f"################### Loading Optimizer and Scheduler #####################", logger, color='blue')
    optimizer = builder.build_optimizer(base_model, config) # optimizer & scheduler
    if args.resume: # resume ckpts
        start_epoch, best_cdl1_score = b.resume_model(base_model, args, config, logger)
        print_log(f"start_epoch: {start_epoch}, best_cdl1_score: {best_cdl1_score}", logger, color='red')
        builder.resume_optimizer(optimizer, args, None)

    if not args.resume:
        b.partly_resume_model(base_model, args, config, freeze=False, logger=logger)
            
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    ############################## Training Loop ######################################################################
    if args.test:
        raise NotImplementedError("Test is not implemented yet.")
    else:
        for epoch in range(start_epoch, config.max_epoch + 1):
            train(base_model, train_dataloader, optimizer, scheduler, epoch, args, config, logger, metric_logger)

            if epoch % args.val_freq == 0:
                cdl1_score = validate(base_model, valid_dataloader, epoch, metric_logger, args, config, logger=logger)

                if  best_cdl1_score is None or cdl1_score < best_cdl1_score:
                    best_cdl1_score = cdl1_score
                    b.save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, 'ckpt-best', args, 
                                      logger)

            b.save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, 'ckpt-last', args, logger)
            if (config.max_epoch - epoch) < 2:
                b.save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, f'ckpt-epoch-{epoch:03d}', 
                                  args, logger)

    print_log(f"################### Finished Training #####################", logger, color='blue')

    metric_logger.close()
    return best_cdl1_score


if __name__ == '__main__':
    main()