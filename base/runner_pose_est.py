import time

import torch

from base.scaphoid_utils.logger import print_log, get_table_str
from base.scaphoid_utils.AverageMeter import AverageMeter
import base.scaphoid_utils.transformations as t
from base.scaphoid_insight.AlignerDebugger import Debugger


def forward_model(base_model, net_in_volar, net_in_dorsal, gt_volar, gt_dorsal, transform_with, transform_paras, 
                  args, config, logger, idx):
    
    if transform_with == 'volar':
        # RTS = transform_paras['volar_RTS']
        partial_to_align = net_in_volar
        gt_partial = gt_volar
    elif transform_with == 'dorsal':
        # RTS = transform_paras['dorsal_RTS']
        partial_to_align = net_in_dorsal
        gt_partial = gt_dorsal

    gt_RTS = transform_paras['full_RTS'].cuda()  # gt_RTS is the RTS for the partial transform_with partial point cloud

    #######################################################################################################
    #           |----------------|                   |----------------------------|
    #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
    # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
    #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
    #           |----------------|                   | p_min, range_min, scale, 1 |
    #                                                |----------------------------|     
    #######################################################################################################

    in_out = base_model(net_in_volar.transpose(2, 1).contiguous(), net_in_dorsal.transpose(2, 1).contiguous(), 
                        gt_RTS[:, 3].contiguous())
    pred_RTS_mat = in_out['6d_pose']

    if args.debug and idx % 100 == 0:
        debugger = Debugger(base_model, config, args, logger)
        # debugger.visualize_pose_est(
        #     (net_input_volar, net_input_dorsal, partial_to_align), 
        #     (transform_paras['volar_RTS'].cuda(), transform_paras['dorsal_RTS'].cuda(), gt_RTS), 
        #     (gt_volar, gt_dorsal, gt_full), 
        #     pred_RTS_mat
        # )

    return in_out, pred_RTS_mat, gt_RTS, partial_to_align, gt_partial



def train(base_model, train_dataloader, optimizer, scheduler, epoch, args, config, logger, metric_logger) -> None:
    base_model.train()

    epoch_start_time, batch_start_time = time.time(), time.time()
    batch_time, data_time, losses = AverageMeter('BatchTime'), AverageMeter('DataTime'), AverageMeter(init_later=True)

    num_iter = 0

    n_batches = len(train_dataloader)

    for idx, data in enumerate(train_dataloader):
        data_time.update(time.time() - batch_start_time)
        num_iter += 1
        
        ############ Handle different models ##########################################################################
        if config.model_name == 'PoseEstPointAttN':
            input = data[0]
            net_in_volar, net_in_dorsal, full = input[0].cuda(), input[1].cuda(), input[2].cuda()
            gt = data[1]
            gt_volar, gt_dorsal, gt_full = gt[0].cuda(), gt[1].cuda(), gt[2].cuda()

            transform_paras = data[2]
            transform_with = config.dataset.transform_with


            in_out, pred_RTS_mat, gt_RTS, partial_to_align, gt_partial = forward_model(
                base_model, net_in_volar, net_in_dorsal, gt_volar, gt_dorsal, transform_with, transform_paras, 
                args, config, logger, idx)
            

            metrics_dict = base_model.module.get_metrics(pred_RTS_mat, gt_RTS, partial_to_align, gt_partial, 
                                                         calc_f1=False)
            loss_dict = base_model.module.get_loss(pred_RTS_mat, gt_RTS)
            _loss = loss_dict['Loss/Total']

        else:
            raise NotImplementedError(f'Train phase do not support {config.model_name}')
        ###############################################################################################################

        

        total_loss = _loss * 10 + 30 * metrics_dict['Pose/CDL1']
        loss_dict['Loss/Total'] = total_loss
        loss_dict['Loss/CDL1'] = metrics_dict['Pose/CDL1']

        total_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
        
        if num_iter == config.step_per_update:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
            num_iter = 0
            optimizer.step()
            base_model.zero_grad()

        losses_dict = {'Loss/Total': total_loss, 'Loss/Rotation': loss_dict['Loss/Rotation'], 
                       'Loss/Translation': loss_dict['Loss/Translation'], 
                       'Loss/CDL1': torch.tensor(metrics_dict['Pose/CDL1'])}
        losses.update_via_dict(losses_dict)
            
        n_itr = epoch * n_batches + idx
            
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()

        if idx % 100 == 0:
            header = [name.replace('Loss/', '') for name in losses.get_names()] + ['LearningRate']
            values = ["%.4f" % l for l in losses.val()] + ["%.6f" % optimizer.param_groups[0]['lr']]
            table, (header_str, values_str) = get_table_str(header, values)

            print_log(f'Train[{idx + 1}/{n_batches}] | Losses: {header_str}  |', logger=logger)
            print_log(f'Train[{idx + 1}/{n_batches}] | Losses: {values_str}  |\n', logger=logger)
            

        if config.scheduler.type == 'GradualWarmup':
            if n_itr < config.scheduler.kwargs_2.total_epoch:
                scheduler.step()
        if args.debug:
            break   ####################################### break #####################################################
    ##### end for train_dataloader

    if isinstance(scheduler, list):
        for item in scheduler:
            item.step()
    else:
        scheduler.step()
    epoch_end_time = time.time()

    if metric_logger is not None:
        metrics_log = {**losses.get_epoch_log_dict(val=True), 'LearningRate': optimizer.param_groups[0]['lr']}
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='train')
        

    epoch_time = epoch_end_time - epoch_start_time
    minutes, seconds = (epoch_time // 60, epoch_time % 60)

    header = [name.replace('Loss/', '') for name in losses.get_names()]
    values = ["%.4f" % l for l in losses.avg()]
    stds = ["%.4f" % s for s in losses.std()]
    table, (header_str, values_str, std_str) = get_table_str(header, values, stds)

    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | Losses: {header_str} |', logger)
    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | Losses: {values_str} |\n', logger)
    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | STD: {std_str} |\n', logger)



def validate(base_model, valid_dataloader, epoch, metric_logger, args, config, logger) -> float:
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(init_later=True)
    local_test_metrics = AverageMeter(['Total', 'Rotation', 'Translation', 'CDL1', 'CDL2', 'F1'])
    advanced_metrics = AverageMeter(init_later=True)

    n_samples = len(valid_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, data in enumerate(valid_dataloader):

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name != 'ScaphoidDataset':
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if config.model_name == 'PoseEstPointAttN':
                input = data[0]
                gt = data[1]
                net_in_volar, net_in_dorsal, full = input[0].cuda(), input[1].cuda(), input[2].cuda()
                gt_volar, gt_dorsal, gt_full = gt[0].cuda(), gt[1].cuda(), gt[2].cuda()

                transform_paras = data[2]
                transform_with = config.dataset.transform_with

                in_out, pred_RTS_mat, gt_RTS, partial_to_align, gt_partial = forward_model(
                    base_model, net_in_volar, net_in_dorsal, gt_volar, gt_dorsal, transform_with, transform_paras, 
                    args, config, logger, idx)

                # _loss = base_model.module.get_loss(affine_matrix_pred, gt_affine_matrix)
                metrics_dict = base_model.module.get_metrics(pred_RTS_mat, gt_RTS, partial_to_align, gt_partial, 
                                                             calc_f1=True)
                loss_dict = base_model.module.get_loss(pred_RTS_mat, gt_RTS)
                _loss = loss_dict['Loss/Total']


                total_loss = _loss * 10 + 30 * metrics_dict['Pose/CDL1']
                loss_dict['Loss/Total'] = total_loss
                loss_dict['Loss/CDL1'] = metrics_dict['Pose/CDL1']

            else:
                raise NotImplementedError(f'Train phase does not support {config.model_name}')

            
            ##### Advanced Metrics Calculation for ScaphoidAdaPoinTr and PoseEstPointAttN #############################
            advanced_metrics.update_via_dict(metrics_dict)
            local_test_metrics.update([total_loss, loss_dict['Loss/Rotation'], loss_dict['Loss/Translation'], 
                                       metrics_dict['Pose/CDL1'], metrics_dict['Pose/CDL2'], metrics_dict['Pose/F1']])
            test_losses.update_via_dict(loss_dict)
            ###########################################################################################################

            if metric_logger is not None and idx % 200 == 0:
                if epoch % 15 == 0:      # avoid huge amount of data
                    gt_partial_transformed = t.apply_RTS_transformation(gt_partial, pred_RTS_mat)
                    gt_partial_transformed = gt_partial_transformed.squeeze().cpu().numpy()
                    partial_to_align = partial_to_align.squeeze().cpu().numpy()

                    metric_logger.add_pred_3d('Model%02d/Pred'% idx, gt_partial_transformed, partial_to_align, epoch)

     
            if (idx+1) % interval == 0:
                str(test_losses)
                _, (loss_names, loss_values) = get_table_str(
                    [name.replace('Loss/', '') for name in test_losses.get_names()], 
                    ["%.4f" % l for l in test_losses.val()]
                )
                _, (metric_names, metric_values) = get_table_str(local_test_metrics.get_names(), 
                                                                 ["%.4f" % m for m in local_test_metrics.val()])

                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_names } | Metrics: {metric_names } |', logger)
                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_values} | Metrics: {metric_values} |\n', logger)


        header = [name.replace('Loss/', '') for name in test_losses.get_names()]
        values = ["%.4f" % l for l in test_losses.avg()]
        stds = ["%.4f" % s for s in test_losses.std()]
        _, (header_str, values_str, std_str) = get_table_str(header, values, stds)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {header_str}  |', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {values_str}  |\n', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | STD: {std_str}  |\n', logger=logger)

     
    # Print validation results
    print_log('============================ VAL RESULTS ============================', logger)
    header = ['#Samples'] + [name.replace('Loss/', '') for name in local_test_metrics.get_names()]
    values = [str(local_test_metrics.count(0))] + ["%.4f" % l for l in local_test_metrics.avg()]
    stds = ["STD +/-"] + ["%.4f" % s for s in local_test_metrics.std()]
    _, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
    print_log(header_str, logger)
    print_log(values_str, logger)
    print_log(stds_str, logger)

    # Add validation results to TensorBoard
    if metric_logger is not None:
        metrics_log = {**advanced_metrics.get_epoch_log_dict(val=True), **test_losses.get_epoch_log_dict(val=True)}
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='val')

    return local_test_metrics.avg('CDL1')

