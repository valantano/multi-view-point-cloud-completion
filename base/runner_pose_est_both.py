import torch
import time

from base.scaphoid_utils.logger import print_log, get_table_str

from base.scaphoid_utils.AverageMeter import AverageMeter

from base.scaphoid_insight.AlignerDebugger import Debugger


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
        if config.model_name == 'BothPoseEstPointAttN':
            input = data[0]
            net_input_volar, net_input_dorsal, full = input[0].cuda(), input[1].cuda(), input[2].cuda()
            gt = data[1]
            gt_volar, gt_dorsal, gt_full = gt[0].cuda(), gt[1].cuda(), gt[2].cuda()

            transform_paras = data[2]
            transform_with = config.dataset.transform_with

            gt_RTS_mat_v = transform_paras['volar_RTS'].cuda()  # RTS for the volar partial point cloud
            gt_RTS_mat_d = transform_paras['dorsal_RTS'].cuda()  # RTS for the dorsal partial point cloud

            ###########################################################################################################
            #           |----------------|                   |----------------------------|
            #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
            # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
            #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
            #           |----------------|                   | p_min, range_min, scale, 1 |
            #                                                |----------------------------|     
            ###########################################################################################################

            pred_RTS_mat_v, pred_RTS_mat_d, volar_pose_code, dorsal_pose_code = base_model(
                net_input_volar.transpose(2, 1).contiguous(), 
                net_input_dorsal.transpose(2, 1).contiguous(), 
                gt_RTS_mat_v[:, 3].contiguous()
            )

            if args.debug: # and idx == 0:
                debugger = Debugger(base_model, config, args, logger)
                # debugger.visualize_pose_est_comp(
                #     (net_input_volar, net_input_dorsal), 
                #     (transform_paras['volar_RTS'].cuda(), transform_paras['dorsal_RTS'].cuda()), 
                #     (pose_gt_volar, pose_gt_dorsal, pose_gt_full), 
                #     (in_out['6d_pose_v'], in_out['6d_pose_d']), 
                #     in_out, 
                #     full_pcd
                # )

            model_ret_list = [pred_RTS_mat_v, pred_RTS_mat_d, net_input_volar, net_input_dorsal]
            gt_list = [gt_RTS_mat_v, gt_RTS_mat_d, gt_volar, gt_dorsal]
        else:
            raise NotImplementedError(f'Train phase does not support {config.model_name}')
        ###############################################################################################################

        metrics_dict = base_model.module.get_metrics(model_ret_list, gt_list)
        loss_dict = base_model.module.get_loss(model_ret_list, gt_list)
        # total_loss = loss_dict['Loss/Total'] * 10
        # total_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())

        total_loss = (loss_dict['Loss/Total'] * 10 +
                      (30 * metrics_dict['Volar/Pose/CDL1'] + 
                       30 * metrics_dict['Dorsal/Pose/CDL1']) /2
                     ).mean()
        loss_dict['Loss/Total'] = total_loss
        loss_dict['Loss/Volar/CDL1'] = metrics_dict['Volar/Pose/CDL1']
        loss_dict['Loss/Dorsal/CDL1'] = metrics_dict['Dorsal/Pose/CDL1']

        total_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
        
        if num_iter == config.step_per_update:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
            num_iter = 0
            optimizer.step()
            base_model.zero_grad()

        losses_dict = {'Loss/Total': total_loss, 
                       'Loss/Volar/Rotation': loss_dict['Loss/Volar/Rotation'], 
                       'Loss/Volar/Translation': loss_dict['Loss/Volar/Translation'], 
                       'Loss/Dorsal/Rotation': loss_dict['Loss/Dorsal/Rotation'], 
                       'Loss/Dorsal/Translation': loss_dict['Loss/Dorsal/Translation'], 
                       'Loss/Volar/CDL1': loss_dict['Loss/Volar/CDL1'], 
                       'Loss/Dorsal/CDL1': loss_dict['Loss/Dorsal/CDL1']}
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
            break   ######################################## break ####################################################
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

    val_losses = AverageMeter(init_later=True)
    local_val_metrics = AverageMeter(['Total', 'Volar/Rotation', 'Volar/Translation', 'Dorsal/Rotation', 
                                      'Dorsal/Translation'])
    advanced_metrics = AverageMeter(init_later=True)

    n_samples = len(valid_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, data in enumerate(valid_dataloader):

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name != 'ScaphoidDataset':
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            if config.model_name == 'BothPoseEstPointAttN':
                input = data[0]
                net_input_volar, net_input_dorsal, full = input[0].cuda(), input[1].cuda(), input[2].cuda()
                gt = data[1]
                gt_volar, gt_dorsal, gt_full = gt[0].cuda(), gt[1].cuda(), gt[2].cuda()

                transform_paras = data[2]
                transform_with = config.dataset.transform_with

                gt_RTS_mat_v = transform_paras['volar_RTS'].cuda()  # RTS for the volar partial point cloud
                gt_RTS_mat_d = transform_paras['dorsal_RTS'].cuda()  # RTS for the dorsal partial point cloud

                #######################################################################################################
                #           |----------------|                   |----------------------------|
                #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
                # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
                #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
                #           |----------------|                   | p_min, range_min, scale, 1 |
                #                                                |----------------------------|     
                #######################################################################################################

                pred_RTS_mat_v, pred_RTS_mat_d, volar_pose_code, dorsal_pose_code = base_model(
                    net_input_volar.transpose(2, 1).contiguous(),
                    net_input_dorsal.transpose(2, 1).contiguous(), 
                    gt_RTS_mat_v[:, 3].contiguous()
                )

                if args.debug: # and idx == 0:
                    debugger = Debugger(base_model, config, args, logger)
                    # debugger.visualize_pose_est_comp(
                    #     (net_input_volar, net_input_dorsal),
                    #     (transform_paras['volar_RTS'].cuda(), transform_paras['dorsal_RTS'].cuda()),
                    #     (pose_gt_volar, pose_gt_dorsal, pose_gt_full), 
                    #     (in_out['6d_pose_v'], in_out['6d_pose_d']), 
                    #     in_out, 
                    #     full_pcd
                    # )

                model_ret_list = [pred_RTS_mat_v, pred_RTS_mat_d, net_input_volar, net_input_dorsal]
                gt_list = [gt_RTS_mat_v, gt_RTS_mat_d, gt_volar, gt_dorsal]

            else:
                raise NotImplementedError(f'Train phase does not support {config.model_name}')

            # _loss = base_model.module.get_loss(affine_matrix_pred, gt_affine_matrix)
            metrics_dict = base_model.module.get_metrics(model_ret_list, gt_list)
            loss_dict = base_model.module.get_loss(model_ret_list, gt_list)


            total_loss = (loss_dict['Loss/Total'] * 10 + 
                          (30 * metrics_dict['Volar/Pose/CDL1'] + 
                           30 * metrics_dict['Dorsal/Pose/CDL1']) /2
                         ).mean()
            loss_dict['Loss/Total'] = total_loss
            loss_dict['Loss/Volar/CDL1'] = metrics_dict['Volar/Pose/CDL1']
            loss_dict['Loss/Dorsal/CDL1'] = metrics_dict['Dorsal/Pose/CDL1']

            
            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            Advanced Metrics Calculation for ScaphoidAdaPoinTr and PoseEstPointAttN
            """

            advanced_metrics.update_via_dict(metrics_dict)
            local_val_metrics.update([total_loss, loss_dict['Loss/Volar/Rotation'], loss_dict['Loss/Volar/Translation'], 
                                      loss_dict['Loss/Dorsal/Rotation'], loss_dict['Loss/Dorsal/Translation']])
            val_losses.update_via_dict(loss_dict)
            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            """
     
            if (idx+1) % interval == 0:
                str(val_losses)
                _, (loss_names, loss_values) = get_table_str(
                    [name.replace('Loss/', '') for name in val_losses.get_names()], 
                    ["%.4f" % l for l in val_losses.val()]
                )
                _, (metric_names, metric_values) = get_table_str(local_val_metrics.get_names(), 
                                                                 ["%.4f" % m for m in local_val_metrics.val()])

                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_names } | Metrics: {metric_names } |', logger)
                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_values} | Metrics: {metric_values} |\n', logger)


        header = [name.replace('Loss/', '') for name in val_losses.get_names()]
        values = ["%.4f" % l for l in val_losses.avg()]
        stds = ["%.4f" % s for s in val_losses.std()]
        table, (header_str, values_str, std_str) = get_table_str(header, values, stds)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {header_str}  |', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {values_str}  |\n', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | STD: {std_str}  |\n', logger=logger)

     
    # Print validation results
    print_log('============================ VAL RESULTS ============================',logger=logger)
    header = ['#Samples'] + [name.replace('Loss/', '') for name in local_val_metrics.get_names()]
    values = [str(local_val_metrics.count(0))] + ["%.4f" % l for l in local_val_metrics.avg()]
    stds = ["STD +/-"] + ["%.4f" % s for s in local_val_metrics.std()]
    table, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
    print_log(header_str, logger)
    print_log(values_str, logger)
    print_log(stds_str, logger)

    # Add validation results to TensorBoard
    if metric_logger is not None:
        metrics_log = {**advanced_metrics.get_epoch_log_dict(val=True), **val_losses.get_epoch_log_dict(val=True)}
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='val')

    return local_val_metrics.avg('CDL1')

