import torch
import time
from tqdm import tqdm


from submodules.PoinTr.extensions.emd import emd_module as emd
from PointAttn.utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D


from base.scaphoid_insight.AlignerDebugger import Debugger
from base.scaphoid_utils.logger import print_log, get_table_str, SimulLogger
from base.scaphoid_utils.AverageMeter import AverageMeter
import base.scaphoid_utils.transformations as t
from base.scaphoid_utils.SamplesTracker import DoubleSampleTracker

def get_pc_subdivision(inds):
    """
    Get the subdivision of the point cloud based on the indices.
    :param inds: Indices of the point cloud.
    :return: pc_subdivision dictionary with subdivisions for volar, dorsal, articular, distal, and proximal.
    """
    return {
        'volar': {'weighting': 1, 'ind': inds[0].cuda()},
        'dorsal': {'weighting': 1, 'ind': inds[1].cuda()},
        'articular': {'weighting': 1, 'ind': inds[2].cuda()},
        'distal': {'weighting': 1, 'ind': inds[3].cuda()},
        'proximal': {'weighting': 1, 'ind': inds[4].cuda()}
    }

def forward_model(base_model, net_in_volar, net_in_dorsal, full_pcd, transform_with, approach_dependent_gts,
                   transform_paras, args, config, logger):
    if config.model_name == 'ScaphoidAdaPoinTr':
        sparse_pc, dense1 = base_model(net_in_volar, net_in_dorsal, use=transform_with)
        output = base_model(net_in_volar, net_in_dorsal, use=transform_with)
        output = list(output)
        seeds = output[0]
        if len(output) == 2:
            output.reverse()

        model_ret_list = output
        gt_list = [full_pcd]

    elif config.model_name == 'ScaphoidPointAttN':
        dense1, dense, sparse_pc, seeds = base_model(net_in_volar.transpose(2, 1).contiguous(), 
                                                     net_in_dorsal.transpose(2, 1).contiguous())
        model_ret_list = [dense1, dense, sparse_pc, seeds]
        gt_list = [full_pcd]

    elif config.model_name == 'CompletionPoseEstPointAttN':
        pose_gt_volar = approach_dependent_gts[0].cuda()
        pose_gt_dorsal = approach_dependent_gts[1].cuda()
        pose_gt_full = approach_dependent_gts[2].cuda()

        gt_RTS_mat_v = transform_paras['volar_RTS'].cuda()  # RTS for the volar partial point cloud
        gt_RTS_mat_d = transform_paras['dorsal_RTS'].cuda()  # RTS for the dorsal partial point cloud

        ###############################################################################################################
        #           |----------------|                   |----------------------------|
        #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
        # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
        #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
        #           |----------------|                   | p_min, range_min, scale, 1 |
        #                                                |----------------------------|     
        ###############################################################################################################

        in_out = base_model(net_in_volar.transpose(2, 1).contiguous(), net_in_dorsal.transpose(2, 1).contiguous(), 
                            gt_RTS_mat_v[:, 3].contiguous())
        dense1, dense, sparse_pc, seeds = in_out['dense1'], in_out['dense'], in_out['sparse_pc'], in_out['seeds']
        pred_RTS_mat_v, pred_RTS_mat_d = in_out['6d_pose_v'], in_out['6d_pose_d']

        if "GT_augment_RTS" in in_out:
            # Depending on internal Aligner, the GT needs to be transformed the same way as the anchor point cloud was 
            # transformed.
            pose_gt_full = t.apply_RTS_transformation(pose_gt_full, in_out['GT_augment_RTS'])
            full_pcd = pose_gt_full

        if args.debug:
            debugger = Debugger(base_model, config, args, logger)
            # debugger.visualize_pose_est_comp(
            #     (net_in_volar, net_in_dorsal), 
            #     (transform_paras['volar_RTS'].cuda(), transform_paras['dorsal_RTS'].cuda()), 
            #     (pose_gt_volar, pose_gt_dorsal, pose_gt_full), 
            #     (in_out['6d_pose_v'], in_out['6d_pose_d']), 
            #     in_out, 
            #     full_pcd
            # )

        
        model_ret_list = [dense1, dense, sparse_pc, seeds, pred_RTS_mat_v, pred_RTS_mat_d, net_in_volar, net_in_dorsal]
        gt_list = [full_pcd, gt_RTS_mat_v, gt_RTS_mat_d, pose_gt_volar, pose_gt_dorsal]
    
    else:
        raise NotImplementedError(f'Train phase does not support {config.model_name}')
    
    return model_ret_list, gt_list
    

def test(base_model, test_dataloader, epoch, args, config, logger, metric_logger: SimulLogger):
    print_log(f"[TEST] Start Test", logger)
    base_model.eval()  # set model to eval mode

    EMD = emd.emdModule()
    test_losses = AverageMeter(init_later=True)
    test_metrics = AverageMeter(['F1', 'CDL1', 'CDL2', 'EMD'])
    advanced_metrics = AverageMeter(init_later=True)
    
    n_samples = len(test_dataloader) # bs is 1
    sample_tracker = DoubleSampleTracker(max_samples=10, smaller_is_better=True)

    dist_vals = []

    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing")
    with torch.no_grad():
        for idx, data in bar:
            sample_start_time = time.time()

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name != 'ScaphoidDataset':
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            transformed_data, approach_dependent_gts, inds, transform_paras = data
            transform_with = config.dataset.transform_with

            net_input_volar = transformed_data[0].cuda()
            net_input_dorsal = transformed_data[1].cuda()
            full_pcd = transformed_data[2].cuda()

            ################################################## Forward pass ###########################################
            model_ret_list, gt_list = forward_model(base_model, net_input_volar, net_input_dorsal, full_pcd, 
                                                    transform_with, approach_dependent_gts, transform_paras, args, 
                                                    config, logger)
            ################################################## Forward pass ###########################################

            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            Advanced Metrics Calculation for ScaphoidAdaPoinTr and ScaphoidPointAttN
            """
            scale = transform_paras['transform_full'][2]['rescale'][2]
            mm_thresh = [0.25]  # in mm
            thresholds = [(thr, scale * thr) for thr in mm_thresh]

            pc_subdivision = get_pc_subdivision(inds)
            detailed_metrics, _ = base_model.module.get_metrics(model_ret_list, gt_list, pc_subdivision, thresholds)
            
            dist, _ = EMD(model_ret_list[0], gt_list[0], 0.005, 10000)
            emd_val = torch.mean(torch.sqrt(dist)) * 1000
            # emd_val = torch.tensor(0.0)


            cham_loss = dist_chamfer_3D.chamfer_3DDist()
            dist1, dist2, idx1, idx2 = cham_loss(full_pcd, model_ret_list[0])  # dist2 output to gt
            dist2 = dist2.detach().sqrt().cpu().numpy() / scale
            dist_vals.append(dist2)
            

            # print_log(f"[TEST] EMD: {emd_val}", logger, color='green')
            # never call get_loss before get_metrics!!!
            loss_dict = base_model.module.get_loss(model_ret_list, gt_list, pc_subdivision)

            sample_tracker.add_sample(idx, detailed_metrics['Full/CDL1'].item()*1000)


            detailed_metrics['Full/EMD'] = emd_val
            advanced_metrics.update_via_dict(detailed_metrics)
            test_metrics.update([detailed_metrics['Full/F1'], detailed_metrics['Full/CDL1'] * 1000, 
                                 detailed_metrics['Full/CDL2'] * 1000, emd_val])
            test_losses.update_via_dict(loss_dict)
            bar.set_description(f"EMD: {emd_val:.4f} EMD AVG: {advanced_metrics.avg('Full/EMD'):.4f}")
            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            """

        header = [name.replace('Loss/', '') for name in test_losses.get_names()]
        values = ["%.4f" % l for l in test_losses.avg()]
        stds = ["%.4f" % l for l in test_losses.std()]
        table, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
        print_log(f'[Test] EPOCH: {epoch} | Losses: {header_str}  |', logger=logger)
        print_log(f'[Test] EPOCH: {epoch} | Losses: {values_str}  |\n', logger=logger)
        print_log(f'[Test] EPOCH: {epoch} | STD +/-: {stds_str}  |\n', logger=logger)

    # Print validation results
    print_log('============================ TEST RESULTS ============================',logger=logger)
    header = ['#Samples'] + [name.replace('Loss/', '') for name in test_metrics.get_names()]
    values = [str(test_metrics.count(0))] + ["%.4f" % l for l in test_metrics.avg()]
    stds = ['STD +/-'] + ["%.4f" % l for l in test_metrics.std()]
    table, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
    print_log(header_str, logger)
    print_log(values_str, logger)
    print_log(stds_str, logger)

    print_log(f"Sample Tracker: \n{sample_tracker}", logger)

    # Add validation results to TensorBoard
    if metric_logger is not None:
        metrics_log = {**advanced_metrics.get_epoch_log_dict(val=True), **test_losses.get_epoch_log_dict(val=True)}
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='test')

    # write dist_vals to file
    # np.save(f"dist_vals_{config.model.arch}.npy", np.array(dist_vals))

    return test_metrics.avg('CDL1')
    


def train(base_model, train_dataloader, optimizer, scheduler, epoch, args, config, logger, metric_logger):
    base_model.train()

    epoch_start_time, batch_start_time = time.time(), time.time()
    batch_time, data_time, losses = AverageMeter('BatchTime'), AverageMeter('DataTime'), AverageMeter(init_later=True)


    num_iter = 0

    n_batches = len(train_dataloader)

    for idx, data in enumerate(train_dataloader):
        data_time.update(time.time() - batch_start_time)
        num_iter += 1

        transformed_data, approach_dependent_gts, ind, transform_paras = data
        transform_with = config.dataset.transform_with

        net_in_volar = transformed_data[0].cuda()
        net_in_dorsal = transformed_data[1].cuda()
        full_pcd = transformed_data[2].cuda()

        
        ################################################## Forward pass ###############################################
        model_ret_list, gt_list = forward_model(base_model, net_in_volar, net_in_dorsal, full_pcd, transform_with,
                                                approach_dependent_gts, transform_paras, args, config, logger)
        ################################################## Forward pass ###############################################

        pc_subdivision = get_pc_subdivision(ind)
        loss_dict = base_model.module.get_loss(model_ret_list, gt_list, pc_subdivision=pc_subdivision)
        _loss = loss_dict['Loss/Total']
        _loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())

        if num_iter == config.step_per_update:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
            num_iter = 0
            optimizer.step()
            base_model.zero_grad()

        losses.update_via_dict(loss_dict)
            
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
            break   ############################################## break ##############################################
    ##### end for train_dataloader

    if isinstance(scheduler, list):
        for item in scheduler:
            item.step()
    else:
        scheduler.step()
    epoch_end_time = time.time()

    if metric_logger is not None:
        metrics_log = losses.get_epoch_log_dict(val=False)
        metrics_log['LearningRate'] = optimizer.param_groups[0]['lr']
        # metrics_log={'Loss/Epoch/Sparse': losses.avg('Loss/Sparse'), 'Loss/Epoch/Dense': losses.avg('Loss/Dense'), 
        #              'Loss/Total': losses.avg('Loss/Total'), }
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='train')
        

    epoch_time = epoch_end_time - epoch_start_time
    minutes, seconds = (epoch_time // 60, epoch_time % 60)

    header = [name.replace('Loss/', '') for name in losses.get_names()]
    values = ["%.4f" % l for l in losses.avg()]
    stds = ["%.4f" % l for l in losses.std()]
    table, (header_str, values_str, stds_str) = get_table_str(header, values, stds)

    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | Losses: {header_str} |', logger)
    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | Losses: {values_str} |', logger)
    print_log(f'[Train] EPOCH: {epoch} EpochTime = {minutes} min {seconds:.3f} (s) | STD +/-: {stds_str} |\n', logger)



def validate(base_model, valid_dataloader, epoch, metric_logger, args, config, logger, calc_emd=False):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger)
    base_model.eval()  # set model to eval mode

    val_losses = AverageMeter(init_later=True)
    val_metrics = AverageMeter(['F1', 'CDL1', 'CDL2', 'EMD'])
    advanced_metrics = AverageMeter(init_later=True)

    n_samples = len(valid_dataloader) # bs is 1
    if calc_emd:
        EMD = emd.emdModule()

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, data in enumerate(valid_dataloader):

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name != 'ScaphoidDataset':
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            transformed_data, approach_dependent_gts, inds, transform_paras = data
            transform_with = config.dataset.transform_with

            net_in_volar = transformed_data[0].cuda()
            net_in_dorsal = transformed_data[1].cuda()
            full_pcd = transformed_data[2].cuda()

            pc_subdivision = get_pc_subdivision(inds)
            
            ################################################## Forward pass ###########################################
            model_ret_list, gt_list = forward_model(base_model, net_in_volar, net_in_dorsal, full_pcd, transform_with, 
                                                    approach_dependent_gts, transform_paras, args, config, logger)
            ################################################## Forward pass ###########################################

            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            Advanced Metrics Calculation for ScaphoidAdaPoinTr and ScaphoidPointAttN
            """
            scale = transform_paras['transform_full'][2]['rescale'][2]
            mm_thresh = [0.25]  # in mm
            thresholds = [(thr, scale * thr) for thr in mm_thresh]

            detailed_metrics, _ = base_model.module.get_metrics(model_ret_list, gt_list, pc_subdivision, thresholds)

            if calc_emd:
                dist, _ = EMD(model_ret_list[0], gt_list[0], 0.005, 10000)
                emd_val = torch.mean(torch.sqrt(dist)) * 1000    
            else:
                emd_val = torch.tensor(0.0, device=full_pcd.device)

            # never call get_loss before get_metrics!!!
            loss_dict = base_model.module.get_loss(model_ret_list, gt_list, pc_subdivision=pc_subdivision)
            detailed_metrics['Full/EMD'] = emd_val
            advanced_metrics.update_via_dict(detailed_metrics)
            val_metrics.update([detailed_metrics['Full/F1'],detailed_metrics['Full/CDL1'] * 1000, 
                                detailed_metrics['Full/CDL2'] * 1000, emd_val])
            val_losses.update_via_dict(loss_dict)
            """
            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            """

     
            if (idx+1) % interval == 0:
                _, (loss_names, loss_values) = get_table_str(
                    [name.replace('Loss/', '') for name in val_losses.get_names()], 
                    ["%.4f" % l for l in val_losses.val()]
                )
                _, (metric_names, metric_values) = get_table_str(val_metrics.get_names(), 
                                                                 ["%.4f" % m for m in val_metrics.val()])

                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_names } | Metrics: {metric_names } |', logger)
                print_log(f'Val[{idx + 1}/{n_samples}] | Losses: {loss_values} | Metrics: {metric_values} |\n', logger)

        header = [name.replace('Loss/', '') for name in val_losses.get_names()]
        values = ["%.4f" % l for l in val_losses.avg()]
        stds = ["%.4f" % l for l in val_losses.std()]
        _, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {header_str}  |', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | Losses: {values_str}  |\n', logger=logger)
        print_log(f'[Validation] EPOCH: {epoch} | STD +/-: {stds_str }  |\n', logger=logger)


    # Print validation results
    print_log('============================ VAL RESULTS ============================',logger=logger)
    header = ['#Samples'] + [name.replace('Loss/', '') for name in val_metrics.get_names()]
    values = [str(val_metrics.count(0))] + ["%.4f" % l for l in val_metrics.avg()]
    stds = ['STD +/-'] + ["%.4f" % l for l in val_metrics.std()]
    _, (header_str, values_str, stds_str) = get_table_str(header, values, stds)
    print_log(header_str, logger)
    print_log(values_str, logger)
    print_log(stds_str, logger)

    # Add validation results to TensorBoard
    if metric_logger is not None:
        metrics_log = {**advanced_metrics.get_epoch_log_dict(val=True), **val_losses.get_epoch_log_dict(val=True)}
        metric_logger.log_epoch_metrics(metrics_log, epoch, mode='val')

    return val_metrics.avg('CDL1')
