import torch


from PointAttn.utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from PointAttn.utils.ChamferDistancePytorch.fscore import fscore
from submodules.PoinTr.utils.metrics import Metrics

from base.scaphoid_utils.logger import print_log


def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    CDL1 = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    CDL2 = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore_cdl2(dist1, dist2)
        return CDL1, CDL2, f1
    else:
        return CDL1, CDL2
    

def fscore_cdl2(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    TODO: Currently does not handle batch size > 1 correctly.
    :param dist1: Batch, N-Points GT in L2 norm
    :param dist2: Batch, N-Points PRED
    :param th: float
    :return: fscore, precision, recall
    """
    gt = torch.sqrt(dist1)  # L1 norm needed
    pred = torch.sqrt(dist2)
    th = threshold
    device = pred.device

    b = pred.size(0)
    assert pred.size(0) == gt.size(0)
    if b != 1:
        f_score_list, precision_list, recall_list = [], [], []
        for idx in range(b):
            fscore_v, precision, recall = fscore_cdl2(pred[idx:idx+1], gt[idx:idx+1])
            f_score_list.append(fscore_v)
            precision_list.append(precision)
            recall_list.append(recall)
        return (sum(f_score_list)/len(f_score_list), 
                sum(precision_list)/len(precision_list), 
                sum(recall_list)/len(recall_list))
    else:
        recall = torch.mean((gt < th).float())
        precision = torch.mean((pred < th).float())

        fscore_v = torch.tensor(0., device=device)
        if recall + precision:
            fscore_v = 2 * recall * precision / (recall + precision)
    

    return fscore_v, precision, recall
    

def fscore_cdl1(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    TODO: Currently does not handle batch size > 1 correctly.
    :param dist1: Batch, N-Points GT in L1 norm
    :param dist2: Batch, N-Points PRED
    :param th: float
    :return: fscore, precision, recall
    """
    gt = dist1  # L1 norm needed
    pred = dist2
    th = threshold
    device = pred.device

    b = pred.size(0)
    assert pred.size(0) == gt.size(0)
    if b != 1:
        f_score_list, precision_list, recall_list = [], [], []
        for idx in range(b):
            fscore_v, precision, recall = fscore_cdl1(pred[idx:idx+1], gt[idx:idx+1])
            f_score_list.append(fscore_v)
            precision_list.append(precision)
            recall_list.append(recall)
        return (sum(f_score_list)/len(f_score_list), 
                sum(precision_list)/len(precision_list), 
                sum(recall_list)/len(recall_list))
    else:
        recall = torch.mean((gt < th).float())
        precision = torch.mean((pred < th).float())

        fscore_v = torch.tensor(0., device=device)
        if recall + precision:
            fscore_v = 2 * recall * precision / (recall + precision)
        
    return fscore_v, precision, recall


def calc_weighted_cd(output, gt, pc_subdivision, calc_f1=False, calc_detailed_metrics=False, 
                     scaled_real_world_thresh=None):
    """
    :param output: [b, 16.000, 3]
    :param gt: [b, 8.000, 3]
    :param pc_subdivision: dict of the form e.g. 
        {'volar': {'weighting': 2, 'ind': volar_ind}, 'dorsal': {'weighting': 1, 'ind': dorsal_ind}}
    
    dist1: distance from each point in gt to the nearest point in output
    dist2: distance from each point in output to the nearest point in gt
    idx1: for each point in gt, the index of the nearest point in output -> contains ids from output
    idx2: for each point in output, the index of the nearest point in gt -> contains ids from gt

    :return: dict of the form {'Full': {'CDL1': [b], 'CDL2': [b], 'F1': [b]}, 
        'Full_weighted': {'cd_p': [b], 'cd_t': [b], 'F1': [b]}, ...}
    """
    # print_log(f"calc_weighted_cd: {output.shape}, {gt.shape}", color='green')
    if pc_subdivision is None:
        pc_subdivision = {}

    detailed_metrics = {}

    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    
    # calc detailed metrics (for each label e.g. 'volar', 'dorsal', 'distal', etc.)
    if calc_detailed_metrics:
        for key, value in pc_subdivision.items():

            index = value['ind']
            dist1_sub = torch.gather(dist1, 1, index)
            dist2_sub = torch.gather(dist2, 1, torch.gather(idx1, 1, index).to(torch.int64))

            detailed_metrics[key] = {}
            dist1_sub_cdl1 = torch.sqrt(dist1_sub)
            dist2_sub_cdl1 = torch.sqrt(dist2_sub)
            detailed_metrics[key]['CDL1'] = (dist1_sub_cdl1.mean(1) + dist2_sub_cdl1.mean(1)) / 2
            detailed_metrics[key]['CDL2'] = (dist1_sub.mean(1) + dist2_sub.mean(1))
            if calc_f1:
                f1, prec, rec = fscore_cdl1(dist1_sub_cdl1, dist2_sub_cdl1)
                detailed_metrics[key]['F1'] = f1
                detailed_metrics[key]['precision'] = prec
                detailed_metrics[key]['recall'] = rec
            if scaled_real_world_thresh is not None:
                if type(scaled_real_world_thresh) == list:
                    for i, (name, thresh) in enumerate(scaled_real_world_thresh):
                        thresh = thresh.item()
                        f1, prec, rec = fscore_cdl1(dist1_sub_cdl1, dist2_sub_cdl1, threshold=thresh)
                        detailed_metrics[key][f'F1_{name}'] = f1
                        detailed_metrics[key][f'precision_{name}'] = prec
                        detailed_metrics[key][f'recall_{name}'] = rec
                else:
                    raise ValueError("scaled_real_world_thresh should be a list of tuples (name, threshold) or a \
                                     single threshold value.")


    # calc full and full_weighted metrics
    weights1, weights2 = torch.ones_like(dist1).cuda(), torch.ones_like(dist2).cuda()
    
    
    keys = []
    weights = []
    for key, value in pc_subdivision.items():
        keys.append(key)
        weights.append(value['weighting'])

    # order ensures that the highest weighting is applied last
    sorted_pairs = sorted(zip(weights, keys))

    for weighting, key in sorted_pairs:
        if weighting == 1:
            continue

        inds = pc_subdivision[key]['ind']
        inds = inds.cuda()
        weights1[:, inds] = weighting
        # weights1[:, inds] = torch.max(weights1[:, inds], weighting)

        idx2_expanded = idx2.unsqueeze(2)
        inds_expanded = inds.unsqueeze(1)
        match = (idx2_expanded == inds_expanded)

        matched_any = match.any(dim=2)
        # print_log(f"matched_any: {matched_any.shape}, {weights2.shape}", color='blue')
        weights2[matched_any] = weighting


    detailed_metrics['Full'] = {}
    dist1_cdl1, dist2_cdl1 = torch.sqrt(dist1), torch.sqrt(dist2)
    detailed_metrics['Full']['CDL1'] = (dist1_cdl1.mean(1) + dist2_cdl1.mean(1)) / 2
    # detailed_metrics['Full']['CDL1'] = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    detailed_metrics['Full']['CDL2'] = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, prec, rec = fscore_cdl1(dist1_cdl1, dist2_cdl1)
        detailed_metrics['Full']['F1'] = f1
        detailed_metrics['Full']['precision'] = prec
        detailed_metrics['Full']['recall'] = rec
    if scaled_real_world_thresh is not None:
        if type(scaled_real_world_thresh) == list:
            for i, (name, thresh) in enumerate(scaled_real_world_thresh):
                thresh = thresh.item()
                f1, prec, rec = fscore_cdl1(dist1_sub_cdl1, dist2_sub_cdl1, threshold=thresh)
                detailed_metrics['Full'][f'F1_{name}'] = f1
                detailed_metrics['Full'][f'precision_{name}'] = prec
                detailed_metrics['Full'][f'recall_{name}'] = rec
        else:
            raise ValueError("scaled_real_world_thresh should be a list of tuples (name, threshold) or a single \
                             threshold value.")
    
    # dist1 = dist1 * weights1
    # dist2 = dist2 * weights2
    # detailed_metrics['Full_weighted'] = {}
    # detailed_metrics['Full_weighted']['CDL1'] = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    # detailed_metrics['Full_weighted']['CDL2'] = (dist1.mean(1) + dist2.mean(1))
    # if calc_f1:   does not make sense to calculate f1 for weighted metrics
    #     f1, prec, rec = fscore(dist1, dist2)
    #     detailed_metrics['Full_weighted']['F1'] = f1
    #     detailed_metrics['Full_weighted']['precision'] = prec
    #     detailed_metrics['Full_weighted']['recall'] = rec

    return detailed_metrics