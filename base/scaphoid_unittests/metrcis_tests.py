import os, sys

import torch

# Add the main directory to sys.path such that submodules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from PointAttn.utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

from base.scaphoid_utils.logger import print_log
from base.scaphoid_metrics.weighted_dist_chamfer_3D import calc_weighted_cd

def test_calc_weighted_cd():
    size = 10
    output = torch.zeros((1, size, 3)) + torch.tensor([0, 0, 1]) * torch.arange(size).reshape(1, size, 1) + torch.tensor([0, 1, 0])
    gt = torch.zeros((1, size, 3)) + torch.tensor([0, 0, 1]) * torch.arange(size).reshape(1, size, 1) + torch.tensor([0, -1, 0])
    output = output.cuda()
    gt = gt.cuda()
    volar_ind = torch.zeros((1, size//2), dtype=torch.int64) + torch.arange(size//2).reshape(1, size//2)
    dorsal_ind = torch.zeros((1, size//2), dtype=torch.int64) + torch.arange(size//2).reshape(1, size//2) + 5
    pc_subdivision = {
        'volar': {'weighting': 2, 'ind': volar_ind.cuda()},
        'dorsal': {'weighting': 1, 'ind': dorsal_ind.cuda()},
    }

    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    true_dist1, true_dist2, true_idx1, true_idx2 = cham_loss(gt, output)
    detailed_metrics = calc_weighted_cd(output, gt, pc_subdivision, calc_f1=False, calc_detailed_metrics=True)

    cdl1 = (torch.sqrt(true_dist1).mean(1) + torch.sqrt(true_dist2).mean(1)) / 2
    cdl2 = (true_dist1.mean(1) + true_dist2.mean(1))


    print_log(f"Testing full metrics", color='blue')
    test_cdl2 = detailed_metrics['full']['CDL2']
    test_cdl1 = detailed_metrics['full']['CDL1']
    print_log(f"CDL1: {test_cdl1}, expected: {cdl1}", color='green' if test_cdl1 == cdl1 else 'red')
    print_log(f"CDL2: {test_cdl2}, expected: {cdl2}", color='green' if test_cdl2 == cdl2 else 'red')
    assert torch.allclose(test_cdl1, cdl1, atol=1e-5), f"Full CDL1: {test_cdl1} != {cdl1}"
    assert torch.allclose(test_cdl2, cdl2, atol=1e-5), f"Full CDL2: {test_cdl2} != {cdl2}"


    print_log(f"Testing volar metrics", color='blue')
    v_cdl1 = (torch.sqrt(true_dist1[0:5]).mean(1) + torch.sqrt(true_dist2[0:5]).mean(1)) / 2
    v_cdl2 = (true_dist1[0:5].mean(1) + true_dist2[0:5].mean(1))
    test_cdl2 = detailed_metrics['volar']['CDL2']
    test_cdl1 = detailed_metrics['volar']['CDL1']
    print_log(f"CDL1: {test_cdl1}, expected: {v_cdl1}", color='green' if test_cdl1 == v_cdl1 else 'red')
    print_log(f"CDL2: {test_cdl2}, expected: {v_cdl2}", color='green' if test_cdl2 == v_cdl2 else 'red')
    assert torch.allclose(test_cdl1, v_cdl1, atol=1e-5), f"Volar CDL1: {test_cdl1} != {v_cdl1}"
    assert torch.allclose(test_cdl2, v_cdl2, atol=1e-5), f"Volar CDL2: {test_cdl2} != {v_cdl2}"


    print_log(f"Testing dorsal metrics", color='blue')
    d_cdl1 = (torch.sqrt(true_dist1[0:5]).mean(1) + torch.sqrt(true_dist2[0:5]).mean(1)) / 2
    d_cdl2 = (true_dist1[0:5].mean(1) + true_dist2[0:5].mean(1))
    test_cdl2 = detailed_metrics['dorsal']['CDL2']
    test_cdl1 = detailed_metrics['dorsal']['CDL1']
    print_log(f"CDL1: {test_cdl1}, expected: {d_cdl1}", color='green' if test_cdl1 == d_cdl1 else 'red')
    print_log(f"CDL2: {test_cdl2}, expected: {d_cdl2}", color='green' if test_cdl2 == d_cdl2 else 'red')
    

    print_log(f"Testing full weighted metrics", color='blue')
    true_dist1[:, 0:5] *= 2
    true_dist2[:, 0:5] *= 2
    w_cdl1 = (torch.sqrt(true_dist1).mean(1) + torch.sqrt(true_dist2).mean(1)) / 2
    w_cdl2 = (true_dist1.mean(1) + true_dist2.mean(1))

    test_w_cdl2 = detailed_metrics['full_weighted']['CDL2']
    test_w_cdl1 = detailed_metrics['full_weighted']['CDL1']
    print_log(f"CDL1: {test_w_cdl1}, expected: {w_cdl1}", color='green' if test_w_cdl1 == w_cdl1 else 'red')
    print_log(f"CDL2: {test_w_cdl2}, expected: {w_cdl2}", color='green' if test_w_cdl2 == w_cdl2 else 'red')
    assert torch.allclose(test_w_cdl1, w_cdl1, atol=1e-5), f"Full weighted CDL1: {test_w_cdl1} != {w_cdl1}"
    assert torch.allclose(test_w_cdl2, w_cdl2, atol=1e-5), f"Full weighted CDL2: {test_w_cdl2} != {w_cdl2}"
    

if __name__ == "__main__":
    test_calc_weighted_cd()


