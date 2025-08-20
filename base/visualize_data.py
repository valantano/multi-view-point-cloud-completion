import time
import os
import pyvista as pv
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


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

from base.scaphoid_datasets.Transforms import ReverseTransforms
from base.scaphoid_datasets.ScaphoidDataset import ScaphoidDataset

from base.scaphoid_datasets.Transforms import RandomMirrorPoints
# from base.scaphoid_datasets.RotationDataset import PoseEstScaphoidDataset


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' %
              args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

def visualize_transform_differences(args, config, logger):
    transforms = None
    transform_with = None
    try:
        transforms = config.dataset.transforms
        transform_with = config.dataset.transform_with
    except:
        pass
    transforms = [
    'CoupledDemeaning', 
    'CoupledRandomRotation',
    'CoupledRescale'
    ]
    transforms = [
    'DecoupledDemeaning', 
    'DecoupledRandomRotation',
    'DecoupledRescale'
    ]
    transforms = [
    'StaticDecoupledDemeaning', 
    'StaticDecoupledRandomRotation',
    'StaticDecoupledRescale'
    ]
    transforms = [
    'StaticDecoupledDemeaning', 
    'StaticDecoupledRandomRotation',
    'CoupledRescale'
    ]
    args.debug = True
    # t_dataset = EnrichedScaphoidDataset('train', config.dataset.train._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)
    v_dataset = ScaphoidDataset('val', config.dataset.val._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)

    # train_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=1, shuffle=True, drop_last=True, 
    #                                               num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    valid_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=2, shuffle=False, drop_last=False,
                                                  num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    
    transforms = []
    second_v_dataset = ScaphoidDataset('val', config.dataset.val._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)
    second_v_dataloader = torch.utils.data.DataLoader(second_v_dataset, batch_size=2, shuffle=False, drop_last=False,
                                                  num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)

    for idx, [(taxonomy_ids, model_ids, data), (tax_ids, m_ids, data2)] in enumerate(zip(valid_dataloader, second_v_dataloader)):

        partial_volar, partial_dorsal, gt = data[0].numpy(), data[1].numpy(), data[2].numpy()
        partial_volar2, partial_dorsal2, gt2 = data2[0].numpy(), data2[1].numpy(), data2[2].numpy()
        
        transform_paras = data[4]
        transform_paras2 = data2[4]

        plotter = pv.Plotter()
        pv.set_plot_theme("document")
        B, N, C = gt.shape
        B = 2
        rev_trans = ReverseTransforms()
        reversed_gt = rev_trans(gt, transform_paras['transform_gt'])
        reversed_v = rev_trans(partial_volar, transform_paras['transform_volar'])
        reversed_d = rev_trans(partial_dorsal, transform_paras['transform_dorsal'])
        for i in range(B):
            batch_offset = np.array([i*40, 0, 0])

            gt_offset = np.array([0, 0, 0]) + batch_offset
            plotter.add_points(gt[i] + gt_offset, color=const.gt_unfocused_rgb, render_points_as_spheres=True)
            plotter.add_points(gt2[i] + gt_offset, color=const.red_rgb, render_points_as_spheres=True)
            plotter.add_points(reversed_gt[i] + gt_offset, color=const.green_rgb, render_points_as_spheres=True)

            volar_offset = np.array([0,20,0]) + batch_offset
            plotter.add_points(partial_volar[i] + volar_offset, color=const.gt_unfocused_rgb, render_points_as_spheres=True)
            plotter.add_points(partial_volar2[i] + volar_offset, color=const.red_rgb, render_points_as_spheres=True)
            plotter.add_points(reversed_v[i] + volar_offset, color=const.green_rgb, render_points_as_spheres=True)

            dorsal_offset = np.array([0, 20, 0]) + batch_offset
            plotter.add_points(partial_dorsal[i] + dorsal_offset, color=const.gt_unfocused_rgb, render_points_as_spheres=True)
            plotter.add_points(partial_dorsal2[i] + dorsal_offset, color=const.red_rgb, render_points_as_spheres=True)
            plotter.add_points(reversed_d[i] + dorsal_offset, color=const.green_rgb, render_points_as_spheres=True)

            gt_succs = np.allclose(gt2[i], reversed_gt[i], atol=1e-5)
            volar_succs = np.allclose(partial_volar2[i], reversed_v[i], atol=1e-5)
            dorsal_succs = np.allclose(partial_dorsal2[i], reversed_d[i], atol=1e-5)
            color = 'green' if gt_succs and volar_succs and dorsal_succs else 'red'
            print_log(f"GT Success: {gt_succs}, Volar Success: {volar_succs}, Dorsal Success: {dorsal_succs}", color=color)

        plotter.show()

from scipy.spatial.transform import Rotation
from base.scaphoid_utils.transformations import get_affine_matrix, get_reverse_affine_matrix, apply_affine_transformation

# def visualize_rotation_dataset(args, config, logger):
#     transforms = []
#     transform_with = 'does not matter'

#     args.debug = True
#     # t_dataset = EnrichedScaphoidDataset('train', config.dataset.train._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)
#     v_dataset = PoseEstScaphoidDataset('val', config.dataset.val._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)

#     # train_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=1, shuffle=True, drop_last=True, 
#     #                                               num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
#     valid_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=2, shuffle=False, drop_last=False,
#                                                   num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    
#     for idx, (data) in enumerate(valid_dataloader):
        

#         partial_volar, partial_dorsal, gt_dorsal = data[0], data[1], data[2]

#         transform_paras = data[3]

#         rotated_partial_dorsal = partial_dorsal.clone()

#         print_log(f"Transform paras: {transform_paras}", color='blue')

#         affine_matrix = transform_paras['affine_dorsal']
#         affine_matrix_gt = transform_paras['affine_gt']
#         print_log(f"Affine matrix: {affine_matrix}", color='red')
#         print_log(f"Affine matrix gt: {affine_matrix_gt}", color='red')

#         reverse_affine_matrix2 = get_reverse_affine_matrix(affine_matrix)
#         reverse_gt = get_reverse_affine_matrix(affine_matrix_gt)

#         print_log(f"Reverse affine matrix: {affine_matrix_gt@reverse_gt}", color='red')

#         combined_affine_matrix = torch.bmm(affine_matrix_gt, reverse_affine_matrix2)
#         rotated_partial_dorsal = apply_affine_transformation(rotated_partial_dorsal, combined_affine_matrix)

#         # translation2 = transform_paras2['transform_dorsal'][0]['demean'].squeeze()
#         # rotation2 = transform_paras2['transform_dorsal'][1]['rotation'].squeeze()
#         # rescale2 = transform_paras2['transform_dorsal'][2]['rescale']
#         # translation = transform_paras['transform_dorsal'][0]['demean'].squeeze()
#         # rotation = transform_paras['transform_dorsal'][1]['rotation'].squeeze()
#         # rescale = transform_paras['transform_dorsal'][2]['rescale']

#         # affine_matrix = get_affine_matrix(rescale, rotation, translation)
#         # affine_matrix2 = get_affine_matrix(rescale2, rotation2, translation2)
#         # print_log(f"Affine matrix: {affine_matrix}", color='red')
#         # print_log(f"Affine matrix: {affine_matrix2}", color='red')
#         # reverse_affine_matrix2 = get_reverse_affine_matrix(affine_matrix2)
        

#         # combined_affine_matrix = affine_matrix @ reverse_affine_matrix2
#         # rotated_partial_dorsal = apply_affine_transformation(rotated_partial_dorsal, combined_affine_matrix)
#         # print_log(f"Affine matrix: {affine_matrix}", color='red')
#         # print_log(f"Reverse affine matrix: {reverse_affine_matrix2}", color='red')


       
#         # rotated_partial_dorsal = apply_affine_transformation(rotated_partial_dorsal, reverse_affine_matrix2)
#         # rotated_partial_dorsal = apply_affine_transformation(rotated_partial_dorsal, affine_matrix_gt)

#         # reverse_gt = apply_affine_transformation(gt_dorsal, reverse_gt)
#         # reverse_gt = apply_affine_transformation(gt_dorsal, affine_matrix_gt)
        
#         # gt_dorsal = apply_affine_transformation(gt_dorsal, reverse_gt)



        

#         ################ Plotting ################
#         plotter = pv.Plotter()
#         plotter.show_grid()
#         pv.set_plot_theme("document")
#         plotter.add_points(partial_volar[0].numpy() + np.array([1,0,0]), color=const.gt_unfocused_rgb, render_points_as_spheres=True)
#         plotter.add_points(partial_dorsal[0].numpy() + np.array([1,0,0]), color=const.gt_unfocused_rgb, render_points_as_spheres=True)
#         plotter.add_points(gt_dorsal[0].numpy(), color=const.red_rgb, render_points_as_spheres=True)
#         plotter.add_points(rotated_partial_dorsal[0].numpy(), color=const.green_rgb, render_points_as_spheres=True)

#         plotter.show()




def reverse_rescale(to_be_reversed, scale_vals):
    """
    Reverse rescale
    :param to_be_reversed: point cloud to reverse
    :param scale_vals: scale values containing p_min, range_min, and scale
    :return: reversed point cloud
    """
    # return to_be_reversed
    p_min, range_min, scale = scale_vals
    if type(to_be_reversed) != type(p_min):
        p_min = np.array(p_min)
        range_min = np.array(range_min)
        scale = np.array(scale)
    if scale == 0:
        raise ValueError("Scale factor cannot be zero.")
    return (to_be_reversed - range_min + (p_min * scale)) / scale


def rotation_matrix_from_angles(roll, pitch, yaw):
    # Rotation matrix around X-axis (roll)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])

    # Rotation matrix around Y-axis (pitch)
    Ry = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])

    # Rotation matrix around Z-axis (yaw)
    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def main():
    # args
    args = parser.get_args()

    ###### Experiment Path ######
    exp_batch_path = os.path.join('./mvpcc_experiments', Path(args.config).parent.stem)
    args.experiment_path = os.path.join(exp_batch_path, Path(args.config).stem, args.exp_name)
    args.tfboard_path = os.path.join(exp_batch_path, Path(args.config).stem, 'TFBoard', args.exp_name)
    args.log_name = args.experiment_path

    create_experiment_dir(args)
    ##############################


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
    metric_logger = TensorboardLogger(args, config)
    logger = Logger(args.log_name)

    # build dataset
    

    # visualize_transform_differences(args, config, logger)

    transforms = [
    # 'DecoupledDemeaning', 
    # 'DecoupledRandomRotation',
    # 'CoupledRescale'
    ]
    transform_with = 'dorsal'
    args.debug = True
    # t_dataset = EnrichedScaphoidDataset('train', config.dataset.train._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)
    v_dataset = ScaphoidDataset('val', config.dataset.val._base_, transforms, transform_with=transform_with, debug=args.debug, logger=logger)

    # train_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=1, shuffle=True, drop_last=True, 
    #                                               num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    valid_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=2, shuffle=False, drop_last=False,
                                                  num_workers=int(args.num_workers), worker_init_fn=worker_init_fn)
    

    for idx, data in enumerate(valid_dataloader):

        transformed_data, _, inds, transform_paras = data


        partial_volar, partial_dorsal, gt = transformed_data[0].numpy(), transformed_data[1].numpy(), transformed_data[2].numpy()
        
        gt_mirr0 = RandomMirrorPoints(None)(gt[0].copy(), 0.0)
        gt_mirr1 = RandomMirrorPoints(None)(gt[0].copy(), 0.4)
        gt_mirr2 = RandomMirrorPoints(None)(gt[0].copy(), 0.7)
        gt_mirr3 = RandomMirrorPoints(None)(gt[0].copy(), 0.8)

        plotter = pv.Plotter()
        pv.set_plot_theme("document")
        B, N, C = gt.shape
        B = 2
        plotter.add_points(gt[0], color='green', render_points_as_spheres=True)
        # plotter.add_points(gt_mirr0, color=const.red_rgb, render_points_as_spheres=True)
        # plotter.add_points(gt_mirr1, color=const.blue_rgb, render_points_as_spheres=True)
        plotter.add_points(gt_mirr0, color=const.yellow_rgb, render_points_as_spheres=True)
        # plotter.add_points(gt_mirr3, color='purple', render_points_as_spheres=True)

        plotter.show()
    
    

        # plotter.add_points(partial_volar[0], color=const.partial_volar_rgb, render_points_as_spheres=True)
        # plotter.add_points(partial_dorsal[0], color=const.partial_dorsal_rgb, render_points_as_spheres=True)

        # def log_image(ptc: np.array, name: str, reverse_transforms: bool, transform_paras=None):
        #     if reverse_transforms:
        #         rev_trans = ReverseTransforms()
        #         ptc = rev_trans(ptc, transform_paras).squeeze()
        #     ptc_img = get_ptcloud_img(ptc)
        #     return ptc_img
        #     # metric_logger.add_image(name, ptc_img, epoch, mode='val')
        #     rev_trans = ReverseTransforms()
        # reversed_gt = 
        # plotter.add_points(, color=const.gt_unfocused_rgb, render_points_as_spheres=True)

        # gt = data[2].squeeze().numpy()
        # partial_volar = data[0].squeeze().numpy()
        # partial_dorsal = data[1].squeeze().numpy()

        # [(210, 45)] # z
        # # y [(60, 90), (690, 270)]
        # # open the images
        # angle_pairs = [(60, 90), (60, 90+180)]
        # for el, az in angle_pairs:
        #     print(f"Elevation: {el}, Azimuth: {az}")
        #     def log_image(ptc: np.array, name: str, reverse_transforms: bool, transform_paras=None):
        #         if reverse_transforms:
        #             rev_trans = ReverseTransforms()
        #             ptc = rev_trans(ptc, transform_paras).squeeze()
        #         ptc_img = get_ptcloud_img(ptc, el, az)
        #         return ptc_img
        #     gt_img = log_image(gt, 'gt', reverse_transforms=False, transform_paras=transform_paras['transform_gt'])
        #     plt.imshow(gt_img)
        #     plt.show()

        # v_img = log_image(partial_volar, 'partial_volar', reverse_transforms=True, transform_paras=transform_paras['transform_volar'])
        # d_img = log_image(partial_dorsal, 'partial_dorsal', reverse_transforms=True, transform_paras=transform_paras['transform_dorsal'])
        
        # plt.imshow(v_img)
        # plt.show()
        # plt.imshow(d_img)
        # plt.show()

        # metric_logger.add_input_3d('visualize', gt, [partial_volar, partial_dorsal], idx, mode='val')
        # plotter.show()


if __name__ == '__main__':
    main()
