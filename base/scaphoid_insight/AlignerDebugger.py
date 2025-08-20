import time

import numpy as np
import torch
from torch import Tensor
import pyvista as pv

from base.scaphoid_utils.logger import print_log
import base.scaphoid_utils.transformations as t


def ensure_pcd_correctly_transposed(pcds: list):
    """
    Ensure that the point clouds in the list are correctly transposed to have shape (B, N, 3).
    :param pcds: List of point clouds, each with shape (B, N, 3) or (B, 3, N).
    :return: List of point clouds with shape (B, N, 3).
    """
    for i in range(len(pcds)):
        if pcds[i].shape[-1] != 3:
            pcds[i] = pcds[i].transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
    return pcds


class Debugger:

    def __init__(self, base_model, config, args, logger=None):
        self.base_model = base_model
        self.config = config
        self.args = args
        self.logger = logger


        self.in_out = base_model.module.in_out

    def visualize_completion(self, net_input: tuple[Tensor], full_pcd: Tensor, dense1: Tensor):
        """
        Visualize the completion dataset output and shows whether or not the dataset and dataloaders are working 
        correctly.
        :param net_input: Input point clouds for the network, should be a tuple of (volar, dorsal).
        :param full_pcd: Full point cloud, should be a tensor of shape (B, N, 3).
        :param dense1: Dense point cloud output from the network, should be a tensor of shape (B, N, 3).
        :return: None, but visualizes the point clouds in a 3D plot using PyVista.
        """
        p = pv.Plotter(window_size=[1600, 900], notebook=False)

        net_input_volar, net_input_dorsal = ensure_pcd_correctly_transposed(net_input)
        full_pcd = ensure_pcd_correctly_transposed([full_pcd])[0]
        dense1 = ensure_pcd_correctly_transposed([dense1])[0]

        offset = np.array([0, 0, 0.0])
        point_actor = p.add_points(torch_to_numpy(dense1[0])+offset, render_points_as_spheres=True, 
                                   point_size=60, color='yellow', opacity=0.05)
        offset = np.array([0, 0, 0.0])
        point_actor = p.add_points(torch_to_numpy(dense1[-1])+offset, render_points_as_spheres=True, 
                                   point_size=60, color='blue', opacity=0.05)

        # offset = np.array([0, 1.0, 0.0])
        # for batch_id in range(dense1.shape[0]):
        #     offset = offset + np.array([0.8, 0.0, 0.0])  # Increment offset for each batch to avoid overlap
        #     points = torch_to_numpy(dense1[batch_id]) + offset
        #     # Remove previous points if they exist
            # point_actor = p.add_points(points, render_points_as_spheres=True, 
            #                            point_size=20, color='green', opacity=1.0)


        batch_id = 0
        p.open_gif("completion.gif")  # Optional: record as gif, remove if not needed
        point_actor = None
        last_point_actor = None
        while True:
            
            for batch_id in range(dense1.shape[0]):
                offset = np.array([0, 0, 0.0])
                points = torch_to_numpy(dense1[batch_id]) + offset
                # Remove previous points if they exist
                point_actor = p.add_points(points, render_points_as_spheres=True, point_size=60, color='green', 
                                           opacity=1.0)
                if last_point_actor is not None:
                    # p.remove_actor(last_point_actor)
                    p.remove_actor(last_point_actor)
                
                
                
                time.sleep(0.2)  # Adjust sleep time as needed for visualization speed
                p.render()
                last_point_actor = point_actor
                p.write_frame()  # Only needed if recording gif
            # break
        p.close()  # Closes the window after all frames are rendered
        # If not recording gif, use p.show(auto_close=True) at the end instead of p.close()
        
        offset = np.array([0-3, 0, 0.0])
        p.add_points(torch_to_numpy(dense1[0])+offset, render_points_as_spheres=True, point_size=60, 
                     color='green', opacity=0.5)
        p.add_points(torch_to_numpy(dense1[-1])+offset, render_points_as_spheres=True, point_size=60, 
                     color='red', opacity=1.0)

        p.show()

    def visualize_pcd_aligner(self, full: Tensor, gt_RTS_mat: Tensor, org_full: Tensor, 
                              pred_RTS_mat: Tensor = None):
        """
        Visualize the point cloud aligner dataset output and shows whether or not the dataset and dataloaders are 
        working correctly.
        :param net_input: Full point cloud for the network, should be a tensor of shape (B, N, 3).
        :param gt_RTS_mat: Ground truth rotation, translation, and scale matrix for the point cloud, should be a tensor 
            of shape (B, 4, 4).
        :param org_full: Original point cloud, should be a tensor of shape (B, N, 3). On this pcd the RTS 
            transformations have not been applied yet.
        :param pred_RTS_mat: Predicted rotation, translation, and scale matrix for the point cloud, should be a tensor 
            of shape (B, 4, 4).
        :return: None, but visualizes the point clouds in a 3D plot using PyVista.
        """
        p = pv.Plotter(window_size=[1600, 900], notebook=False)

        full, org_full = ensure_pcd_correctly_transposed([full, org_full])

        full_gt_mapped = t.apply_reverse_RTS_transformation(full, gt_RTS_mat)
        full_pred_mapped = t.apply_reverse_RTS_transformation(full, pred_RTS_mat)
        # 7.5078 +- 2.6392, CDL2: 0.1374 +- 0.0930


        org_full = t.apply_S_of_RTS_transformation(org_full, gt_RTS_mat)
        full_gt_mapped = t.apply_S_of_RTS_transformation(full_gt_mapped, gt_RTS_mat)
        full_pred_mapped = t.apply_S_of_RTS_transformation(full_pred_mapped, pred_RTS_mat)


        batch_id = 0
        ########### Visualize original point clouds ###########
        offset = np.array([0, 0, 0.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(full[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(torch_to_numpy(org_full[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)

        ############ Visualize gt mapped point clouds ###########
        offset = np.array([0, 0, 1.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(org_full[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(full_gt_mapped[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='green', opacity=1.0)

        ############# Visualize pred mapped point clouds ###########
        offset = np.array([0, 0, 2.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(full_gt_mapped[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(full_pred_mapped[batch_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='green', opacity=1.0)

        p.show()


    def visualize_pose_est_comp(self, net_input: tuple[Tensor], gt_RTS_mats: tuple[Tensor], org_pcds: tuple[Tensor], 
                                pred_RTS_mats: Tensor = None, in_out: dict = None, full_pcd=None):
        """
        Visualize the pose estimation dataset output and shows whether or not the dataset and dataloaders are working 
        correctly.
        :param net_input: Input point clouds for the network, should be a tuple of (volar, dorsal)
        :param gt_RT_mats: Ground truth rotation, translation, and scale matrices for the point clouds, should be a 
            tuple of (volar, dorsal).
        :param org_pcds: Original point clouds, should be a tuple of (volar, dorsal, full). On those pcds the RTS 
            transformations have not been applied yet.
        :return: None, but visualizes the point clouds in a 3D plot using PyVista.
        """
        p = pv.Plotter(window_size=[1600, 900], notebook=False)
        p.show_grid()

        volar_aligned = in_out.get('volar_aligned', None)
        dorsal_aligned = in_out.get('dorsal_aligned', None)

        org_volar, org_dorsal, org_full = ensure_pcd_correctly_transposed(org_pcds)

        if volar_aligned is not None and dorsal_aligned is not None and full_pcd is not None:
            B, _, _ = volar_aligned.shape
            print_log("Visualizing pose estimation with aligned point clouds and full point cloud.")
            volar_aligned, dorsal_aligned, full_pcd = ensure_pcd_correctly_transposed([volar_aligned, 
                                                                                       dorsal_aligned, full_pcd])

            p.add_points(torch_to_numpy(volar_aligned[0]), render_points_as_spheres=True, 
                         point_size=5, color='blue', opacity=1.0)
            p.add_points(torch_to_numpy(dorsal_aligned[0]), render_points_as_spheres=True, 
                         point_size=5, color='orange', opacity=1.0)
            p.add_points(torch_to_numpy(full_pcd[0]), render_points_as_spheres=True, 
                         point_size=5, color='grey', opacity=1.0)
            # p.add_points(torch_to_numpy(org_full[0]), render_points_as_spheres=True, 
                        #  point_size=5, color='red', opacity=1.0)
            p.show()
            
            if B > 1:

                p = pv.Plotter(window_size=[1600, 900], notebook=False)
                p.add_points(torch_to_numpy(volar_aligned[1]), render_points_as_spheres=True, 
                             point_size=5, color='blue', opacity=1.0)
                p.add_points(torch_to_numpy(dorsal_aligned[1]), render_points_as_spheres=True, 
                             point_size=5, color='orange', opacity=1.0)
                p.add_points(torch_to_numpy(full_pcd[1]), render_points_as_spheres=True, 
                             point_size=5, color='grey', opacity=1.0)
                # p.add_points(torch_to_numpy(org_full[1]), render_points_as_spheres=True, 
                            #  point_size=5, color='red', opacity=1.0)
                p.show()

                p = pv.Plotter(window_size=[1600, 900], notebook=False)
                p.add_points(torch_to_numpy(volar_aligned[2]), render_points_as_spheres=True, 
                             point_size=5, color='blue', opacity=1.0)
                p.add_points(torch_to_numpy(dorsal_aligned[2]), render_points_as_spheres=True, 
                             point_size=5, color='orange', opacity=1.0)
                p.add_points(torch_to_numpy(full_pcd[2]), render_points_as_spheres=True, 
                             point_size=5, color='grey', opacity=1.0)
                # p.add_points(torch_to_numpy(org_full[2]), render_points_as_spheres=True, 
                #              point_size=5, color='red', opacity=1.0)
                p.add_points(torch_to_numpy(volar_aligned[3]), render_points_as_spheres=True, 
                             point_size=5, color='blue', opacity=1.0)
                p.add_points(torch_to_numpy(dorsal_aligned[3]), render_points_as_spheres=True, 
                             point_size=5, color='orange', opacity=1.0)
                p.add_points(torch_to_numpy(full_pcd[3]), render_points_as_spheres=True, 
                             point_size=5, color='grey', opacity=1.0)
                # p.add_points(torch_to_numpy(org_full[3]), render_points_as_spheres=True, 
                # point_size=5, color='red', opacity=1.0)
                p.show()

        p = pv.Plotter(window_size=[1600, 900], notebook=False)
        net_input_volar, net_input_dorsal = ensure_pcd_correctly_transposed(net_input)
        gt_RTS_v, gt_RTS_d = gt_RTS_mats
        pred_RTS_v, pred_RTS_d = pred_RTS_mats
        org_volar, org_dorsal, org_full = ensure_pcd_correctly_transposed(org_pcds)

        org_volar_mapped = t.apply_RTS_transformation(org_volar, gt_RTS_v)
        org_dorsal_mapped = t.apply_RTS_transformation(org_dorsal, gt_RTS_d)

        volar_gt_mapped = t.apply_reverse_RTS_transformation(net_input_volar, gt_RTS_v)
        dorsal_gt_mapped = t.apply_reverse_RTS_transformation(net_input_dorsal, gt_RTS_d)

        volar_pred_mapped = t.apply_reverse_RTS_transformation(net_input_volar, pred_RTS_v)
        dorsal_pred_mapped = t.apply_reverse_RTS_transformation(net_input_dorsal, pred_RTS_d)

        aligned = ensure_pcd_correctly_transposed([torch.cat([in_out['volar_aligned'], in_out['dorsal']], dim=2)])[0]

        b_id = 0
        offset = np.array([0, 0, 0.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(volar_gt_mapped[b_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(torch_to_numpy(volar_pred_mapped[b_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='grey', opacity=1.0)

        p.add_points(torch_to_numpy(dorsal_gt_mapped[b_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(torch_to_numpy(dorsal_pred_mapped[b_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='grey', opacity=1.0)

        p.add_points(torch_to_numpy(aligned[b_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='orange', opacity=1.0)


        volar_pred_aligned = t.apply_RTS_transformation(volar_pred_mapped, pred_RTS_d)

        p.add_points(torch_to_numpy(volar_pred_aligned[b_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='purple', opacity=1.0)
        p.add_points(torch_to_numpy(net_input_dorsal[b_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='pink', opacity=1.0)


        p.show()

    def visualize_pose_est(self, net_input: tuple[Tensor], gt_RTS_mats: tuple[Tensor], org_pcds: tuple[Tensor],
                           pred_RTS_mat: Tensor = None):
        """
        Visualize the pose estimation dataset output and shows whether or not the dataset and dataloaders are working 
        correctly.
        :param net_input: Input point clouds for the network, should be a tuple of (volar, dorsal, v_d) with v_d being 
            either the volar or dorsal point cloud depending on the context.
        :param gt_RT_mats: Ground truth rotation, translation, and scale matrices for the point clouds, should be a 
            tuple of (volar, dorsal, full).
        :param org_pcds: Original point clouds, should be a tuple of (volar, dorsal, full). On those pcds the RTS 
            transformations have not been applied yet.
        :return: None, but visualizes the point clouds in a 3D plot using PyVista.
        """
        p = pv.Plotter(window_size=[1600, 900], notebook=False)

        net_input_volar, net_input_dorsal, partial_to_align = ensure_pcd_correctly_transposed(net_input)
        gt_RTS_v, gt_RTS_d, gt_RTS_full = gt_RTS_mats
        org_volar, org_dorsal, org_full = ensure_pcd_correctly_transposed(org_pcds)

        org_volar_mapped = t.apply_RTS_transformation(org_volar, gt_RTS_v)
        org_dorsal_mapped = t.apply_RTS_transformation(org_dorsal, gt_RTS_d)

        input_volar_mapped = t.apply_reverse_RTS_transformation(net_input_volar, gt_RTS_v)
        input_dorsal_mapped = t.apply_reverse_RTS_transformation(net_input_dorsal, gt_RTS_d)

        gt_partial_to_align_mapped = t.apply_reverse_RTS_transformation(partial_to_align, gt_RTS_full)
        partial_to_align_mapped = t.apply_reverse_RTS_transformation(partial_to_align, pred_RTS_mat)

        pred_RT = self.in_out['6d_pose']

        batch_id = 0
        offset = np.array([0, 0, 0.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(org_volar[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(torch_to_numpy(org_dorsal[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(torch_to_numpy(org_full[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)

        offset = np.array([0, 0, -1.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(net_input_volar[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='blue', opacity=1.0)
        p.add_points(torch_to_numpy(net_input_dorsal[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='orange', opacity=1.0)

        # ####### Map Org to Input ########
        offset = np.array([0, 1.0, 0.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(org_volar_mapped[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(net_input_volar[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='blue', opacity=1.0)

        offset = np.array([0, 1.0, 1.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(org_dorsal_mapped[batch_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(net_input_dorsal[batch_id])+offset, render_points_as_spheres=True,
                     point_size=5, color='orange', opacity=1.0)

        ####### Map Input to GT ########
        offset = np.array([0, 3.0, 0.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(t.apply_RTS_rescale_transformation(org_volar, gt_RTS_v)[batch_id])+offset, 
                     render_points_as_spheres=True, point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(t.apply_RTS_rescale_transformation(org_dorsal, gt_RTS_d)[batch_id])+offset, 
                     render_points_as_spheres=True, point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(t.apply_RTS_rescale_transformation(input_volar_mapped, gt_RTS_v)[batch_id])+offset,
                     render_points_as_spheres=True, point_size=5, color='blue', opacity=1.0)
        p.add_points(torch_to_numpy(t.apply_RTS_rescale_transformation(input_dorsal_mapped, gt_RTS_d)[batch_id])+offset,
                     render_points_as_spheres=True, point_size=5, color='orange', opacity=1.0)

        offset = np.array([0, 3.0, 1.0])  # offset to avoid overlap in visualization
        p.add_points(torch_to_numpy(gt_partial_to_align_mapped[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(partial_to_align_mapped[batch_id])+offset, render_points_as_spheres=True, 
                     point_size=5, color='blue', opacity=1.0)

        p.show()
        



    def visualize(self, net_input, gt_affine_mats, org_partials):

        p = pv.Plotter(window_size=[1600, 900], notebook=False)

        net_input_volar, net_input_dorsal = net_input
        gt_affine_mat_v, gt_affine_mat_d = gt_affine_mats
        org_volar, org_dorsal = org_partials

        gt_reverse_affine_mat_v = t.get_reverse_affine_matrix(gt_affine_mat_v)
        gt_reverse_affine_mat_d = t.get_reverse_affine_matrix(gt_affine_mat_d)


        
        # check if in_out contains 'volar_aligned'
        if 'volar_aligned' in self.in_out:
            alignment = 'volar'
        elif 'dorsal_aligned' in self.in_out:
            alignment = 'dorsal'
        else:
            raise ValueError("No alignment found in in_out dictionary.")
        
        volar_aligned = self.in_out.get(f'volar_aligned', None)
        dorsal_aligned = self.in_out.get(f'dorsal_aligned', None)
        pred_affine_mat_d, pred_affine_mat_v = self.in_out['affine_mat_d'], self.in_out['affine_mat_v']

        gt_volar_reversed = t.apply_affine_transformation(net_input_volar, gt_reverse_affine_mat_v)
        gt_dorsal_reversed = t.apply_affine_transformation(net_input_dorsal, gt_reverse_affine_mat_d)

        pred_volar_reversed = t.apply_affine_transformation(net_input_volar, pred_affine_mat_v)
        pred_dorsal_reversed = t.apply_affine_transformation(net_input_dorsal, pred_affine_mat_d)

        b_id = 3    # batch id
        offset = np.array([0, 0, 1.0])  # offset to avoid overlap in visualization
        p.add_points(net_input_dorsal[b_id].squeeze().cpu().numpy()+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(net_input_volar[b_id].squeeze().cpu().numpy()+offset, render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        # p.add_points(torch_to_numpy(net_input_volar_aligned[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='red', opacity=1.0)
        # p.add_points(torch_to_numpy(gt_partial[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='green', opacity=1.0)
        # p.show()

        p.add_points(gt_volar_reversed[b_id].squeeze().cpu().numpy(), render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)
        p.add_points(gt_dorsal_reversed[b_id].squeeze().cpu().numpy(), render_points_as_spheres=True, 
                     point_size=5, color='grey', opacity=1.0)

        p.add_points(torch_to_numpy(pred_volar_reversed[b_id]), render_points_as_spheres=True, 
                     point_size=5, color='red', opacity=1.0)
        p.add_points(torch_to_numpy(pred_dorsal_reversed[b_id]), render_points_as_spheres=True, 
                     point_size=5, color='orange', opacity=1.0)

        if alignment == 'volar':
            offset = np.array([0, 1.0, 0])  # offset to avoid overlap in visualization
            dorsal_alignment_mat = t.get_reverse_affine_matrix(pred_affine_mat_d)

            org_volar_aligned = t.apply_affine_transformation(org_volar, dorsal_alignment_mat)
            org_dorsal_aligned = t.apply_affine_transformation(org_dorsal, dorsal_alignment_mat)
            p.add_points(torch_to_numpy(org_volar_aligned[b_id])+offset, render_points_as_spheres=True, 
                         point_size=5, color='yellow', opacity=1.0)
            p.add_points(torch_to_numpy(org_dorsal_aligned[b_id])+offset, render_points_as_spheres=True, 
                         point_size=5, color='yellow', opacity=1.0)
            

            
            p.add_points(net_input_dorsal[b_id].squeeze().cpu().numpy()+offset, render_points_as_spheres=True, 
                         point_size=5, color='orange', opacity=1.0)

            pred_volar_reversed = t.apply_affine_transformation(net_input_volar, pred_affine_mat_v)
            pred_volar_aligned = t.apply_affine_transformation(pred_volar_reversed, dorsal_alignment_mat)

            p.add_points(torch_to_numpy(pred_volar_aligned[b_id])+offset, render_points_as_spheres=True, 
                         point_size=5, color='blue', opacity=1.0)

            volar_aligned = volar_aligned.transpose(1, 2)  # transpose to match the shape of pred_volar_aligned
            p.add_points(torch_to_numpy(volar_aligned[b_id])+offset, render_points_as_spheres=True,
                         point_size=5, color='green', opacity=1.0)

 
        # p.add_points(torch_to_numpy(volar[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='blue', opacity=1.0)
        # p.add_points(torch_to_numpy(dorsal[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='yellow', opacity=1.0)
        # p.add_points(torch_to_numpy(net_input_volar_aligned[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='red', opacity=1.0)
        # p.add_points(torch_to_numpy(gt_partial[batch_id]), render_points_as_spheres=True, 
        #              point_size=5, color='green', opacity=1.0)
        p.show()


def torch_to_numpy(tensor):
    return tensor.squeeze().detach().cpu().numpy()