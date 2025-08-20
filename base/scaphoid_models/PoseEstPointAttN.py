from __future__ import print_function
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scipy.stats import special_ortho_group

from submodules.PointAttn.utils.mm3d_pn2 import furthest_point_sample, gather_points

from base.scaphoid_models.ArchBuilder import ArchConfigBuilder
from base.scaphoid_models import ScaphoidModules
from base.scaphoid_metrics.weighted_dist_chamfer_3D import calc_weighted_cd
from base.scaphoid_utils.AttnWeightsCollector import AttnWeightsCollector
from base.scaphoid_utils.logger import print_log
import base.scaphoid_models.ScaphoidModules as modules
from base.scaphoid_models.ScaphoidPointAttN import ScaphoidPointAttN
import base.scaphoid_utils.transformations as t
import base.scaphoid_utils.scaphoid_utils as utils


store_attn_weights = False
collector = AttnWeightsCollector()

ScaphoidModules.store_attn_weights = store_attn_weights
ScaphoidModules.collector = collector

def set_store_attn_weights(value: bool):
    """
    Set the store_attn_weights variable to True or False.
    :param value: True or False
    """
    global store_attn_weights
    store_attn_weights = value
    ScaphoidModules.store_attn_weights = value

        
class PoseEstPointAttN(nn.Module):
    def __init__(self, config):
        super(PoseEstPointAttN, self).__init__()
        print(config.dataset)
        
        self.N_POINTS = config.model.num_points   # should be 8192
        
        self.arch_builder = ArchConfigBuilder(config)

        self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
        self.in_out = None

    def forward(self, xyz_volar: Tensor, xyz_dorsal: Tensor, scale_paras: Tensor):
        """
        Forward pass for the PoseEstPointAttN model.
        :param xyz_volar: Volar point cloud tensor of shape [B, 3, N]
        :param xyz_dorsal: Dorsal point cloud tensor of shape [B, 3, N]
        :param scale_paras: Scale parameters tensor of shape [B, 4]
        :return: in_out dictionary containing intermediate outputs and final outputs
        """
        batch_size, _, N = xyz_volar.size()

        if N >= 2048:
            xyz_volar = gather_points(xyz_volar, furthest_point_sample(xyz_volar.transpose(1, 2).contiguous(), 2048))
            xyz_dorsal = gather_points(xyz_dorsal, furthest_point_sample(xyz_dorsal.transpose(1, 2).contiguous(), 2048))

        in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

        if store_attn_weights:              # store attn weights for visualization
            collector.add_in_out(in_out)

        for block in self.net_blocks:

            if block.type == 'PE':     ############################## Pose Extractor ##################################
                pe_input = in_out[block.input]

                pose_code = block(pe_input)

                in_out[block.output] = pose_code

            elif block.type == 'PoseG':    ############################## Pose Generator ##############################
                pose_code = in_out[block.pose_code]

                pose_6d = block(pose_code, batch_size)

                pred_RT_mat = pose_6d
                scale = scale_paras[:, 2]   # scale
                pred_RTS_mat = torch.zeros((batch_size, 4, 4), dtype=torch.float32).to(pose_6d.device)
                pred_RTS_mat[:, :3, :3] = pred_RT_mat[:, :3, :3].type(torch.float32)
                pred_RTS_mat[:, :3, 3] = pred_RT_mat[:, :3, 3].type(torch.float32) / scale.unsqueeze(1)
                pred_RTS_mat[:, 3] = scale_paras    # scale

                #######################################################################################################
                #           |----------------|                   |----------------------------|
                #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
                # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
                #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
                #           |----------------|                   | p_min, range_min, scale, 1 |
                #                                                |----------------------------|     
                #######################################################################################################

                in_out[block.output] = pred_RTS_mat

            else:
                raise ValueError(f"Unknown block type {block.type}. Available types are: \
                                 {self.arch_builder.available_types}")
            
        self.in_out = in_out
        return in_out

    
    @staticmethod
    def get_loss(pred_RTS_mat: Tensor, gt_RTS_mat: Tensor):
        """
        Get the loss for the model.
        :param pred_RTS_mat: predicted 6D pose [B,4,4]
        :param gt_RTS_mat: ground truth 6D pose [B,4,4]
        :return: loss
        """
        gt_RT_mat = gt_RTS_mat[:, :3, :4]  # Ensure gt_6d_pose is in the correct shape [B,3,4] [R|t]
        pred_RT_mat = pred_RTS_mat[:, :3, :4]  # Ensure pred_6d_pose is in the correct shape [B,3,4] [R|t]

        pred_R = pred_RT_mat[:, :3, :3]
        gt_R = gt_RT_mat[:, :3, :3]
        geodesic_loss = PoseEstPointAttN.get_geodesic_loss(pred_R, gt_R)

        pred_t = pred_RT_mat[:, :3, 3]
        gt_t = gt_RT_mat[:, :3, 3]
        translation_loss = PoseEstPointAttN.get_translation_loss(pred_t, gt_t)

        total_loss = translation_loss + 5.0 * geodesic_loss

        loss_dict = {'Loss/Total': total_loss.mean(), 'Loss/Rotation': geodesic_loss.mean(), 
                     'Loss/Translation': translation_loss.mean()}
        return loss_dict

    @staticmethod
    def get_translation_loss(pred_t: Tensor, gt_t: Tensor):
        """
        Get the translation loss for the model.
        Euclidean distance between predicted and ground truth translation vectors.
        :param pred_t: predicted translation vector [B,3]
        :param gt_t: ground truth translation vector [B,3]
        :return: translation loss
        """
        translation_loss = torch.norm(pred_t - gt_t, dim=-1)
        return translation_loss

    @staticmethod
    def get_geodesic_loss(pred_R, gt_R):
        """
        Get the geodesic loss for the model.
        :param pred_R: predicted rotation matrix [B,3,3]
        :param gt_R: ground truth rotation matrix [B,3,3]
        :return: geodesic loss
        """
        R_diff = torch.bmm(gt_R, pred_R.transpose(1, 2)) # = torch.bmm(pred_R.transpose(1, 2), gt_R)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        geodesic_loss = torch.acos(torch.clamp((trace - 1) / 2, -1+1e-6, 1-1e-6))  
        # Ensure the value is in the range [-1, 1] for acos and avoid numerical issues by adding a small epsilon
        
        return geodesic_loss
    
    @staticmethod
    def get_euler_angle_difference(pred_R, gt_R):
        """
        Get the difference in Euler angles between predicted and ground truth rotation matrices.
        :param pred_R: predicted rotation matrix [B,3,3]
        :param gt_R: ground truth rotation matrix [B,3,3]
        :return: difference in Euler angles
        """
        degrees = True  # Set to True if you want the angles in degrees, False for radians
        pred_euler = utils.rotation_matrix_to_euler_angles(pred_R, degrees)
        gt_euler = utils.rotation_matrix_to_euler_angles(gt_R, degrees)
        diff = pred_euler - gt_euler

        if degrees:
            diff = (diff + 180) % 360 - 180
        else:
            diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        return diff.abs()
    
    @staticmethod
    def get_translation_difference(pred_t, gt_t):
        """
        Get the difference in translation vectors between predicted and ground truth translations.
        :param pred_t: predicted translation vector [B,3]
        :param gt_t: ground truth translation vector [B,3]
        :return: difference in translation vectors
        """
        diff = pred_t - gt_t
        diff_x, diff_y, diff_z = diff[:, 0], diff[:, 1], diff[:, 2]
        return diff_x.abs(), diff_y.abs(), diff_z.abs()

    @staticmethod
    def get_metrics(pred_RTS_mat, gt_RTS_mat, transformed_pcd, untransformed_pcd, calc_f1=False):
        """
        Get the metrics for the model.
        :param pred_RTS_mat: predicted 6D pose [B,4,4]
        :param gt_RTS_mat: ground truth 6D pose
        :param transformed_pcd: transformed point cloud
        :param untransformed_pcd: untransformed point cloud
        :param calc_f1: whether to calculate F1 score
        :return: metrics dictionary
        """
        B = pred_RTS_mat.shape[0]

        gt_transformed_pcd = t.apply_RTS_transformation(untransformed_pcd, pred_RTS_mat)


        metrics = calc_weighted_cd(transformed_pcd, gt_transformed_pcd, {}, calc_detailed_metrics=False, calc_f1=calc_f1)

        metrics = metrics['Full']

        tmp_metrics = {}
        for key, value in metrics.items():
            tmp_metrics[f"Pose/{key}"] = torch.mean(value)
        metrics = tmp_metrics

        rotation_metrics = PoseEstPointAttN.get_euler_angle_difference(pred_RTS_mat[:, :3, :3], gt_RTS_mat[:, :3, :3])
        rotation_x, rotation_y, rotation_z = rotation_metrics[:, 0], rotation_metrics[:, 1], rotation_metrics[:, 2]
        metrics['Rotation/Mean_X'] = rotation_x.mean()
        metrics['Rotation/Mean_Y'] = rotation_y.mean()
        metrics['Rotation/Mean_Z'] = rotation_z.mean()

        metrics['Rotation/Mean'] = (rotation_x + rotation_y + rotation_z).mean()

        translation_metrics = PoseEstPointAttN.get_translation_difference(pred_RTS_mat[:, :3, 3], gt_RTS_mat[:, :3, 3])
        metrics['Translation/Mean_X'] = translation_metrics[0].mean()
        metrics['Translation/Mean_Y'] = translation_metrics[1].mean()
        metrics['Translation/Mean_Z'] = translation_metrics[2].mean()


        return metrics


class CompletionPoseEstPointAttN(nn.Module):
    def __init__(self, config, pretrained: bool = False):
        super(CompletionPoseEstPointAttN, self).__init__()
        
        self.N_POINTS = config.model.num_points   # should be 8192

        self.arch_builder = ArchConfigBuilder(config, pretrained)

        self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
        self.in_out = None

    def forward(self, xyz_volar: Tensor, xyz_dorsal: Tensor, scale_paras: Tensor):
        """
        Forward pass for the completion model.
        :param xyz_volar: Volar point cloud tensor of shape [B, 3, N]
        :param xyz_dorsal: Dorsal point cloud tensor of shape [B, 3, N]
        :param scale_paras: Scale parameters tensor of shape [B, 3]
        :return: in_out dictionary containing intermediate outputs and final outputs
        """
        batch_size, _, N = xyz_volar.size()

        in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

        if store_attn_weights:              # store attn weights for visualization
            collector.add_in_out(in_out)


        for block in self.net_blocks:

            if block.type == 'ALIGNER':   #############################################################################
                op_input = [in_out[block.src_pose], in_out[block.tgt_pose], in_out[block.src], in_out[block.tgt]]

                op_output = block(op_input)
                support_pcds, anchor_pcds = op_output

                
                if block.mode == 'aligner_SSM': # needs post processing
                    augmented_pcds, augment_RTS = t.augment_pcds([support_pcds, anchor_pcds], rdm_rot=True, 
                                                                 demean=True, scale_paras=scale_paras)
                    support_pcds, anchor_pcds = augmented_pcds

                    in_out[f"GT_align_RTS"] = in_out[block.tgt_pose]
                    in_out[f"GT_augment_RTS"] = augment_RTS

                elif block.mode == 'aligner_anchor':
                    # no post processing needed
                    pass

                in_out[f'{block.src}_aligned'] = support_pcds
                in_out[f'{block.tgt}_aligned'] = anchor_pcds


            elif block.type == 'OP':      #############################################################################
                op_input = [in_out[inp] for inp in block.input]

                op_output = block(op_input)

                in_out[block.output] = op_output


            elif block.type == 'FE':      ########################################## Feature Extractor ################
                fe_input = in_out[block.input]
                in_out['input_logging'] = fe_input

                fe_output = block.forward(fe_input)

                in_out[block.output] = fe_output

            elif block.type == 'SG':      ########################################### Seed Generator ##################
                if block.mode == 'pointattn' and block.mode != 'SG-':
                    sg_input = in_out[block.input][:, :3, :] # [B, C, N] -> [B, 3, N]

                sg_shape_code = in_out[block.shape_code]

                seeds = block(sg_shape_code, batch_size)

                in_out[block.output] = seeds

                sparse_pc = seeds

                if block.mode == 'pointattn' and block.mode != 'SG-':
                    sparse_pc = torch.cat([sg_input, seeds],dim=2)
                    sparse_pc = gather_points(sparse_pc, furthest_point_sample(sparse_pc.transpose(1, 2).contiguous(), 512))

                    in_out[block.output] = sparse_pc
                    in_out['intermediate_seeds'] = seeds

            elif block.type == 'PG':      ########################################### Point Generator #################
                pg_seeds = in_out[block.seeds]  # actually uses the sparse_pc, but the paper calls it seeds
                pg_shape_code = in_out[block.shape_code]

                dense1, dense = block(None, pg_seeds, pg_shape_code)

                in_out[block.output] = dense1

            elif block.type == 'SAM':      ############################################################################
                sam_seeds = in_out[block.seeds]
                sam_features = in_out[block.input]

                shape_code = block(sam_features, sam_seeds)

                in_out[block.output] = shape_code

            elif block.type == 'PE':     ############################## Pose Extractor ################################
                pe_input = in_out[block.input]

                pose_code = block(pe_input)

                in_out[block.output] = pose_code

            elif block.type == 'PoseG':    ############################## Pose Generator ##############################
                pose_code = in_out[block.pose_code]

                pose_6d = block(pose_code, batch_size)

                pred_RT_mat = pose_6d
                scale = scale_paras[:, 2]   # scale
                pred_RTS_mat = torch.zeros((batch_size, 4, 4), dtype=torch.float32).to(pose_6d.device)
                pred_RTS_mat[:, :3, :3] = pred_RT_mat[:, :3, :3].type(torch.float32)
                pred_RTS_mat[:, :3, 3] = pred_RT_mat[:, :3, 3].type(torch.float32) / scale.unsqueeze(1)
                pred_RTS_mat[:, 3] = scale_paras    # scale

                #######################################################################################################
                #           |----------------|                   |----------------------------|
                #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
                # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
                #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
                #           |----------------|                   | p_min, range_min, scale, 1 |
                #                                                |----------------------------|     
                #######################################################################################################

                in_out[block.output] = pred_RTS_mat

            else:
                raise ValueError(f"Unknown block type {block.type}. Available types are: \
                                 {self.arch_builder.available_types}")
            
        seeds = seeds.transpose(1, 2).contiguous()          # [B, 256, 3]
        sparse_pc = sparse_pc.transpose(1, 2).contiguous()  # [B, 512, 3] or [B, 256, 3]
        dense = dense.transpose(1, 2).contiguous()          # [B, 2048, 3]

        if self.N_POINTS == 16384 or dense1.shape[2] == 16384:
            dense1 = gather_points(dense1, furthest_point_sample(dense1.transpose(1, 2).contiguous(), 8192))
        dense1 = dense1.transpose(1, 2).contiguous()        # [B, 8192, 3]

        

        assert seeds.shape[1] == 256 and (sparse_pc.shape[1] == 512 or sparse_pc.shape[1] == 256) and \
            (dense.shape[1] == 2048 or dense.shape[1] == 1024) and \
                dense1.shape[1] == 8192, f"seeds: {seeds.shape}, sparse_pc: {sparse_pc.shape}, dense: {dense.shape}, \
                    dense1: {dense1.shape}, N_POINTS: {self.N_POINTS}"

        in_out['dense1'] = dense1
        in_out['dense'] = dense
        in_out['sparse_pc'] = sparse_pc
        in_out['seeds'] = seeds
            
        self.in_out = in_out
        return in_out

    @staticmethod
    def get_loss(ret: list[Tensor], gt: list[Tensor], pc_subdivision: dict):
        """
        Get the loss for the model.
        :param ret: list containing the outputs of the model 
            [dense1, dense, sparse_pc, seeds, pred_RTS_mat_v, pred_RTS_mat_d]
        :param gt: list containing the ground truth [gt, gt_RTS_mat_v, gt_RTS_mat_d]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :return: loss dictionary with keys 'Loss/Total', 'Loss/Rotation', 'Loss/Translation', 'Loss/Sparse', 
            'Loss/Mid', 'Loss/Dense'
        """
        dense1, dense, sparse_pc, seeds, pred_RTS_mat_v, pred_RTS_mat_d = ret[:6]
        gt, gt_RTS_mat_v, gt_RTS_mat_d = gt[:3]


        pose_loss_dict_v = PoseEstPointAttN.get_loss(pred_RTS_mat_v, gt_RTS_mat_v)
        pose_loss_dict_d = PoseEstPointAttN.get_loss(pred_RTS_mat_d, gt_RTS_mat_d)
        pose_loss_dict_v = {k.replace('/', '/Volar/'): v for k, v in pose_loss_dict_v.items()}
        pose_loss_dict_d = {k.replace('/', '/Dorsal/'): v for k, v in pose_loss_dict_d.items()}


        completion_loss_dict = ScaphoidPointAttN.get_loss([dense1, dense, sparse_pc, seeds], [gt], pc_subdivision)  # 
        # {'Loss/Sparse': seeds_loss, 'Loss/Mid': mid_loss, 'Loss/Dense': dense_loss, 'Loss/Total': total_train_loss}

        loss_dict = {**pose_loss_dict_v, **pose_loss_dict_d, **completion_loss_dict }
        loss_dict['Loss/Total'] = (pose_loss_dict_v['Loss/Volar/Total'] + pose_loss_dict_d['Loss/Dorsal/Total']) / 10 + completion_loss_dict['Loss/Total']

        return loss_dict
    
    @staticmethod
    def get_metrics(ret: list[Tensor], gt: list[Tensor], pc_subdivision: dict, 
                    scaled_real_world_thresh: list[tuple[str, float]]=None) -> tuple[dict, None]:
        """
        Uses the get_metrics method from ScaphoidPointAttN to calculate the metrics for the dense1 and seeds.
        Additionally, it uses the get_metrics method from PoseEstPointAttN to calculate the metrics for the Pose Estimation.
        
        :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3] and predicted RTS 
            matrices with shape [B, 4, 4], as well as the transformed (Demean,Rotation,Scale) 
            point clouds (volar and dorsal) with shape [B, N_i, 3]
        :param gt: ground truth point cloud with shape [B, N, 3] and ground truth RTS matrices with shape [B, 4, 4] as 
            well as untransformed versions of the partial point clouds (volar and dorsal) with shape [B, N_i, 3]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :param scaled_real_world_thresh: optional thresholds for the F1 score that can be dynamically adjusted based on 
            the scale of the input point cloud
        :return: detailed metrics for the completed point cloud as well as the pose estimation metrics
                like {'Full/F1': 0.5, 'Full/CDL1': 0.1, ... 'Volar/F1': 0.6, 'Volar/CDL1': 0.2, ...} + 
                {Volar/Rotation/Mean: ..., Volar/Translation/Mean_X: ..., Dorsal/Rotation/Mean: ..., ...}
        """
        dense1, _, _, seeds, pred_RTS_mat_v, pred_RTS_mat_d, transformed_pcd_v, transformed_pcd_d = ret
        gt_completion, gt_RTS_mat_v, gt_RTS_mat_d, untransformed_pcd_v, untransformed_pcd_d = gt

        ren_det_metrics_dense, _ = ScaphoidPointAttN.get_metrics([dense1, _, _, seeds], [gt_completion], pc_subdivision, 
                                                                 scaled_real_world_thresh=scaled_real_world_thresh)
        pose_metrics_v = PoseEstPointAttN.get_metrics(pred_RTS_mat_v, gt_RTS_mat_v, transformed_pcd_v, 
                                                      untransformed_pcd_v)
        pose_metrics_d = PoseEstPointAttN.get_metrics(pred_RTS_mat_d, gt_RTS_mat_d, transformed_pcd_d, 
                                                      untransformed_pcd_d)

        pose_metrics_v = {f"Volar/{k}": v for k, v in pose_metrics_v.items()}
        pose_metrics_d = {f"Dorsal/{k}": v for k, v in pose_metrics_d.items()}

        metrics = {
            **ren_det_metrics_dense,
            **pose_metrics_v,
            **pose_metrics_d
        }
        return metrics, _
    



class BothPoseEstPointAttN(nn.Module):
    def __init__(self, config, pretrained: bool = False):
        super(BothPoseEstPointAttN, self).__init__()

        self.N_POINTS = config.model.num_points   # should be 8192

        self.arch_builder = ArchConfigBuilder(config, pretrained)

        self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
        self.in_out = None

    def forward(self, xyz_volar: Tensor, xyz_dorsal: Tensor, scale_paras: Tensor):
        """
        Forward pass for the completion model.
        :param xyz_volar: Volar point cloud tensor of shape [B, 3, N]
        :param xyz_dorsal: Dorsal point cloud tensor of shape [B, 3, N]
        :param scale_paras: Scale parameters tensor of shape [B, 3]
        :return: in_out dictionary containing intermediate outputs and final outputs
        """
        batch_size, _, N = xyz_volar.size()

        in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

        if store_attn_weights:              # store attn weights for visualization
            collector.add_in_out(in_out)


        for block in self.net_blocks:


            if block.type == 'OP':      ###############################################################################
                op_input = [in_out[inp] for inp in block.input]

                op_output = block(op_input)

                in_out[block.output] = op_output


            elif block.type == 'PoseEst':      ########################################## Feature Extractor ###########
                volar, dorsal = in_out['volar'], in_out['dorsal']

                pred_RT_mat_volar, pred_RT_mat_dorsal, volar_pose_code, dorsal_pose_code = block.forward(volar, dorsal)

                scale = scale_paras[:, 2]   # scale
                pred_RTS_mat_volar = torch.zeros((batch_size, 4, 4), dtype=torch.float32).to(pred_RT_mat_volar.device)
                pred_RTS_mat_volar[:, :3, :3] = pred_RT_mat_volar[:, :3, :3].type(torch.float32)
                pred_RTS_mat_volar[:, :3, 3] = pred_RT_mat_volar[:, :3, 3].type(torch.float32) / scale.unsqueeze(1)
                pred_RTS_mat_volar[:, 3] = scale_paras    # scale

                pred_RTS_mat_dorsal = torch.zeros((batch_size, 4, 4), dtype=torch.float32).to(pred_RT_mat_dorsal.device)
                pred_RTS_mat_dorsal[:, :3, :3] = pred_RT_mat_dorsal[:, :3, :3].type(torch.float32)
                pred_RTS_mat_dorsal[:, :3, 3] = pred_RT_mat_dorsal[:, :3, 3].type(torch.float32) / scale.unsqueeze(1)
                pred_RTS_mat_dorsal[:, 3] = scale_paras    # scale

                # in_out[block.output] = fe_output

            # elif block.type == 'PE':     ############################## Pose Extractor ##############################
            #     pe_input = in_out[block.input]

            #     pose_code = block(pe_input)

            #     in_out[block.output] = pose_code

            # elif block.type == 'PoseG':    ############################## Pose Generator ############################
            #     pose_code = in_out[block.pose_code]

            #     pose_6d = block(pose_code, batch_size)

            #     pred_RT_mat = pose_6d
            #     scale = scale_paras[:, 2]   # scale
            #     pred_RTS_mat = torch.zeros((batch_size, 4, 4), dtype=torch.float32).to(pose_6d.device)
            #     pred_RTS_mat[:, :3, :3] = pred_RT_mat[:, :3, :3].type(torch.float32)
            #     pred_RTS_mat[:, :3, 3] = pred_RT_mat[:, :3, 3].type(torch.float32) / scale.unsqueeze(1)
            #     pred_RTS_mat[:, 3] = scale_paras    # scale

            #     #####################################################################################################
            #     #           |----------------|                   |----------------------------|
            #     #           | R1, R2, R3, T1 |                   | R1,    R2,        R3,    T1|
            #     # RT_mat =  | R4, R5, R6, T2 |       RTS_mat =   | R4,    R5,        R6,    T2|
            #     #           | R7, R8, R9, T3 |                   | R7,    R8,        R9,    T3|
            #     #           |----------------|                   | p_min, range_min, scale, 1 |
            #     #                                                |----------------------------|     
            #     #####################################################################################################

            #     in_out[block.output] = pred_RTS_mat

            else:
                raise ValueError(f"Unknown block type {block.type}. Available types are: \
                                 {self.arch_builder.available_types}")
            
        self.in_out = in_out
        return pred_RTS_mat_volar, pred_RTS_mat_dorsal, volar_pose_code, dorsal_pose_code

    @staticmethod
    def get_loss(ret: list[Tensor], gt: list[Tensor]):
        """
        Get the loss for the model.
        :param ret: list containing the outputs of the model [pred_RTS_mat_v, pred_RTS_mat_d]
        :param gt: list containing the ground truth [gt, gt_RTS_mat_v, gt_RTS_mat_d]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :return: loss dictionary with keys 'Loss/Total', 'Loss/Rotation', 'Loss/Translation', 'Loss/Sparse', 
            'Loss/Mid', 'Loss/Dense'
        """
        pred_RTS_mat_v, pred_RTS_mat_d = ret[:2]
        gt_RTS_mat_v, gt_RTS_mat_d = gt[:2]

        #
        pose_loss_dict_v = PoseEstPointAttN.get_loss(pred_RTS_mat_v, gt_RTS_mat_v)
        pose_loss_dict_v = {k.replace('/', '/Volar/'): v for k, v in pose_loss_dict_v.items()}
        pose_loss_dict_d = PoseEstPointAttN.get_loss(pred_RTS_mat_d, gt_RTS_mat_d)
        pose_loss_dict_d = {k.replace('/', '/Dorsal/'): v for k, v in pose_loss_dict_d.items()}

        #
        # {'Loss/Sparse': seeds_loss, 'Loss/Mid': mid_loss, 'Loss/Dense': dense_loss, 'Loss/Total': total_train_loss}

        loss_dict = {**pose_loss_dict_v, **pose_loss_dict_d,}
        loss_dict['Loss/Total'] = (pose_loss_dict_v['Loss/Volar/Total'] + pose_loss_dict_d['Loss/Dorsal/Total'])

        return loss_dict
    #
    @staticmethod
    def get_metrics(ret: list[Tensor], gt: list[Tensor]) -> tuple[dict, None]:
        """
        Uses the get_metrics method from ScaphoidPointAttN to calculate the metrics for the dense1 and seeds.
        Additionally, it uses the get_metrics method from PoseEstPointAttN to calculate the metrics for the Pose Estimation.
        :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3] and predicted RTS 
            matrices with shape [B, 4, 4], as well as the transformed (Demean,Rotation,Scale) point clouds (volar and 
            dorsal) with shape [B, N_i, 3]
        :param gt: ground truth point cloud with shape [B, N, 3] and ground truth RTS matrices with shape [B, 4, 4] as 
            well as untransformed versions of the partial point clouds (volar and dorsal) with shape [B, N_i, 3]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :param scaled_real_world_thresh: optional thresholds for the F1 score that can be dynamically adjusted based on 
            the scale of the input point cloud
        :return: detailed metrics for the completed point cloud as well as the pose estimation metrics
                like {'Full/F1': 0.5, 'Full/CDL1': 0.1, ... 'Volar/F1': 0.6, 'Volar/CDL1': 0.2, ...} + 
                {Volar/Rotation/Mean: ..., Volar/Translation/Mean_X: ..., Dorsal/Rotation/Mean: ..., ...}
        """
        pred_RTS_mat_v, pred_RTS_mat_d, transformed_pcd_v, transformed_pcd_d = ret
        gt_RTS_mat_v, gt_RTS_mat_d, untransformed_pcd_v, untransformed_pcd_d = gt

        pose_metrics_v = PoseEstPointAttN.get_metrics(pred_RTS_mat_v, gt_RTS_mat_v, transformed_pcd_v, 
                                                      untransformed_pcd_v, calc_f1=True)
        pose_metrics_d = PoseEstPointAttN.get_metrics(pred_RTS_mat_d, gt_RTS_mat_d, transformed_pcd_d, 
                                                      untransformed_pcd_d, calc_f1=True)

        pose_metrics_v = {f"Volar/{k}": v for k, v in pose_metrics_v.items()}
        pose_metrics_d = {f"Dorsal/{k}": v for k, v in pose_metrics_d.items()}

        metrics = {
            **pose_metrics_v,
            **pose_metrics_d
        }
        return metrics
    