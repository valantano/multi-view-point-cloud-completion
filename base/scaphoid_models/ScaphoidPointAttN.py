# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from submodules.PointAttn.utils.mm3d_pn2 import furthest_point_sample, gather_points

from base.scaphoid_models.ArchBuilder import ArchConfigBuilder
import base.scaphoid_models.ScaphoidModules as modules
from base.scaphoid_metrics.weighted_dist_chamfer_3D import calc_weighted_cd
from base.scaphoid_utils.AttnWeightsCollector import AttnWeightsCollector
from base.scaphoid_utils.logger import print_log



store_attn_weights = False  # set to True if you want to store attention weights
collector = AttnWeightsCollector()

modules.store_attn_weights = store_attn_weights
modules.collector = collector

def set_store_attn_weights(value: bool):
    """
    Set the store_attn_weights variable to True or False.
    :param value: True or False
    """
    global store_attn_weights
    store_attn_weights = value
    modules.store_attn_weights = value


class ScaphoidPointAttN(nn.Module):
    def __init__(self, config, pretrained=False):
        super(ScaphoidPointAttN, self).__init__()
        print(config.dataset)
        
        self.N_POINTS = config.model.num_points   # should be 8192
        self.N_POINTS = 8192

        dropout = getattr(config.model, 'dropout', 0.0)

        # try:
        #     dropout = config.model.dropout
        # except:
        #     dropout = 0.0       # If dropout not specified in 
        modules.set_dropout(dropout)
        
        self.arch_builder = ArchConfigBuilder(config, pretrained)   # Uses the config to build the architecture of the model

        self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
        self.in_out = None  # used to make intermediate outputs available for visualization or debugging outside the class

    def forward(self, xyz_volar: torch.Tensor, xyz_dorsal: torch.Tensor):
        """
        Forward pass for the ScaphoidPointAttN model.
        :param xyz_volar: Tensor of shape [B, 3, N] representing the volar point cloud
        :param xyz_dorsal: Tensor of shape [B, 3, N] representing the dorsal point cloud
        """
        #feat_g=shape_code, coarse=fine
        batch_size, _, N = xyz_volar.size()

        in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

        if store_attn_weights:              # store attn weights for visualization
            collector.add_in_out(in_out)


        for block in self.net_blocks:

            if block.type == 'OP':      ###############################################################################
                op_input = [in_out[inp] for inp in block.input]

                op_output = block(op_input)

                in_out[block.output] = op_output

            elif block.type == 'TNet':      ###########################################################################
                t_input = in_out[block.input]

                t_output = block(t_input)

                in_out[block.output] = t_output
                
            elif block.type == 'FE':      #############################################################################
                fe_input = in_out[block.input]
                in_out['input_logging'] = fe_input

                fe_output = block.forward(fe_input)

                if block.mode == 'TNet':      #########################################################################
                    shape_code, matrix = fe_output
                    fe_output = shape_code

                    transformed_point_cloud = torch.bmm(fe_input.transpose(1, 2), matrix)
                    transformed_point_cloud = transformed_point_cloud.transpose(1, 2)  # shape [B, 3, N]

                    in_out[block.input] = transformed_point_cloud.contiguous()

                in_out[block.output] = fe_output


                # # shape_codes [B,512,1]
                # code1 = fe_output[6]
                # code2 = fe_output[7]
                # alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                # codes = [alpha * code1 + (1 - alpha) * code2 for alpha in alphas]
                # codes[0] = torch.zeros_like(codes[0])  
                # intermediate_codes = torch.stack(codes)  # [11, B, 512, 1]
                # print_log(f"ScaphoidFeatureExtractor: shape codes {code1.shape}, {code2.shape}, intermediate codes 
                #           {intermediate_codes.shape}", color='red')

                # in_out[block.output] = intermediate_codes

            elif block.type == 'SG':      #############################################################################
                if block.mode == 'pointattn' and block.mode != 'SG-':
                    sg_input = in_out[block.input]

                sg_shape_code = in_out[block.shape_code]
                batch_size, _, _ = sg_shape_code.size()
                seeds = block(sg_shape_code, batch_size)

                in_out[block.output] = seeds

                sparse_pc = seeds

                if block.mode == 'pointattn' and block.mode != 'SG-':
                    # continue
                    # if batch_size == 11:
                    #     continue
                    # print_log(f"{'xyz_affil' in in_out}SG block input shape: {sg_input.shape}, \
                    #           seeds shape: {seeds.shape}", color='red')
                    # if block.input == 'xyz_affil':
                    #     affil_2 = torch.zeros_like(seeds[:, :1, :])
                    #     seeds = torch.cat([seeds, affil_2], dim=1)

                    sparse_pc = torch.cat([sg_input, seeds],dim=2)
                    sparse_pc = gather_points(sparse_pc, furthest_point_sample(sparse_pc.transpose(1, 2).contiguous(), 512))

                    in_out[block.output] = sparse_pc
                    in_out['intermediate_seeds'] = seeds

                # if True:
                #     sg_input = in_out['concat']
                #     sparse_pc = gather_points(sg_input, furthest_point_sample(sg_input.transpose(1, 2).contiguous(), 256))
                #     sparse_pc = torch.cat([sparse_pc, seeds],dim=2)

                #     in_out[block.output] = sparse_pc

            elif block.type == 'PG':      #############################################################################
                pg_seeds = in_out[block.seeds]  # actually uses the sparse_pc, but the paper calls it seeds
                pg_shape_code = in_out[block.shape_code]

                dense1, dense = block(None, pg_seeds, pg_shape_code)
                
                in_out["pre_" + block.output] = dense
                in_out[block.output] = dense1

            elif block.type == 'SAM':      ############################################################################
                sam_seeds = in_out[block.seeds]
                sam_features = in_out[block.input]

                shape_code = block(sam_features, sam_seeds)

                in_out[block.output] = shape_code

            elif block.type == 'SCF':      ############################################################################
                shape_code_1 = in_out[block.input[0]]
                shape_code_2 = in_out[block.input[1]]

                shape_code_fused = block(shape_code_1, shape_code_2)

                in_out[block.output] = shape_code_fused

            elif block.type == 'ASSIGN':      #########################################################################
                seeds = in_out[block.seeds]
                sparse_pc = in_out[block.sparse_pc]
                dense = in_out[block.pre_points]
                dense1 = in_out[block.points]

                # print_log(f"seeds: {seeds.shape}, sparse_pc: {sparse_pc.shape}, dense: {dense.shape}, \
                #           dense1: {dense1.shape}", color='red')


            else:
                raise ValueError(f"Unknown block type {block.type}. Available types are: {self.arch_builder.available_types}")
            


        self.N_POINTS = 8192
        seeds = gather_points(seeds, furthest_point_sample(seeds.transpose(1, 2).contiguous(), 256))  # [B, 256, 3]
        sparse_pc = gather_points(sparse_pc, furthest_point_sample(sparse_pc.transpose(1, 2).contiguous(), 512))  # [B, 512, 3] or [B, 256, 3]
        dense = gather_points(dense, furthest_point_sample(dense.transpose(1, 2).contiguous(), 2048))  # [B, 2048, 3]
        dense1 = gather_points(dense1, furthest_point_sample(dense1.transpose(1, 2).contiguous(), self.N_POINTS))  # [B, 8192, 3]

        seeds = seeds.transpose(1, 2).contiguous()          # [B, 256, 3]
        sparse_pc = sparse_pc.transpose(1, 2).contiguous()  # [B, 512, 3] or [B, 256, 3]
        dense = dense.transpose(1, 2).contiguous()          # [B, 2048, 3]
        dense1 = dense1.transpose(1, 2).contiguous()        # [B, 8192, 3]

       
        assert seeds.shape[1] == 256 and (sparse_pc.shape[1] == 512 or sparse_pc.shape[1] == 256) and \
            (dense.shape[1] == 2048 or dense.shape[1] == 1024) and dense1.shape[1] == self.N_POINTS, \
                f"seeds: {seeds.shape}, sparse_pc: {sparse_pc.shape}, dense: {dense.shape}, dense1: {dense1.shape}, \
                    N_POINTS: {self.N_POINTS}"
        return dense1, dense, sparse_pc, seeds
    
    @staticmethod
    def adjust_subdivision(pc_subdivision: dict, ids_to_keep):
        """
        After downsampling the gt point cloud, the indices of the labels in pc_subdivision need to be adjusted to the 
        new indices of the point cloud.
        :param pc_subdivision: dictionary with the labels and their indices
        :param ids_to_keep: indices of the points to keep
        :return: adjusted pc_subdivision
        """
        for key in pc_subdivision.keys():
            matches = (ids_to_keep.unsqueeze(-2) == pc_subdivision[key]['ind'].unsqueeze(-1))
            mapped_indices = matches.float().argmax(dim=-1)
            pc_subdivision[key]['ind'] = mapped_indices

        return pc_subdivision
    
    @staticmethod
    def get_loss(ret: list[torch.Tensor], gt_list: list[torch.Tensor], pc_subdivision: dict):
        """
        :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3]
        :param gt_list: list with one element, the ground truth point cloud with shape [B, N, 3]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :return: loss dictionary with keys 'Loss/Sparse', 'Loss/Mid', 'Loss/Dense', 'Loss/Total'
        """
        dense1, dense, sparse_pc, seeds = ret
        gt = gt_list[0]

        assert gt.shape[1] == 8192, f"Expected gt shape [B, 8192, 3], but got {gt.shape}"
        assert dense1.shape[1] == 8192, f"Expected dense1 shape [B, 8192, 3], but got {dense1.shape}"
        assert dense.shape[1] == 2048 or dense.shape[1] == 1024, f"Expected dense shape [B, 2048, 3], but got \
            {dense.shape}"
        assert sparse_pc.shape[1] == 512 or sparse_pc.shape[1] == 256, f"Expected sparse_pc shape [B, 512, 3] or \
            [B, 256, 3], but got {sparse_pc.shape}"
        assert seeds.shape[1] == 256, f"Expected seeds shape [B, 256, 3], but got {seeds.shape}"

        dense1_ids = furthest_point_sample(dense1, gt.shape[1]) # (B, N, 3)
        dense1 = gather_points(dense1.transpose(1, 2).contiguous(), dense1_ids).transpose(1, 2).contiguous() # (B, C, N)

        if sparse_pc.shape[1] == 256:
            pass
            # print_log(f"Warning: sparse_pc is seeds: {sparse_pc.shape}", color='red')

        metrics = calc_weighted_cd(dense1, gt, pc_subdivision, calc_detailed_metrics=False)
        dense_loss = metrics['Full']['CDL1']
        
        gt_fine_ids = furthest_point_sample(gt, dense.shape[1])
        gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), gt_fine_ids).transpose(1, 2).contiguous()

        gt_fine_subdivision = ScaphoidPointAttN.adjust_subdivision(pc_subdivision, gt_fine_ids)
        
        metrics = calc_weighted_cd(dense, gt_fine1, gt_fine_subdivision, calc_detailed_metrics=False)
        mid_loss = metrics['Full']['CDL1']

        gt_sparse_ids =furthest_point_sample(gt_fine1, seeds.shape[1])
        gt_sparse = gather_points(gt_fine1.transpose(1, 2).contiguous(), gt_sparse_ids).transpose(1, 2).contiguous()

        gt_sparse_subdivision = ScaphoidPointAttN.adjust_subdivision(gt_fine_subdivision, gt_sparse_ids)

        metrics = calc_weighted_cd(seeds, gt_sparse, gt_sparse_subdivision, calc_detailed_metrics=False)
        seeds_loss = metrics['Full']['CDL1']

        total_train_loss = seeds_loss.mean() + mid_loss.mean() + dense_loss.mean()

        loss_dict = {
            'Loss/Sparse': seeds_loss, 'Loss/Mid': mid_loss, 'Loss/Dense': dense_loss, 'Loss/Total': total_train_loss
        }

        return loss_dict

    @staticmethod
    def get_metrics(ret: list[torch.Tensor], gt_list: list[torch.Tensor], pc_subdivision: dict, scaled_real_world_thresh: list[tuple[str, float]]=None):
        """
        Get metrics for the model's predictions.
        Calculates CDL1, CDL2, F1, precision and recall by comparing the predicted point clouds (dense1, dense, seeds) 
        with the ground truth point cloud (gt).
        :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3]
        :param gt_list: list with one element, the ground truth point cloud with shape [B, N, 3]
        :param pc_subdivision: dictionary containing the labels of each point in the gt (like 
            {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
        :param scaled_real_world_thresh: optional thresholds for the F1 score that can be dynamically adjusted based on 
            the scale of the input point cloud
        :return: detailed metrics for the model's predictions like 
            {'Full/F1': 0.5, 'Full/CDL1': 0.1, ... 'Volar/F1': 0.6, 'Volar/CDL1': 0.2, ...}
        """
        dense1, _, _, seeds = ret
        gt = gt_list[0]

        assert gt.shape[1] == 8192, f"Expected gt shape [B, 8192, 3], but got {gt.shape}"
        assert dense1.shape[1] == 8192, f"Expected dense1 shape [B, 8192, 3], but got {dense1.shape}"
        assert seeds.shape[1] == 256, f"Expected seeds shape [B, 256, 3], but got {seeds.shape}"

        dense1_ids = furthest_point_sample(dense1, gt.shape[1])
        dense1 = gather_points(dense1.transpose(1, 2).contiguous(), dense1_ids).transpose(1, 2).contiguous()


        detailed_metrics_dense = calc_weighted_cd(dense1, gt, pc_subdivision, calc_detailed_metrics=True, calc_f1=True, 
                                                  scaled_real_world_thresh=scaled_real_world_thresh)

        # Rename dict with structure {prefix: {metric_name: metric_value}} to {prefix/metric_name: metric_value}
        renamed_detailed_metrics_dense = {}
        for prefix, metrics in detailed_metrics_dense.items():
            for metric_name, metric_value in metrics.items():
                renamed_detailed_metrics_dense[f'{prefix[0].upper() + prefix[1:]}/{metric_name}'] = metric_value

        return renamed_detailed_metrics_dense, _



# class ScaphoidRotationPointAttN(nn.Module):
#     """
#     Combines RotationPointAttN and ScaphoidPointAttN.
#     Architecture of RotationPointAttN is used to extract the affine matrices.
#     Architecture of ScaphoidPointAttN is used to use the aligned point clouds to complete the scaphoid point cloud.
#     """


#     def __init__(self, config, pretrained=False):
#         super(ScaphoidRotationPointAttN, self).__init__()
#         print(config.dataset)
        
#         self.N_POINTS = config.model.num_points   # should be 8192
        
#         self.arch_builder = ArchConfigBuilder(config, pretrained)

#         self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
#         self.in_out = None

#     def forward(self, xyz_volar: torch.Tensor, xyz_dorsal: torch.Tensor):
#         """
#         Forward pass for the ScaphoidRotationPointAttN model.
#         :param xyz_volar: Tensor of shape [B, 3, N] representing the volar point cloud
#         :param xyz_dorsal: Tensor of shape [B, 3, N] representing the dorsal point cloud
#         :return: dense1, dense, sparse_pc, seeds, volar_affine, dorsal_affine, align_affine
#         """
#         batch_size, _, N = xyz_volar.size()

#         in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

#         if store_attn_weights:              # store attn weights for visualization
#             collector.add_in_out(in_out)


#         for block in self.net_blocks:

#             if block.type == 'OP':      #############################################################################
#                 op_input = [in_out[inp] for inp in block.input]

#                 op_output = block(op_input)

#                 if block.mode == 'aligner':
#                     pcd_aligned, align_affine = op_output
#                     op_output = pcd_aligned

#                 in_out[block.output] = op_output

#             elif block.type == 'TNet':      #########################################################################
#                 t_input = in_out[block.input]

#                 t_output = block(t_input)

#                 in_out[block.output] = t_output
                
#             elif block.type == 'FE':      ###########################################################################
#                 fe_input = in_out[block.input]
#                 in_out['input_logging'] = fe_input

#                 fe_output = block.forward(fe_input)

#                 if block.mode == 'TNet':      #######################################################################
#                     shape_code, matrix = fe_output
#                     fe_output = shape_code

#                     transformed_point_cloud = torch.bmm(fe_input.transpose(1, 2), matrix)
#                     transformed_point_cloud = transformed_point_cloud.transpose(1, 2)  # shape [B, 3, N]

#                     in_out[block.input] = transformed_point_cloud.contiguous()

#                 in_out[block.output] = fe_output

#             elif block.type == 'SG':      ###########################################################################
#                 if block.mode == 'pointattn' and block.mode != 'SG-':
#                     sg_input = in_out[block.input]

#                 sg_shape_code = in_out[block.shape_code]

#                 seeds = block(sg_shape_code, batch_size)

#                 in_out[block.output] = seeds

#                 sparse_pc = seeds

#                 if block.mode == 'pointattn' and block.mode != 'SG-':
#                     sparse_pc = torch.cat([sg_input, seeds],dim=2)
#                     sparse_pc = gather_points(sparse_pc, furthest_point_sample(sparse_pc.transpose(1, 2).contiguous(), 512))

#                     in_out[block.output] = sparse_pc
#                     in_out['intermediate_seeds'] = seeds

#             elif block.type == 'PG':      ###########################################################################
#                 pg_seeds = in_out[block.seeds]  # actually uses the sparse_pc, but the paper calls it seeds
#                 pg_shape_code = in_out[block.shape_code]

#                 dense1, dense = block(None, pg_seeds, pg_shape_code)

#                 in_out[block.output] = dense1

#             elif block.type == 'SAM':      ##########################################################################
#                 sam_seeds = in_out[block.seeds]
#                 sam_features = in_out[block.input]

#                 shape_code = block(sam_features, sam_seeds)

#                 in_out[block.output] = shape_code

#             elif block.type == 'SCF':      ##########################################################################
#                 shape_code_1 = in_out[block.input[0]]
#                 shape_code_2 = in_out[block.input[1]]

#                 shape_code_fused = block(shape_code_1, shape_code_2)

#                 in_out[block.output] = shape_code_fused

#             elif block.type == 'RE':      ############################# Rotation Extractor ##########################
#                 re_input = in_out[block.input]

#                 shape_code, affine_matrix = block(re_input)


#                 in_out['affine_matrix'] = affine_matrix
#                 in_out['shape_code'] = shape_code

#             elif block.type == 'RG':      ############################## Rotation Generator #########################
#                 rg_shape_code = in_out[block.shape_code]

#                 affine_matrix = block(rg_shape_code)

#                 in_out['affine_matrix'] = affine_matrix

#             elif block.type == 'RERG':      ############################# Rotation Extractor and Rotation Generator #
#                 rerg_input = in_out[block.input]

#                 B, _, _ = rerg_input.size()

#                 affine_mat = block(rerg_input, B)

#                 in_out[block.output] = affine_mat

#                 if block.input == 'volar':
#                     volar_affine = affine_mat
#                 elif block.input == 'dorsal':
#                     dorsal_affine = affine_mat
                

#             else:
#                 raise ValueError(f"Unknown block type {block.type}. Available types are: {self.arch_builder.available_types}")
            
#         seeds = seeds.transpose(1, 2).contiguous()          # [B, 256, 3]
#         sparse_pc = sparse_pc.transpose(1, 2).contiguous()  # [B, 512, 3] or [B, 256, 3]
#         dense = dense.transpose(1, 2).contiguous()          # [B, 2048, 3]
#         dense1 = dense1.transpose(1, 2).contiguous()        # [B, 8192, 3]

#         self.in_out = in_out
#         assert seeds.shape[1] == 256 and (sparse_pc.shape[1] == 512 or sparse_pc.shape[1] == 256) and (dense.shape[1] == 2048 or dense.shape[1] == 1024) and dense1.shape[1] == self.N_POINTS, f"seeds: {seeds.shape}, sparse_pc: {sparse_pc.shape}, dense: {dense.shape}, dense1: {dense1.shape}, N_POINTS: {self.N_POINTS}"
#         return dense1, dense, sparse_pc, seeds, volar_affine, dorsal_affine, align_affine      # TODO: previously dense and not dense1 was returned... why? Does it matter?

#     @staticmethod
#     def get_loss(ret: list[torch.Tensor], gt: list[torch.Tensor], pc_subdivision: dict):
#         """
#         Get the loss for the model. Uses the get_loss method from ScaphoidPointAttN to calculate the loss for the dense1, dense, sparse_pc and seeds and adds the rotation loss.
#         :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3] and affine matrices of the volar and dorsal partial point clouds with shape [B, 4, 4]
#         :param gt: ground truth point cloud with shape [B, N, 3] and ground truth affine matrix with shape [B, 4, 4]
#         :param pc_subdivision: dictionary containing the labels of each point in the gt (like {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
#         :return: loss dictionary with keys 'Loss/Sparse', 'Loss/Mid', 'Loss/Dense', 'Loss/Total'
#         """
#         dense1, dense, sparse_pc, seeds, affine_mat_v, affine_mat_d, pred_affine_mat = ret
#         gt, gt_affine_mat = gt

#         loss_dict = ScaphoidPointAttN.get_loss([dense1, dense, sparse_pc, seeds], gt, pc_subdivision)

#         mse_rotation_loss = RotationPointAttN.get_loss(pred_affine_mat, gt_affine_mat)['Loss/MSE']

#         loss_dict['Loss/Total'] = loss_dict['Loss/Total'] + 10 * mse_rotation_loss.mean()
#         loss_dict['Loss/Rotation'] = mse_rotation_loss

#         return loss_dict

#     @staticmethod
#     def get_metrics(ret: list[torch.Tensor], gt: list[torch.Tensor], pc_subdivision: dict, scaled_real_world_thresh: list[tuple[str, float]]=None):
#         """
#         Uses the get_metrics method from ScaphoidPointAttN to calculate the metrics for the dense1 and seeds.
#         Does not add any other metrics, but can be extended to do so.
#         :param ret: output of the network, contains dense1, dense, seeds all with shape [B, N_i, 3] and affine matrices of the volar and dorsal partial point clouds with shape [B, 4, 4]
#         :param gt: ground truth point cloud with shape [B, N, 3] and ground truth affine matrix with shape [B, 4, 4]
#         :param pc_subdivision: dictionary containing the labels of each point in the gt (like {'volar': {'weighting': 1, 'ind': [point_ids...]}, 'dorsal': {'weighting': 1, 'ind': [point_ids...]}, ...})
#         :param scaled_real_world_thresh: optional thresholds for the F1 score that can be dynamically adjusted based on the scale of the input point cloud
#         :return: detailed metrics for the model's predictions like {'Full/F1': 0.5, 'Full/CDL1': 0.1, ... 'Volar/F1': 0.6, 'Volar/CDL1': 0.2, ...}
#         """
#         dense1, _, _, seeds, affine_mat_v, affine_mat_d, pred_affine_mat = ret
#         gt, gt_affine_mat = gt

#         renamed_detailed_metrics_dense, _ = ScaphoidPointAttN.get_metrics([dense1, _, _, seeds], gt, pc_subdivision, scaled_real_world_thresh)

#         return renamed_detailed_metrics_dense, _
    
#     @staticmethod
#     def get_rotation_metrics(ret_affine_mat: torch.Tensor, gt_affine_mat: torch.Tensor, partial_pcd: torch.Tensor, gt_pcd: torch.Tensor, calc_f1=False):
#         """
#         Get the rotation metrics for the model.
#         :param ret_affine_mat: predicted affine matrices with shape [B, 4, 4]
#         :param gt_affine_mat: ground truth affine matrices with shape [B, 4, 4]
#         :param partial_pcd: partial point cloud with shape [B, N, 3] 
#         :param gt_pcd: ground truth point cloud that has not been rotated with shape [B, N, 3]
#         """
#         metrics, sure_metrics = RotationPointAttN.get_metrics(ret_affine_mat, gt_affine_mat, partial_pcd, gt_pcd, calc_f1=calc_f1)

#         return metrics, sure_metrics
