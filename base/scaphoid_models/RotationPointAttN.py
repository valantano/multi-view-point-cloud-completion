# from __future__ import print_function
# import torch
# import torch.nn as nn

# from submodules.PointAttn.utils.mm3d_pn2 import furthest_point_sample, gather_points


# from base.scaphoid_models.ArchBuilder import ArchConfigBuilder
# from base.scaphoid_models import ScaphoidModules
# from base.scaphoid_metrics.weighted_dist_chamfer_3D import calc_cd, calc_weighted_cd
# from base.scaphoid_utils.AttnWeightsCollector import AttnWeightsCollector
# from base.scaphoid_utils.logger import print_log
# import base.scaphoid_models.ScaphoidModules as modules

# from base.scaphoid_utils.transformations import get_affine_matrix, get_reverse_affine_matrix, apply_affine_transformation



# store_attn_weights = False
# collector = AttnWeightsCollector()

# ScaphoidModules.store_attn_weights = store_attn_weights
# ScaphoidModules.collector = collector

# def set_store_attn_weights(value: bool):
#     """
#     Set the store_attn_weights variable to True or False.
#     :param value: True or False
#     """
#     global store_attn_weights
#     store_attn_weights = value
#     ScaphoidModules.store_attn_weights = value


# # class RotCompPointAttN(nn.Module):
# #     def __init__(self, config):
# #         super(RotCompPointAttN, self).__init__()
# #         print(config.dataset)
# #         if config.dataset == 'pcn':
# #             step1 = 4
# #             step2 = 8
# #         elif config.dataset == 'c3d':
# #             step1 = 1
# #             step2 = 4
# #         elif config.dataset.train._base_.NAME == 'ScaphoidDataset':
# #             step1 = 4
# #             step2 = 8
# #         else:
# #             raise ValueError('dataset is not exist')

# #         self.N_POINTS = config.model.num_points   # should be 8192
        
# #         self.arch_builder = ArchConfigBuilder(config)

# #         self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
# #         self.rot_fe_v = modules.ScaphoidFeatureExtractor({'input': 'volar', 'output': 'rot_code_v', 'type': 'FE', 'mode': 'pointattn'})
# #         self.rot_fe_d = modules.ScaphoidFeatureExtractor({'input': 'dorsal', 'output': 'rot_code_d', 'type': 'FE', 'mode': 'pointattn'})
# #         self.rot_gen = modules.ScaphoidRotationGenerator({'shape_code': 'rot_code_v', 'output': 'affine_mat_v', 'type': 'RG', 'mode': 'affine'})
# #         self.rot_gen2 = modules.ScaphoidRotationGenerator({'shape_code': 'rot_code_d', 'output': 'affine_mat_d', 'type': 'RG', 'mode': 'affine'})

# #         self.shape_fe_v = modules.ScaphoidFeatureExtractor({'input': 'volar', 'output': 'shape_code_v', 'type': 'FE', 'mode': 'pointattn'})
# #         self.shape_fe_d = modules.ScaphoidFeatureExtractor({'input': 'dorsal', 'output': 'shape_code_d', 'type': 'FE', 'mode': 'pointattn'})

# #         # ARN
# #         self.arn = modules.ScaphoidAffineRefinementNetwork({'input': 'affine_mat_v', 'output': 'affine_mat_v', 'type': 'ARN', 'mode': 'mult'})
# #         self.arn = modules.ScaphoidAffineRefinementNetwork({'input': 'affine_mat_v', 'output': 'affine_mat_v', 'type': 'ARN', 'mode': 'gate'})

# #         # mult

# #         # Fusion Network

# #         self.seed_gen = modules.ScaphoidSeedGenerator({'shape_code': 'shape_code_c', 'output': 'seeds', 'type': 'SG', 'mode': 'SG-'})
# #         self.point_get = modules.ScaphoidPointGenerator({'seeds': 'seeds', 'shape_code': 'shape_code_c', 'output': 'points', 'type': 'PG', 'mode': 'pointattn'}, [step1, step2])

# #         self.in_out = None

# #     def forward(self, xyz_volar, xyz_dorsal):
# #         batch_size, _, N = xyz_volar.size()

# #         in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

# #         if store_attn_weights:              # store attn weights for visualization
# #             collector.add_in_out(in_out)

# #         rot_code_v = self.rot_fe_v(in_out['volar'])
# #         rot_code_d = self.rot_fe_d(in_out['dorsal'])
# #         in_out['rot_code_v'], in_out['rot_code_d'] = rot_code_v, rot_code_d

# #         affine_mat_v = self.rot_gen(rot_code_v)
# #         affine_mat_d = self.rot_gen2(rot_code_d)
# #         in_out['affine_mat_v'], in_out['affine_mat_d'] = affine_mat_v, affine_mat_d

# #         shape_code_v = self.shape_fe_v(in_out['volar'])
# #         shape_code_d = self.shape_fe_d(in_out['dorsal'])
# #         in_out['shape_code_v'], in_out['shape_code_d'] = shape_code_v, shape_code_d





        

        
# class RotationPointAttN(nn.Module):
#     def __init__(self, config):
#         super(RotationPointAttN, self).__init__()
#         print(config.dataset)

#         self.N_POINTS = config.model.num_points   # should be 8192
        
#         self.arch_builder = ArchConfigBuilder(config)

#         self.net_blocks = nn.ModuleList(self.arch_builder.get_blocks())
#         self.in_out = None

#     def forward(self, xyz_volar, xyz_dorsal):
#         #feat_g=shape_code, coarse=fine
#         batch_size, _, N = xyz_volar.size()

#         in_out = {'volar': xyz_volar, 'dorsal': xyz_dorsal}

#         if store_attn_weights:              # store attn weights for visualization
#             collector.add_in_out(in_out)


#         for block in self.net_blocks:

#             if block.type == 'FE':      ################################################################################################################
#                 fe_input = in_out[block.input]
#                 in_out['input_logging'] = fe_input

#                 fe_output = block.forward(fe_input)

#                 in_out[block.output] = fe_output

#             elif block.type == 'OP':      ##################################################################################################################
#                 op_input = [in_out[inp] for inp in block.input]

#                 op_output = block(op_input)

#                 in_out[block.output] = op_output

#             elif block.type == 'RE':      ############################# Rotation Extractor ############################################################
#                 re_input = in_out[block.input]

#                 shape_code, affine_matrix = block(re_input)


#                 in_out['affine_matrix'] = affine_matrix
#                 in_out['shape_code'] = shape_code

#             elif block.type == 'RG':      ############################## Rotation Generator ###########################################################
#                 rg_shape_code = in_out[block.shape_code]

#                 affine_matrix = block(rg_shape_code, batch_size)

#                 in_out['shape_code'] = rg_shape_code    # TODO: remove and remove from runner

#                 in_out['affine_matrix'] = affine_matrix

#             else:
#                 raise ValueError(f"Unknown block type {block.type}. Available types are: {self.arch_builder.available_types}")
            
#         # seeds = seeds.transpose(1, 2).contiguous()          # [B, 256, 3]
#         # sparse_pc = sparse_pc.transpose(1, 2).contiguous()  # [B, 512, 3] or [B, 256, 3]
#         # dense = dense.transpose(1, 2).contiguous()          # [B, 2048, 3]
#         # dense1 = dense1.transpose(1, 2).contiguous()        # [B, 8192, 3]


#         # assert seeds.shape[1] == 256 and (sparse_pc.shape[1] == 512 or sparse_pc.shape[1] == 256) and (dense.shape[1] == 2048 or dense.shape[1] == 1024) and dense1.shape[1] == self.N_POINTS, f"seeds: {seeds.shape}, sparse_pc: {sparse_pc.shape}, dense: {dense.shape}, dense1: {dense1.shape}, N_POINTS: {self.N_POINTS}"
#         return in_out

#     @staticmethod
#     def get_loss(ret_affine_matrix, gt_affine_matrix):
#         import torch.nn.functional as F
#         """
#         Get the loss for the model.
#         :param ret_affine_matrix: affine matrix of the model
#         :param gt_affine_matrix: affine matrix of the ground truth
#         :return: loss
#         """
#         # TODO: regularization
#         loss = F.mse_loss(ret_affine_matrix, gt_affine_matrix)
#         loss_dict = {'Loss/MSE': loss}
#         return loss_dict

#     @staticmethod
#     def get_metrics(ret_affine_mat, gt_affine_mat, partial_pcd, gt_pcd, calc_f1=False):

#         # reverse_affine_matrix2 = get_reverse_affine_matrix(ret_affine_mat)
#         rotated_partial_pcd = apply_affine_transformation(partial_pcd.clone(), ret_affine_mat)
#         rotated_partial_pcd_sure = apply_affine_transformation(partial_pcd.clone(), gt_affine_mat)


#         metrics = calc_weighted_cd(rotated_partial_pcd, gt_pcd, {}, calc_detailed_metrics=False, calc_f1=calc_f1)
#         sure_metrics = calc_weighted_cd(rotated_partial_pcd_sure, gt_pcd, {}, calc_detailed_metrics=False, calc_f1=calc_f1)

#         return metrics, sure_metrics