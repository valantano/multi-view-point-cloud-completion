import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict


from submodules.PointAttn.utils.mm3d_pn2 import furthest_point_sample, gather_points


from base.scaphoid_utils.logger import print_log
import base.scaphoid_utils.transformations as t


store_attn_weights = False
collector = None

dropout = 0.0
def set_dropout(d):
    global dropout
    dropout = d


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024):
        super().__init__()
        global dropout
        if dropout != 0.0:
            print_log(f"cross_transformer dropout: {dropout}", color='red')
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        attn, attn_weights = self.multihead_attn1(query=src1, key=src2, value=src2)

        src1 = src1 + self.dropout12(attn)
        src1 = self.norm12(src1)

        attn = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(attn)


        src1 = src1.permute(1, 2, 0)

        if store_attn_weights:
            attn_weights = attn_weights.squeeze().cpu()
            collector.add_weights(attn_weights)

        return src1

class cross_seed_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024):
        super().__init__()
        global dropout
        print_log(f"cross_seed_transformer dropout: {dropout}", color='red')
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)
        self.input_proj_seeds = nn.Conv1d(3, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, features_in, seeds):
        """
        :param features [B,512, N] N=128 for single and 256 for double | used as query
        :param seeds [B,3,256] | used as key and value
        """
        features = self.input_proj(features_in)    # now should be [B,512, N]
        seeds = self.input_proj_seeds(seeds)    # now should be [B,512,256]

        b, c, _ = features.shape

        features = features.reshape(b, c, -1).permute(2, 0, 1)
        seeds = seeds.reshape(b, c, -1).permute(2, 0, 1)

        # Norm?
        features = self.norm13(features)
        seeds = self.norm13(seeds)

        # Multi-Head Self-Attention
        c_attn, attn_weights = self.multihead_attn1(value=features, key=features, query=seeds)

        # Add & Norm
        seeds_enh = seeds + self.dropout12(c_attn)
        seeds_enh = self.norm12(seeds_enh)

        # FFN
        c_attn = self.linear12(self.dropout1(self.activation1(self.linear11(seeds_enh))))
        # Add
        seeds_enh = seeds_enh + self.dropout13(c_attn)
        x1 = seeds_enh.permute(1, 2, 0)

        # Concat
        # x1 = torch.cat([seeds_enh, x1], dim=1)

        if store_attn_weights:
            attn_weights = attn_weights.squeeze().cpu()
            collector.add_weights(attn_weights)

        return x1
    
class ShapeCodeFuser(nn.Module):
    def __init__(self, seed_attn_config, input_dim=512, hidden_dim=256, output_dim=512):
        super(ShapeCodeFuser, self).__init__()
        self.input = seed_attn_config.input
        self.output = seed_attn_config.output
        self.mode = seed_attn_config.mode
        self.type = seed_attn_config.type

        if self.type not in ['SCF']:
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: {['SCF']}")
        
        if self.mode not in ['mlp']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['mlp']}")

        self.fc1 = nn.Linear(2*input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, code1, code2):
        # Concatenate the two shape codes
        x = torch.cat((code1, code2), dim=1).squeeze(-1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x.unsqueeze(-1)  # Reshape to [B, output_dim, 1]
        

class SeedAttnMatcher(nn.Module):
    def __init__(self, seed_attn_config):
        super(SeedAttnMatcher, self).__init__()
        # self.config = fe_config
        self.input = seed_attn_config.input
        self.seeds = seed_attn_config.seeds
        self.output = seed_attn_config.output
        self.mode = seed_attn_config.mode
        self.type = seed_attn_config.type
        if self.type not in ['SAM']:
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: {['SAM']}")
        
        if self.mode not in ['attn']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['attn']}")
        
        channel = 512
        self.seed_cross_attn = cross_seed_transformer(channel, channel) # [B,512,N]->[B,1024,N]
        self.sfa_1 = cross_transformer(channel,channel) # [B,1024,N]    with concat | without concat [B,512,N]


    def forward(self, features, seeds):
        batch_size, _, N = features.size()
        _, _, N_seeds = seeds.size()

        if store_attn_weights:
            collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='SeedAttnMatcher', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'output': self.output})

        x = self.seed_cross_attn(features, seeds)
        x1 = self.sfa_1(x,x).contiguous()

        ############### Attn Vis ######################################################################################
        if store_attn_weights:
            seed_org_ids = torch.arange(N_seeds).unsqueeze(0).repeat(batch_size, 1).squeeze()

            att_indices = collector.get_last_added_indices()  # workaround to get the indices of the feature_extractor_intermediate

            collector.add_indices(k_v_ids=att_indices['q_ids']['inds'], k_v_ref=att_indices['q_ids']['pcd_ref'], q_ids=seed_org_ids, q_ref=self.seeds)

            collector.deactivate()  # make sure that only feature extractor is stored and not the whole model
        ###############################################################################################################
        
        shape_code = F.adaptive_max_pool1d(x1, 1).view(batch_size, -1).unsqueeze(-1)

        return shape_code


class BothPoseEstimator(nn.Module):
    def __init__(self, config):
        super(BothPoseEstimator, self).__init__()
        self.output = None
        self.type = 'PoseEst'

        self.pose_extractor = ScaphoidPoseExtractor(EasyDict({'input': ['volar', 'dorsal'], 'output': ['pose_code_v', 'pose_code_d'], 'mode': 'pose', 'type': 'PE'}))
        self.pose_generator = ScaphoidPoseGenerator(EasyDict({'pose_code': ['pose_code_v', 'pose_code_d'], 'output': ['pred_RT_mat_volar', 'pred_RT_mat_dorsal'], 'mode': 'pose', 'type': 'PoseG'}))

    def forward(self, xyz_volar, xyz_dorsal):
        B, _, _ = xyz_volar.size()
        volar_pose_code = self.pose_extractor(xyz_volar)
        dorsal_pose_code = self.pose_extractor(xyz_dorsal)

        pred_RT_mat_volar = self.pose_generator(volar_pose_code, B)
        pred_RT_mat_dorsal = self.pose_generator(dorsal_pose_code, B)

        return pred_RT_mat_volar, pred_RT_mat_dorsal, volar_pose_code, dorsal_pose_code


##################################### Pose Extractor #############################################################################################################
class ScaphoidPoseExtractor(nn.Module):
    def __init__(self, fe_config):
        super(ScaphoidPoseExtractor, self).__init__()
        # self.config = fe_config
        self.input = fe_config.input
        self.output = fe_config.output
        self.mode = fe_config.mode
        self.type = fe_config.type
        if self.type != 'PE':
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: 'PE'")
        if self.mode not in ['pose']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['pose']}")

        channel = 64
        input_dim = 3   # xyz


        self.channel = channel
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.relu = nn.GELU()

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)

        self.sa3_1 = cross_transformer((channel)*8,channel*8)   # not needed if intermediate features are returned instead of shape code


        self.fc = nn.Linear(512, 12)   # 


    def forward(self, points):
        batch_size, _, N = points.size()

        if store_attn_weights:
            collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidPoseExtractor', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'output': self.output})

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        ############### GDP ###########################################################################################
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)  # get ids of the sampled points
        x_g0 = gather_points(x0, idx_0)     # use gpu parallelization to get the points using the sampled ids very fast
        points = gather_points(points, idx_0)   # get the points using the sampled ids for later FPS so FPS only uses xyz coordinates and not features
        x1 = self.sa1(x_g0, x0).contiguous()    # x0=v,k x_g0=Y=q(downsampled version) input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        x1 = torch.cat([x_g0, x1], dim=1)
        ################ SFA ##########################################################################################
        x1 = self.sa1_1(x1,x1).contiguous()     # input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        ############### GDP ###########################################################################################
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        ############### SFA ###########################################################################################
        x2 = self.sa2_1(x2, x2).contiguous()
        ############### GDP ###########################################################################################
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)


        ############### SFA ###########################################################################################
        x3 = self.sa3_1(x3,x3).contiguous()

            
        ############### Attn Vis ######################################################################################
        if store_attn_weights:
            idx_0 = idx_0.squeeze().cpu()
            idx_1 = idx_1.squeeze().cpu()
            idx_2 = idx_2.squeeze().cpu()
            org_ids = torch.arange(N).unsqueeze(0).repeat(batch_size, 1).squeeze()

            collector.add_indices(k_v_ids=org_ids,                      q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1[idx_2]]], q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.deactivate()  # make sure that only feature extractor is stored and not the whole model
        ###############################################################################################################

        ############### maxpooling ####################################################################################
        pose_code = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        
        # affine_params = self.fc(pose_code.squeeze(-1))
        # affine_matrix = affine_params.view(batch_size, 3, 4)   # [B,3,4]

        # affine_matrix = torch.cat((affine_matrix, torch.tensor([[0, 0, 0, 1]]).cuda().unsqueeze(0).repeat(batch_size, 1, 1)), dim=1)   # [B,4,4]

        return pose_code
    
class ScaphoidPoseGenerator(nn.Module):
    """
    Shall generate sparse but complete point cloud
    """
    def __init__(self, sg_config):
        super(ScaphoidPoseGenerator, self).__init__()
        self.input = None
        self.pose_code = sg_config.pose_code
        self.output = sg_config.output
        self.mode = sg_config.mode
        self.type = sg_config.type
        if self.type not in ['PoseG']:
            raise ValueError(f"Pose generator type {self.type} is not supported. Supported types are: {['PoseG']}")
        
        if self.mode not in ['pose']:
            raise ValueError(f"Pose generator mode {self.mode} is not supported. Supported modes are: {['pose']}")

        channel = 64
        # if self.shape_code == 'shape_code_c':
        #     channel = 128


        self.channel = channel

        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        
        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        

        self.conv_out_rot = nn.Conv1d(3, 3, kernel_size=64, stride=64)
        # TODO: replace conv_out with llinear layer to get rid 3x4 matrix and use 3x3 rotation matrix instead

    def forward(self, shape_code, batch_size):
        if store_attn_weights:
            # collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidPoseGenerator', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'shape_code': self.pose_code, 'output': self.output})


        ############### seed generator
        x = self.relu(self.ps_adj(shape_code))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        ############### SFA
        x0_d = (self.sa0_d(x, x))
        ############### SFA
        x1_d = (self.sa1_d(x0_d, x0_d))
        ############### SFA + Reshape
        # x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1)   # valentino: adaptation to different N

        ############### MLP
        seeds = self.conv_out(self.relu(self.conv_out1(x2_d)))

        pose = self.conv_out_rot(self.relu(seeds))   # [B,3,4]

        rot1, rot2 = pose[:, 0, :3], pose[:, 1, :3]   # [B,3] each
        t = pose[:, 2, :3]   # [B,3]

        b1 = F.normalize(rot1, dim=1)
        dot_prod = torch.sum(b1 * rot2, dim=1, keepdim=True)   # [B,1]
        proj = dot_prod * b1   # [B,3]
        rot2_ortho = rot2 - proj   # [B,3]
        b2 = F.normalize(rot2_ortho, dim=1)   # [B,3]

        b3 = torch.cross(b1, b2, dim=1)   # [B,3]

        R = torch.stack((b1, b2, b3), dim=1)   # [B,3,3] rotation matrix

        RT = torch.cat([R, t.unsqueeze(2)], dim=2)   # [B,3,4] rotation matrix with translation


        return RT
    

class ScaphoidRotationExtractor(nn.Module):
    def __init__(self, fe_config):
        super(ScaphoidRotationExtractor, self).__init__()
        # self.config = fe_config
        self.input = fe_config.input
        self.output = fe_config.output
        self.mode = fe_config.mode
        self.type = fe_config.type
        if self.type != 'RE':
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: 'RE'")
        if self.mode not in ['affine']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['affine']}")

        channel = 64
        input_dim = 3   # xyz


        self.channel = channel
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.relu = nn.GELU()

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)

        self.sa3_1 = cross_transformer((channel)*8,channel*8)   # not needed if intermediate features are returned instead of shape code


        self.fc = nn.Linear(512, 12)   # 


    def forward(self, points):
        batch_size, _, N = points.size()

        if store_attn_weights:
            collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidFeatureExtractor', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'output': self.output})

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        ############### GDP ###########################################################################################
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)  # get ids of the sampled points
        x_g0 = gather_points(x0, idx_0)     # use gpu parallelization to get the points using the sampled ids very fast
        points = gather_points(points, idx_0)   # get the points using the sampled ids for later FPS so FPS only uses xyz coordinates and not features
        x1 = self.sa1(x_g0, x0).contiguous()    # x0=v,k x_g0=Y=q(downsampled version) input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        x1 = torch.cat([x_g0, x1], dim=1)
        ################ SFA ##########################################################################################
        x1 = self.sa1_1(x1,x1).contiguous()     # input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        ############### GDP ###########################################################################################
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        ############### SFA ###########################################################################################
        x2 = self.sa2_1(x2, x2).contiguous()
        ############### GDP ###########################################################################################
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)


        ############### SFA ###########################################################################################
        x3 = self.sa3_1(x3,x3).contiguous()

            
        ############### Attn Vis ######################################################################################
        if store_attn_weights:
            idx_0 = idx_0.squeeze().cpu()
            idx_1 = idx_1.squeeze().cpu()
            idx_2 = idx_2.squeeze().cpu()
            org_ids = torch.arange(N).unsqueeze(0).repeat(batch_size, 1).squeeze()

            collector.add_indices(k_v_ids=org_ids,                      q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1[idx_2]]], q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.deactivate()  # make sure that only feature extractor is stored and not the whole model
        ###############################################################################################################

        ############### maxpooling ####################################################################################
        rotation_code = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)

        if self.mode == 'TNet':
            matrix = self.feature_transform_net(x3)
            return rotation_code, matrix
        
        affine_params = self.fc(rotation_code.squeeze(-1))
        affine_matrix = affine_params.view(batch_size, 3, 4)   # [B,3,4]

        affine_matrix = torch.cat((affine_matrix, torch.tensor([[0, 0, 0, 1]]).cuda().unsqueeze(0).repeat(batch_size, 1, 1)), dim=1)   # [B,4,4]

        return rotation_code, affine_matrix

class ScaphoidRotationGenerator(nn.Module):
    """
    Shall generate sparse but complete point cloud
    """
    def __init__(self, sg_config):
        super(ScaphoidRotationGenerator, self).__init__()
        self.input = None
        self.shape_code = sg_config.shape_code
        self.output = sg_config.output
        self.mode = sg_config.mode
        self.type = sg_config.type
        if self.type not in ['RG']:
            raise ValueError(f"Rotation generator type {self.type} is not supported. Supported types are: {['RG']}")
        
        if self.mode not in ['affine']:
            raise ValueError(f"Rotation generator mode {self.mode} is not supported. Supported modes are: {['affine']}")

        channel = 64
        # if self.shape_code == 'shape_code_c':
        #     channel = 128


        self.channel = channel

        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        
        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)

        self.conv_out_rot = nn.Conv1d(3, 3, kernel_size=64, stride=64)

    def forward(self, shape_code, batch_size):
        if store_attn_weights:
            # collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidPoseGenerator', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'shape_code': self.shape_code, 'output': self.output})


        ############### seed generator
        x = self.relu(self.ps_adj(shape_code))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        ############### SFA
        x0_d = (self.sa0_d(x, x))
        ############### SFA
        x1_d = (self.sa1_d(x0_d, x0_d))
        ############### SFA + Reshape
        # x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1)   # valentino: adaptation to different N

        ############### MLP
        seeds = self.conv_out(self.relu(self.conv_out1(x2_d)))

        rot = self.conv_out_rot(self.relu(seeds))

        affine_matrix = torch.cat((rot, torch.tensor([[0, 0, 0, 1]]).cuda().unsqueeze(0).repeat(batch_size, 1, 1)), dim=1)   # [B,4,4]

        return affine_matrix
#####################################################################################################################################################################
class RERGenerator(nn.Module):

    def __init__(self, fe_config):
        super(RERGenerator, self).__init__()
        self.input = fe_config.input
        self.output = fe_config.output
        self.mode = fe_config.mode
        self.type = fe_config.type
        if self.type != 'RERG':
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: 'RERG'")
        if self.mode not in ['affine']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['affine']}")
        self.RE = ScaphoidRotationExtractor(EasyDict({'output': None, 'input': None, 'mode': 'affine', 'type': 'RE'}))
        self.RG = ScaphoidRotationGenerator(EasyDict({'output': None, 'shape_code': None, 'mode': 'affine', 'type': 'RG'}))

    def forward(self, points, batch_size):
        """
        :param points: [B, 3, N]
        :return: affine_matrix: [B, 4, 4]
        """
        # Get the rotation code and affine matrix from the ScaphoidRotationExtractor
        rotation_code, affine_matrix = self.RE(points)

        # Generate the affine matrix using the ScaphoidRotationGenerator
        affine_matrix = self.RG(rotation_code, batch_size)

        return affine_matrix

class ScaphoidAffineRefinementNetwork(nn.Module):
    def __init__(self, fe_config):
        super(ScaphoidAffineRefinementNetwork, self).__init__()
        self.input = fe_config.input
        self.output = fe_config.output
        self.mode = fe_config.mode
        self.type = fe_config.type
        if self.type != 'ARN':
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: 'ARN'")
        if self.mode not in ['gate', 'mult']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['gate', 'mult']}")
        
        self.fc1 = nn.Linear(1024 + 16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)

        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.fc_gate = nn.Linear(512, 512)

    def forward(self, shape_code_v, shape_code_d, affine_matrix):
        """
        :param shape_code_v: [B, 512, 1]
        :param shape_code_d: [B, 512, 1]
        :param affine_matrix: [B, 4, 4] or [B, 16]
        """
        shape_codes = torch.cat((shape_code_v, shape_code_d), dim=1).squeeze(-1)
        affine_matrix = affine_matrix.view(affine_matrix.shape[0], 16)

        x = torch.cat((shape_codes, affine_matrix), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        affine_fe_vec = self.fc3(x)

        if self.mode == 'gate':
            transformed_shape_code = self.apply_affine_gate(affine_fe_vec, shape_code_v, shape_code_d)
        elif self.mode == 'mult':
            transformed_shape_code = self.apply_affine_mult(affine_fe_vec, shape_code_v, shape_code_d)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        return transformed_shape_code

    def apply_affine_gate(self, affine_fe_vec, shape_code_v, shape_code_d):
        """
        :param affine_fe_vec: [B, 512, 1]
        :param shape_code_v: [B, 512, 1]
        :param shape_code_d: [B, 512, 1]
        :return: transformed shape code [B, 512, 1]
        """
        # Apply the affine transformation to the shape codes
        gate = self.sigmoid(self.fc_gate(affine_fe_vec.squeeze(-1)))

        shape_code_transformed = (1 + gate) * shape_code_v + affine_fe_vec

        return shape_code_transformed

    def apply_affine_mult(self, affine_fe_vec, shape_code_v, shape_code_d):
        """
        :param affine_fe_vec: [B, 512, 1]
        :param shape_code_v: [B, 512, 1]
        :param shape_code_d: [B, 512, 1]
        :return: transformed shape code [B, 512, 1]
        """
        # Apply the affine transformation to the shape codes
        shape_code_transformed = torch.bmm(affine_fe_vec, shape_code_v)

        return shape_code_transformed


##################################### Feature Extractor #############################################################################################################
class ScaphoidFeatureExtractor(nn.Module):
    def __init__(self, fe_config):
        super(ScaphoidFeatureExtractor, self).__init__()
        # self.config = fe_config
        self.input = fe_config.input
        self.output = fe_config.output
        self.mode = fe_config.mode
        self.type = fe_config.type
        if self.type != 'FE':
            raise ValueError(f"Feature extractor type {self.type} is not supported. Supported types are: 'FE'")
        if self.mode not in ['pointattn', 'affil', 'FE-', 'TNet']:
            raise ValueError(f"Feature extractor mode {self.mode} is not supported. Supported modes are: {['pointattn', 'affil', 'FE-', 'TNet']}")

        channel = 64
        input_dim = 3   # xyz

        # if self.input == 'concat':
        #     channel = 128 only at SG
        if self.mode == 'affil':
            input_dim = 4   # xyz + affiliation

        self.channel = channel
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.relu = nn.GELU()

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)

        if self.mode != 'FE-':
            self.sa3_1 = cross_transformer((channel)*8,channel*8)   # not needed if intermediate features are returned instead of shape code

        if self.mode == 'TNet':
            self.feature_transform_net = FeatureTransformNet(input_dim=128, K=channel)

    def forward(self, points):
        batch_size, _, N = points.size()
        # print_log(f"ScaphoidFeatureExtractor: input shape {points.shape}, batch size {batch_size}, N {N}")

        if store_attn_weights:
            collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidFeatureExtractor', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'output': self.output})

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        ############### GDP ###########################################################################################
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)  # get ids of the sampled points
        x_g0 = gather_points(x0, idx_0)     # use gpu parallelization to get the points using the sampled ids very fast
        points = gather_points(points, idx_0)   # get the points using the sampled ids for later FPS so FPS only uses xyz coordinates and not features
        x1 = self.sa1(x_g0, x0).contiguous()    # x0=v,k x_g0=Y=q(downsampled version) input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        x1 = torch.cat([x_g0, x1], dim=1)
        ################ SFA ##########################################################################################
        x1 = self.sa1_1(x1,x1).contiguous()     # input -> Multi-Head Self-Attention -> Add & Norm -> FFN -> Add -> output
        ############### GDP ###########################################################################################
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        ############### SFA ###########################################################################################
        x2 = self.sa2_1(x2, x2).contiguous()
        ############### GDP ###########################################################################################
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)


        ############### Attn Vis ######################################################################################
        if store_attn_weights:
            idx_0 = idx_0.squeeze().cpu()
            idx_1 = idx_1.squeeze().cpu()
            idx_2 = idx_2.squeeze().cpu()
            org_ids = torch.arange(N).unsqueeze(0).repeat(batch_size, 1).squeeze()

            collector.add_indices(k_v_ids=org_ids,                      q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0],               k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0],               q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))

            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1]],        k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1]],        q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))

            
            collector.deactivate()  # make sure that only feature extractor is stored and not the whole model
        ###############################################################################################################


        if self.mode == 'FE-':  # return intermediate features instead of shape code
            return x3
        
        if store_attn_weights:
            collector.activate()    # activate here such that the sa and sfas will store the weights in the added module

        ############### SFA ###########################################################################################
        x3 = self.sa3_1(x3,x3).contiguous()

        if store_attn_weights:
            collector.add_indices(k_v_ids=org_ids[idx_0[idx_1[idx_2]]], q_ids=org_ids[idx_0[idx_1[idx_2]]], k_v_ref=str(self.input), q_ref=str(self.input))  # SFA
            collector.deactivate()  # make sure that only feature extractor is stored and not the whole model

        # print_log(f"ScaphoidFeatureExtractor: output shape {x3.shape}, batch size {batch_size}, N {N}")
        ############### maxpooling ####################################################################################
        shape_code = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)

        if self.mode == 'TNet':
            matrix = self.feature_transform_net(x3)
            return shape_code, matrix

        return shape_code
#####################################################################################################################################################################
    
##################################### Seed Generator ################################################################################################################
class ScaphoidSeedGenerator(nn.Module):
    """
    Shall generate sparse but complete point cloud
    """
    def __init__(self, sg_config):
        super(ScaphoidSeedGenerator, self).__init__()
        self.input = None
        if sg_config.mode != 'SG-':
            self.input = sg_config.input
        self.shape_code = sg_config.shape_code
        self.output = sg_config.output
        self.mode = sg_config.mode
        self.type = sg_config.type
        if self.type not in ['SG-', 'SG']:
            raise ValueError(f"Seed generator type {self.type} is not supported. Supported types are: {['SG-', 'SG']}")
        
        if self.mode not in ['SG-', 'pointattn']:
            raise ValueError(f"Seed generator mode {self.mode} is not supported. Supported modes are: {['SG-', 'pointattn']}")

        channel = 64
        if self.shape_code == 'shape_code_c':
            channel = 128


        self.channel = channel

        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        
        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        

    def forward(self, shape_code, batch_size):
        if store_attn_weights:
            # collector.activate()    # activate here such that the sa and sfas will store the weights in the added module
            collector.add_module(name='ScaphoidSeedGenerator', module_infos={'type': self.type, 'mode': self.mode, 'input': self.input, 'shape_code': self.shape_code, 'output': self.output})

        ############### seed generator
        x = self.relu(self.ps_adj(shape_code))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))        # -> [B, 512, 128]

        ############### SFA
        x0_d = (self.sa0_d(x, x))               # [B, 512 or 1024, 128] -> [B, 512, 128]
        ############### SFA
        x1_d = (self.sa1_d(x0_d, x0_d))         # [B, 512, 128] -> [B, 512, 128]
        ############### SFA + Reshape
        # x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1)   # valentino: adaptation to different N

        ############### MLP
        seeds = self.conv_out(self.relu(self.conv_out1(x2_d)))  # [B, 256, 256] -> [B, 64, 256] -> [B, 3, 256]

        return seeds
#####################################################################################################################################################################

    
##################################### Point Generator ###############################################################################################################
class ScaphoidPointGenerator(nn.Module):
    def __init__(self, pg_config, steps=[4, 8], input_channels=3):
        super(ScaphoidPointGenerator, self).__init__()

        self.seeds = pg_config.seeds
        self.shape_code = pg_config.shape_code
        self.output = pg_config.output
        self.mode = pg_config.mode
        self.type = pg_config.type
        if self.type != 'PG':
            raise ValueError(f"Point generator type {self.type} is not supported. Supported types are: PG")
        
        if self.mode not in ['pointattn', 'static']:
            raise ValueError(f"Point generator mode {self.mode} is not supported. Supported modes are: {['pointattn', 'static']}")
        
        N_shape_codes = 1
        cut_seeds = False
        if self.shape_code == 'shape_code_c':
            N_shape_codes = 2
            cut_seeds = False

        

        self.refine = PointGenerator(N_shape_codes, ratio=steps[0], input_channels=input_channels, cut_seeds=cut_seeds)
        self.refine1 = PointGenerator(N_shape_codes, ratio=steps[1], input_channels=3)

    def forward(self, x, seeds, shape_code):
        # print_log(f"{shape_code.shape}", color='blue')
        collector.add_module(name='ScaphoidPointGenerator_0', module_infos={'type': self.type, 'mode': self.mode, 'seeds': self.seeds, 'shape_code': self.shape_code, 'output': self.output})
        dense, feat_dense = self.refine(None, seeds, shape_code)        # None and feat_dense are not used in the function???????????
        collector.add_module(name='ScaphoidPointGenerator_1', module_infos={'type': self.type, 'mode': self.mode, 'seeds': self.seeds, 'shape_code': self.shape_code, 'output': self.output})
        dense1, feat_dense1 = self.refine1(feat_dense, dense, shape_code)

        return dense1, dense
    

class PointGenerator(nn.Module):
    def __init__(self, N_shape_codes, channel=128,ratio=1, input_channels=3, cut_seeds=False):
        super(PointGenerator, self).__init__()

        self.mode = 'double' if N_shape_codes == 2 else 'single'
        self.ratio = ratio
        self.cut_seeds = cut_seeds

        # self.mode = 'single'

        if self.mode == 'single':
            self.conv_shape = nn.Conv1d(512, 256, kernel_size=1)   
            self.conv_shape1 = nn.Conv1d(256, channel, kernel_size=1)
        elif self.mode == 'double':
            self.conv_shape = nn.Conv1d(512*2, 256*2, kernel_size=1)   
            self.conv_shape1 = nn.Conv1d(256*2, channel*2, kernel_size=1)
            self.conv_shape2 = nn.Conv1d(channel*2, channel, kernel_size=1)


        self.conv_seeds = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv_seeds1 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)

        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, seeds, shape_code):
        """
        :param x: Not used, no idea why it is passed
        :param seeds: Input seeds of shape [B, 3, N]
        :param shape_code: Shape code of shape [B, 512] or [B, 512*2] if N_shape_codes is 2
        """
        batch_size, _, N = seeds.size()

        y = self.conv_seeds1(self.relu(self.conv_seeds(seeds)))  # [B, 3, N] -> [B, 64, N] -> [B, channel(128), N]

        if self.mode == 'single':
            shape_code = self.conv_shape1(self.relu(self.conv_shape(shape_code)))  # [B, 512] -> [B, 256] -> [B, channel(128)]
        elif self.mode == 'double':
            shape_code = self.conv_shape1(self.relu(self.conv_shape(shape_code)))  # [B, 1028] -> [B, 512] -> [B, channel(128)*2]
            shape_code = self.conv_shape2(self.relu(shape_code))    # [B, channel(128)*2] -> [B, channel(128)]

        y0 = torch.cat([y,shape_code.repeat(1, 1, y.shape[-1])], dim=1)
            

        y1 = self.sa1(y0, y0)   # [B, channel*3(384), N] or [B, channel*3(256), N] -> [B, 512, N]
        y2 = self.sa2(y1, y1)   # [B, 512, N] -> [B, 512, N]
        y3 = self.sa3(y2, y2)   # [B, 512, N] -> [B, channel*ratio(128), N]

        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)   # [B, channel*ratio(128), N] -> [B, channel*ratio(128), N*self.ratio]

        y_up = y.repeat(1,1,self.ratio) # [B, 64, N] -> [B, 64, N*self.ratio]
        y_cat = torch.cat([y3,y_up],dim=1)  
        y4 = self.conv_delta(y_cat)

        if self.cut_seeds:
            try:
                x = self.conv_out(self.relu(self.conv_out1(y4))) + seeds[:,:3,:256].repeat(1,1,self.ratio*2)
            except:
                x = self.conv_out(self.relu(self.conv_out1(y4))) + seeds.repeat(1,1,self.ratio)
        else:
            x = self.conv_out(self.relu(self.conv_out1(y4))) + seeds.repeat(1,1,self.ratio)

        return x, y3

# class PointGenerator(nn.Module):
#     def __init__(self, N_shape_codes, channel=128,ratio=1):
#         super(PointGenerator, self).__init__()

#         self.ratio = ratio

#         self.conv_1 = nn.Conv1d(512 * N_shape_codes, 256, kernel_size=1)   
#         self.conv_11 = nn.Conv1d(256, channel, kernel_size=1)

#         self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
#         self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

#         self.sa1 = cross_transformer(channel*2,512)
#         self.sa2 = cross_transformer(512,512)
#         self.sa3 = cross_transformer(512,channel*ratio)

#         self.relu = nn.GELU()

#         self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

#         self.channel = channel

#         self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
#         self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        

#         self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


#     def forward(self, x, seeds, shape_code):
#         batch_size, _, N = seeds.size()

#         y = self.conv_x1(self.relu(self.conv_x(seeds)))  # B, C, N


#         shape_code = self.conv_1(self.relu(self.conv_1(shape_code)))  # B, C, N

#         y0 = torch.cat([y,shape_code.repeat(1,1,y.shape[-1])],dim=1)

#         y1 = self.sa1(y0, y0)
#         y2 = self.sa2(y1, y1)
#         y3 = self.sa3(y2, y2)
#         y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

#         y_up = y.repeat(1,1,self.ratio)
#         y_cat = torch.cat([y3,y_up],dim=1)
#         y4 = self.conv_delta(y_cat)

#         x = self.conv_out(self.relu(self.conv_out1(y4))) + seeds.repeat(1,1,self.ratio)

#         return x, y3
#####################################################################################################################################################################


from base.scaphoid_models.TNet import InputTransformNet, FeatureTransformNet
class TNet(nn.Module):

    def __init__(self, t_config):
        super(TNet, self).__init__()
        self.input = t_config.input
        self.output = t_config.output
        self.K = t_config.K
        self.mode = t_config.mode
        self.type = t_config.type
        if self.type != 'TNet':
            raise ValueError(f"Seed generator type {self.type} is not supported. Supported types are: 'TNet'")
        
        if self.mode not in ['input', 'features']:
            raise ValueError(f"Seed generator mode {self.mode} is not supported. Supported modes are: {['input', 'features']}")
        
        if self.mode == 'input':
            self.transform_net = InputTransformNet(K=self.K)
        elif self.mode == 'features':
            self.transform_net = FeatureTransformNet(K=self.K)

    def forward(self, point_cloud):
        """
        :param point_cloud: Input point cloud of shape [B, 3, N] or features of point cloud of shape [B, K, N]
        :return : Transformed point cloud of shape [B, 3, N] or features of point cloud of shape [B, K, N]
        """
        transform = self.transform_net(point_cloud) # shape [B, K, K]

        transformed_point_cloud = torch.bmm(point_cloud.transpose(1, 2), transform)
        transformed_point_cloud = transformed_point_cloud.transpose(1, 2)  # shape [B, 3, N]
        
        return transformed_point_cloud.contiguous()

class ASSIGN(nn.Module):
    def __init__(self, assign_config):
        super(ASSIGN, self).__init__()
        self.seeds = assign_config.seeds
        self.sparse_pc = assign_config.sparse_pc
        self.pre_points = assign_config.pre_points
        self.points = assign_config.points
        self.type = assign_config.type
        self.mode = assign_config.mode

        if self.type not in ['ASSIGN']:
            raise ValueError(f"Unknown type {self.type}. Available types are: ASSIGN")
        if self.mode not in ['assign']:
            raise ValueError(f"Unknown mode {self.mode}. Available modes are: {['assign']}")
        
    def __call__(self, input: list):
        """
        Implemented in ScaphoidPointAttN itself.
        """
        pass

class Aligner(nn.Module):
    def __init__(self, aligner_config):
        super(Aligner, self).__init__()
        self.src = aligner_config.src
        self.tgt = aligner_config.tgt
        self.src_pose = aligner_config.src_pose
        self.tgt_pose = aligner_config.tgt_pose
        self.output = aligner_config.output
        self.mode = aligner_config.mode
        self.type = aligner_config.type

        if self.type not in ['ALIGNER']:
            raise ValueError(f"Unknown type {self.type}. Available types are: ALIGNER")
        if self.mode not in ['aligner_SSM', 'aligner_anchor']:
            raise ValueError(f"Unknown mode {self.mode}. Available modes are: {['aligner_SSM', 'aligner_anchor']}")
        
    def __call__(self, input: list):
        """
        :param input: List of inputs containing pose_RTS_support, pose_RTS_anchor, pcd_align_support, pcd_align_anchor
        :return: Aligned point clouds in SSM space or aligned support point cloud to anchor point cloud.
        """

        if self.mode == 'aligner_SSM':
            """
            Align Support and Anchor point clouds into SSM space.
            Later the GT point cloud also needs to be aligned into SSM space.
            """
            if len(input) != 4:
                raise ValueError(f"Expected 4 inputs, but got {len(input)} inputs.")
            
            pose_RTS_support, pose_RTS_anchor, pcd_align_support, pcd_align_anchor = input

            ########################## Map both into SSM space ##########################
            pcd_align_support = t.apply_reverse_RTS_transformation(pcd_align_support, pose_RTS_support)
            pcd_align_anchor = t.apply_reverse_RTS_transformation(pcd_align_anchor, pose_RTS_anchor)


            return pcd_align_support.contiguous(), pcd_align_anchor.contiguous()
        
        elif self.mode == 'aligner_anchor':
            """
            Align Support point cloud to Anchor point cloud.
            """
            if len(input) != 4:
                raise ValueError(f"Expected 4 inputs, but got {len(input)} inputs.")
            
            pose_RTS_support, pose_RTS_anchor, pcd_align_support, pcd_align_anchor = input

            pcd_align_support = t.apply_reverse_RTS_transformation(pcd_align_support, pose_RTS_support)
            pcd_align_support = t.apply_RTS_transformation(pcd_align_support, pose_RTS_anchor)

            return pcd_align_support.contiguous(), pcd_align_anchor.contiguous()

        else:
            raise ValueError(f"Unknown mode {self.mode}. Available modes are: CONCAT, AFFIL")


class OP(nn.Module):
    def __init__(self, op_config):
        super(OP, self).__init__()
        self.input = op_config.input
        self.output = op_config.output
        self.mode = op_config.mode
        self.type = op_config.type

        if self.type not in ['OP']:
            raise ValueError(f"Unknown type {self.type}. Available types are: OP")
        if self.mode not in ['concat-dim-1', 'concat-dim-2', 'affil', 'seeds_to_sparse_pc', 'fps', 'aligner']:
            raise ValueError(f"Unknown mode {self.mode}. Available modes are: {['concat-dim-1', 'concat-dim-2', 'affil', 'seeds_to_sparse_pc', 'fps', 'aligner']}")


    def __call__(self, input: list):
        if self.mode == 'concat-dim-2':
            concat = torch.cat(input, dim=2)
            return concat
        elif self.mode == 'concat-dim-1':
            concat = torch.cat(input, dim=1)
            return concat

        elif self.mode == 'affil':
            if len(input) != 2:
                raise ValueError(f"Expected 2 inputs, but got {len(input)} inputs.")
            xyz_0, xyz_1 = input
            affil_0 = torch.zeros_like(xyz_0[:, :1, :])
            affil_1 = torch.ones_like(xyz_1[:, :1, :])
            xyz_affil_0 = torch.cat([xyz_0, affil_0], dim=1)
            xyz_affil_1 = torch.cat([xyz_1, affil_1], dim=1)
            return torch.cat([xyz_affil_0, xyz_affil_1], dim=2)
        
        elif self.mode == 'seeds_to_sparse_pc':
            if len(input) != 2:
                raise ValueError(f"Expected 2 inputs, but got {len(input)} inputs.")
            seeds, sg_input = input
            sparse_pc = torch.cat([sg_input, seeds], dim=2)
            sparse_pc = gather_points(seeds, furthest_point_sample(seeds.transpose(1, 2).contiguous(), 512))
            return sparse_pc
        
        elif self.mode == 'fps':
            if type(input) == list:
                input = input[0]
            if input.shape[1] == 3:
                idx = furthest_point_sample(input.transpose(1, 2).contiguous(), input.shape[2]) # expects B,N,3
            elif input.shape[2] == 3:
                idx = furthest_point_sample(input.transpose(1, 2).contiguous(), input.shape[1])
            return gather_points(input, idx)
        
        # elif self.mode == 'apply_affine':
        #     if len(input) != 2:
        #         raise ValueError(f"Expected 2 inputs, but got {len(input)} inputs.")
        #     partial_pcd, ret_affine_mat = input
        #     if ret_affine_mat.shape[1] != 4 or ret_affine_mat.shape[2] != 4:
        #         raise ValueError(f"Expected affine matrix of shape [B, 4, 4], but got {ret_affine_mat.shape}.")
            
        #     rotated_partial_pcd = apply_affine_transformation(partial_pcd, ret_affine_mat)

        #     return rotated_partial_pcd
        
        # elif self.mode == 'apply_reverse_affine':
        #     if len(input) != 2:
        #         raise ValueError(f"Expected 2 inputs, but got {len(input)} inputs.")
        #     partial_pcd, ret_affine_mat = input
        #     if ret_affine_mat.shape[1] != 4 or ret_affine_mat.shape[2] != 4:
        #         raise ValueError(f"Expected affine matrix of shape [B, 4, 4], but got {ret_affine_mat.shape}.")
            
        #     reverse_affine_mat = get_reverse_affine_matrix(ret_affine_mat)
        #     rotated_partial_pcd = apply_affine_transformation(partial_pcd, reverse_affine_mat)

        #     return rotated_partial_pcd
        
        elif self.mode == 'aligner':
            if len(input) != 4:
                raise ValueError(f"Expected 4 inputs, but got {len(input)} inputs.")
            affine_matrix_source, affine_matrix_target, pcd_align_source, pcd_align_target = input
            # affine_matrix_target, affine_matrix_source, pcd_align_source, pcd_align_target = input

            # Rotation extractors generate reverse affine mats -> hence reverse source transformations by applying the affine_matrix_source
            # and apply reversed target affine matrix to apply target transformation to the source point cloud

            reverse_affine_mat = t.get_reverse_affine_matrix(affine_matrix_target)
            
            align_affine_mat = torch.bmm(reverse_affine_mat, affine_matrix_source)

            aligned_pcd = t.apply_affine_transformation(pcd_align_source.transpose(1, 2).contiguous(), align_affine_mat).transpose(1, 2).contiguous()  # [B, N, 3] -> [B, 3, N] -> [B, N, 3]



            return aligned_pcd, align_affine_mat
        

        else:
            raise ValueError(f"Unknown mode {self.mode}. Available modes are: CONCAT, AFFIL")
