import torch
import torch.nn as nn
from torch import Tensor
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_

from submodules.PoinTr.utils import misc
from submodules.PoinTr.extensions.chamfer_dist import ChamferDistanceL1
from submodules.PoinTr.models.Transformer_utils import *
from submodules.PoinTr.models.AdaPoinTr import DGCNN_Grouper, SimpleEncoder, PointTransformerEncoderEntry
from submodules.PoinTr.models.AdaPoinTr import PointTransformerDecoderEntry, SimpleRebuildFCLayer, Fold
from submodules.PointAttn.utils.mm3d_pn2 import furthest_point_sample, gather_points

from base.scaphoid_metrics.weighted_dist_chamfer_3D import calc_weighted_cd


class ScaphoidPointsProxiesEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # print_log(f'point_proxies: {point_proxies.size()} coor: {coor.size()}', logger='MODEL')
        # point_proxies: torch.Size([8, 512, 384]) coor: torch.Size([8, 512, 3])
        # from adapointr x+pe: torch.Size([16, 256, 384]), coor: torch.Size([16, 256, 3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        """
        @param xyz: B N 3
        @return center_p: center_points, f: feature, pe: position_embedding, x: input_proj
        """
        center_p, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(center_p)
        x = self.input_proj(f)

        return center_p, pe, x



######################################## PCTransformer ########################################   
class ScaphoidPCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        # print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        self.volar_proxies_encoder = ScaphoidPointsProxiesEncoder(config)
        # self.dorsal_proxies_encoder = ScaphoidPointsProxiesEncoder(config)

        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        center_p_volar, pe_volar, x_volar = self.volar_proxies_encoder(xyz)
        # center_p_dorsal, pe_dorsal, x_dorsal = self.dorsal_proxies_encoder(xyz_dorsal)
        point_proxies_volar = x_volar + pe_volar
        point_proxies = point_proxies_volar
        coor = center_p_volar
        # point_proxies_dorsal = x_dorsal + pe_dorsal

        # point_proxies = torch.cat([point_proxies_volar, point_proxies_dorsal], dim=1)
        # expand 3 channels to 4 channels
        # center_p_volar = torch.cat((center_p_volar, torch.zeros(8, 256, 1).cuda()), dim=2)
        # center_p_dorsal = torch.cat((center_p_dorsal, torch.ones(8, 256, 1).cuda()), dim=2)
        # coor = torch.cat([center_p_volar, center_p_dorsal], dim=1)

        # print_log(f'point_proxies: {point_proxies.size()} coor: {coor.size()}', logger='MODEL')
        # point_proxies: torch.Size([8, 512, 384]) coor: torch.Size([8, 512, 3])
        # from adapointr x+pe: torch.Size([16, 256, 384]), coor: torch.Size([16, 256, 3])

        x = self.encoder(point_proxies, coor) # b n c
        global_feature = self.increase_dim(x) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=center_p_volar, denoise_length=denoise_length)

            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=center_p_volar)

            return q, coarse, 0

    # def forward(self, xyz_volar, xyz_dorsal):
    #     bs = xyz_volar.size(0)
    #     center_p_volar, pe_volar, x_volar = self.volar_proxies_encoder(xyz_volar)
    #     center_p_dorsal, pe_dorsal, x_dorsal = self.dorsal_proxies_encoder(xyz_dorsal)
    #     point_proxies_volar = x_volar + pe_volar
    #     point_proxies_dorsal = x_dorsal + pe_dorsal

    #     point_proxies = torch.cat([point_proxies_volar, point_proxies_dorsal], dim=1)
    #     # expand 3 channels to 4 channels
    #     # center_p_volar = torch.cat((center_p_volar, torch.zeros(8, 256, 1).cuda()), dim=2)
    #     # center_p_dorsal = torch.cat((center_p_dorsal, torch.ones(8, 256, 1).cuda()), dim=2)
    #     coor = torch.cat([center_p_volar, center_p_dorsal], dim=1)

    #     # print_log(f'point_proxies: {point_proxies.size()} coor: {coor.size()}', logger='MODEL')
    #     # point_proxies: torch.Size([8, 512, 384]) coor: torch.Size([8, 512, 3])
    #     # from adapointr x+pe: torch.Size([16, 256, 384]), coor: torch.Size([16, 256, 3])

    #     x = self.encoder(point_proxies, coor) # b n c
    #     global_feature = self.increase_dim(x) # B 1024 N 
    #     global_feature = torch.max(global_feature, dim=1)[0] # B 1024

    #     coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

    #     coarse_inp = misc.fps(xyz_volar, self.num_query//2) # B 128 3
    #     coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

    #     mem = self.mem_link(x)

    #     # query selection
    #     query_ranking = self.query_ranking(coarse) # b n 1
    #     idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
    #     coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

    #     if self.training:
    #         # add denoise task
    #         # first pick some point : 64?
    #         picked_points = misc.fps(xyz_volar, 64)
    #         picked_points = misc.jitter_points(picked_points)
    #         coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
    #         denoise_length = 64     

    #         # produce query
    #         q = self.mlp_query(
    #         torch.cat([
    #             global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
    #             coarse], dim = -1)) # b n c

    #         # forward decoder
    #         q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=center_p_volar, denoise_length=denoise_length)

    #         return q, coarse, denoise_length

    #     else:
    #         # produce query
    #         q = self.mlp_query(
    #         torch.cat([
    #             global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
    #             coarse], dim = -1)) # b n c
            
    #         # forward decoder
    #         q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=center_p_volar)

    #         return q, coarse, 0

######################################## PoinTr ########################################  

# @MODELS.register_module()
class ScaphoidAdaPoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = ScaphoidPCTransformer(config)
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step**2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step**2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret: list[Tensor], gt_list: list[Tensor], pc_subdivision: dict=None):
        """
        :param ret: tuple of (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
        :param gt_list: list with one element: ground truth point cloud of shape [B, N, 3]
        :param pc_subdivision: dict containing subdivision information for the point cloud, like 
            {'volar': [0, 1, 2], 'dorsal': [3, 4, 5]}
        :return loss_dict: dictionary containing the following
        1. loss_denoised: loss for denoised points
        2. loss_recon: loss for coarse and fine reconstruction
        """
        if len(ret) != 4:
            loss_dict = {'Loss/Sparse': torch.tensor(0.0), 'Loss/Dense': torch.tensor(0.0), 
                         'Loss/Denoised': torch.tensor(0.0), 'Loss/Total': torch.tensor(0.0)}
            return loss_dict
        else:
            pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        gt = gt_list[0]  

        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        loss_dict = {'Loss/Sparse': loss_coarse, 'Loss/Dense': loss_fine, 'Loss/Denoised': loss_denoised, 
                     'Loss/Total': loss_recon + loss_denoised}

        return loss_dict

    def get_metrics(self, ret_list: list[Tensor], gt_list: list[Tensor], pc_subdivision: dict, 
                    scaled_real_world_thresh: list[tuple[str, float]]=None):
        """
        :param ret_list: tuple of (pred_coarse, pred_fine)
        :param gt_list: ground truth point cloud of shape [B, N, 3]
        :param pc_subdivision: dict containing subdivision information for the point cloud, like 
            {'volar': [0, 1, 2], 'dorsal': [3, 4, 5]}
        :return: detailed_metrics_dense, detailed_metrics_sparse containing 
            {Full/F1, Volar/F1, Dorsal/F1, Distal/F1, Proximal/F1, ...}
        """
        dense1, sparse_pc = ret_list
        gt = gt_list[0]

        assert gt.shape[1] == 8192, f"Expected gt shape [B, 8192, 3], but got {gt.shape}"
        assert dense1.shape[1] == 8192, f"Expected dense1 shape [B, 8192, 3], but got {dense1.shape}"

        dense1_ids = furthest_point_sample(dense1, gt.shape[1])
        dense1 = gather_points(dense1.transpose(1, 2).contiguous(), dense1_ids).transpose(1, 2).contiguous()

        detailed_metrics_dense = calc_weighted_cd(dense1, gt, pc_subdivision, calc_detailed_metrics=True, calc_f1=True, 
                                                  scaled_real_world_thresh=scaled_real_world_thresh)

        renamed_detailed_metrics_dense = {}
        for prefix, metrics in detailed_metrics_dense.items():
            for metric_name, metric_value in metrics.items():
                renamed_detailed_metrics_dense[f'{prefix[0].upper() + prefix[1:]}/{metric_name}'] = metric_value

        return renamed_detailed_metrics_dense, None #, detailed_metrics_sparse

    def forward(self, xyz_volar, xyz_dorsal, use='volar'):
        """
        :param xyz_volar: [B, N, 3] point cloud for volar side
        :param xyz_dorsal: [B, N, 3] point cloud for dorsal side
        :param use: 'volar' or 'dorsal' or 'both' to specify which side to use for reconstruction
        :return: tuple of (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
        """
        # xyz = torch.cat([xyz_volar, xyz_dorsal], dim=1)  # B N 3
        # xyz = xyz_volar
        if use == 'volar':
            xyz = xyz_volar
        elif use == 'dorsal':
            xyz = xyz_dorsal
        elif use == 'both':
            xyz = torch.cat([xyz_volar, xyz_dorsal], dim=1)
        else:
            raise ValueError(f"Invalid use parameter: {use}. Expected 'volar', 'dorsal', or 'both'.")
        q, coarse_point_cloud, denoise_length = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        
        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret  # sparse, dense