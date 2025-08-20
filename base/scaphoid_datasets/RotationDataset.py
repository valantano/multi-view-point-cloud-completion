# import numpy as np


# import numpy as np
# import torch

# from base.scaphoid_utils.logger import print_log
# from base.scaphoid_datasets.BaseDataset import BaseDataset

# import base.scaphoid_utils.transformations as t



# class RotationDataset(BaseDataset):
#     def __init__(self, subset: str, config: dict, transforms: list=None, transform_with='volar', debug=False, logger=None, strict=False):
#         super().__init__(subset, config, transforms, strict=strict, transform_with=transform_with, debug=debug, logger=logger)
#         # transforms = ['CoupledDemeaning', 'CoupledRandomRotation', 'CoupledRescale']
#         # transform_with = 'volar'
#         # self.transforms_coupled = self._get_transforms(self.subset, transforms, transform_with)

#         # transforms = ['DecoupledDemeaning', 'DecoupledRandomRotation', 'CoupledRescale']
#         transforms = ['StaticDecoupledDemeaning', 'StaticDecoupledRandomRotation', 'CoupledRescale']
#         self.transforms_decoupled = self._get_transforms(self.subset, transforms, transform_with)


#     @staticmethod
#     def get_rotation_data(sample_partial_volar_org, sample_partial_dorsal_org, sample_full_org, transform_paras, transform_with) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

#         for key in list(transform_paras.keys()):    # conversion to list to avoid RuntimeError: dictionary changed size during iteration
#             name = key.split('_')[-1]
#             # rescale = np.array([0, 0, 1])
#             # rotation = np.array([0,0,0])
#             # translation = np.array([0,0,0])
#             rescale = transform_paras[key][2]['rescale']
#             rotation = transform_paras[key][1]['rotation'].squeeze()
#             translation = transform_paras[key][0]['demean'].squeeze()

            
#             transform_paras[f"affine_{name}"] = torch.tensor(t.get_affine_matrix(rescale, rotation, translation), dtype=torch.float32)


#         rescale = transform_paras[f'transform_partial_{transform_with}'][2]['rescale']
#         gt_partial_transform = torch.tensor(t.get_affine_matrix(rescale, np.array([0,0,0]), np.array([0,0,0])), dtype=torch.float32)

#         # rescale not translated and rotated samples according to anchor source
#         sample_partial_volar_org = t.apply_affine_transformation(torch.Tensor(sample_partial_volar_org).unsqueeze(0), gt_partial_transform.unsqueeze(0)).squeeze(0)
#         sample_partial_dorsal_org = t.apply_affine_transformation(torch.Tensor(sample_partial_dorsal_org).unsqueeze(0), gt_partial_transform.unsqueeze(0)).squeeze(0)
#         sample_full_org = t.apply_affine_transformation(torch.Tensor(sample_full_org).unsqueeze(0), gt_partial_transform.unsqueeze(0)).squeeze(0)

#         transform_paras['gt_partial_transform'] = gt_partial_transform

#         return (sample_partial_volar_org, sample_partial_dorsal_org, sample_full_org), transform_paras
    


#     def __getitem__(self, idx):
#         sample_full, sample_partial_volar, sample_partial_dorsal = self.all_full[idx, :, :], self.all_partial_volar[idx, :, :], self.all_partial_dorsal[idx, :, :]
#         # sample_volar_ind, sample_dorsal_ind, sample_articular_ind = self.all_volar_points_ind[idx, :], self.all_dorsal_points_ind[idx, :], self.all_articular_points_ind[idx, :]
#         # sample_distal_ind, sample_proximal_ind = self.all_distal_points_ind[idx, :], self.all_proximal_points_ind[idx, :]
#         org_partial_volar, org_partial_dorsal, org_full = sample_partial_volar.copy(), sample_partial_dorsal.copy(), sample_full.copy()

#         if self.transform_with == 'volar':
#             gt = org_partial_volar
#         elif self.transform_with == 'dorsal':
#             gt = org_partial_dorsal

#         data_dec = {}
#         data_dec['partial_volar'], data_dec['partial_dorsal'], data_dec['gt'] = sample_partial_volar, sample_partial_dorsal, gt
#         data_dec['transform_partial_volar'], data_dec['transform_partial_dorsal'], data_dec['transform_gt'] = [], [], []


#         # if self.transforms is not None:
#         data_dec = self.transforms_decoupled(data_dec)
#         # data_coup = self.transforms_coupled(data_coup)


#         transform_keys = ['transform_partial_volar', 'transform_partial_dorsal', 'transform_gt']
#         transform_paras_dec = {key: data_dec[key] for key in transform_keys}
#         # transform_paras_coup = {key: data_coup[key] for key in transform_keys}


#         for key in transform_keys:
#             name = key.split('_')[-1]
#             # print(f"Transform paras {key}: {transform_paras_dec[key]}")
#             rescale = np.array([])
#             rotation = np.array([0,0,0])
#             translation = np.array([0,0,0])
#             try:
#                 rescale = transform_paras_dec[key][2]['rescale']
#                 rotation = transform_paras_dec[key][1]['rotation'].squeeze()
#                 translation = transform_paras_dec[key][0]['demean'].squeeze()
                
#             except:
#                 pass
            
#             # transform_paras_dec[f"affine_{name}"] = torch.tensor(get_affine_matrix_without_scaling(rescale, rotation, translation), dtype=torch.float32)
#             transform_paras_dec[f"affine_{name}"] = torch.tensor(t.get_affine_matrix(rescale, rotation, translation), dtype=torch.float32)


#         return data_dec['partial_volar'], data_dec['partial_dorsal'], data_dec['gt'], transform_paras_dec