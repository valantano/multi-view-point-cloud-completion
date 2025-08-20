import numpy as np
import torch
from scipy.spatial.transform import Rotation

from submodules.PoinTr.datasets.io import IO
from submodules.PoinTr.utils.logger import *


from base.scaphoid_utils.logger import print_log
from base.scaphoid_datasets.Transforms import Compose
from base.scaphoid_datasets.BaseDataset import BaseDataset

import base.scaphoid_utils.transformations as t


class PoseEstDataset(BaseDataset):
    """
    Dataset used during Point Cloud Pose Estimation Task
    """

    def __init__(self, subset: str, config: dict, transforms: list=None, transform_with='volar', debug=False, 
                 logger=None, strict=False):
        """
        :param subset: dataset subset (train/val/test)
        :param config: dataset configuration
        :param transforms: list of transforms to be applied
        :param transform_with: which part of the point cloud to use for transformation
        :param debug: whether to enable debug mode
        :param logger: logger instance
        :param strict: whether to use strict or unstrict dataset
        """
        super().__init__(subset, config, transforms, strict=strict, transform_with=transform_with, debug=debug, 
                         logger=logger)

        # pre_processing_transforms = ['GeneralRandomMirror','GeneralDemeaning']
        # self.pre_processing_transforms = self._get_pre_processing_transforms(subset, pre_processing_transforms)

        # workaround to train the aligner model which will be used to align the test set before training the pose estimation model
        if transform_with == 'full':    
            transforms = ['GeneralDemeaning', 'GeneralRandomRotation', 'GeneralRescale']
            self.transforms = self._get_aligner_transforms(subset, transforms)
        else:
            transforms = ['DecoupledDemeaning', 'DecoupledRandomRotation', 'CoupledRescale']
            self.transforms = self._get_transforms(self.subset, transforms, transform_with)

    @staticmethod
    def _get_aligner_transforms(subset: str, transform_list: list=None):
        if transform_list is None:
            transform_list = ['GeneralDemeaning', 'GeneralRescale']

        transforms = [{
            'callback': f'{t}',
            'parameters': {
                'input_keys': {
                    'sources': ['partial_volar', 'partial_dorsal', 'full'],
                    'targets': ['partial_volar', 'partial_dorsal', 'full'],
                }
            },
            'objects': ['full']
        } for t in transform_list]

        transforms.append({
                'callback': 'ToTensor',
                'objects': ['partial_volar', 'partial_dorsal', 'full']})
        return Compose(transforms)
    
    
    @staticmethod
    def create_RTS_matrices(transform_paras):
        """
        Create RTS matrices for the point clouds using the transform parameters.
        :param transform_paras: transformation parameters
        :return: RTS matrices for the point clouds
        """
        # conversion to list needed to avoid RuntimeError: dictionary changed size during iteration
        for key in list(transform_paras.keys()):    
            name = key.split('_')[-1]
            rescale = np.array([0, 0, 1])
            rotation = np.array([0,0,0])
            translation = np.array([0,0,0])
            try:
                translation = -transform_paras[key][0]['demean'].squeeze()
                rotation = transform_paras[key][1]['rotation'].squeeze()
                rescale = transform_paras[key][2]['rescale']
            except:
                pass

            transform_paras[f"{name}_RTS"] = torch.tensor(t.get_RTS(rotation, translation, rescale), 
                                                          dtype=torch.float32)

        return transform_paras


    @staticmethod
    def get_pose_estimation_data(sample_partial_volar, sample_partial_dorsal, sample_full, pre_processing_transforms, 
                                 transforms):
        """
        :param sample_partial_volar: partial volar point cloud
        :param sample_partial_dorsal: partial dorsal point cloud
        :param sample_full: full point cloud
        :param pre_processing_transforms: pre-processing transforms to be applied
        :param transforms: transforms to be applied
        :return: transformed data, ground truth point clouds, and transformation parameters
        """
        # Transform Partial Point Clouds according to real scenario, using partial point clouds as source
        pose_pcd_gts, transformed_data, transform_paras = BaseDataset.get_completion_data(
            sample_partial_volar,
            sample_partial_dorsal, 
            sample_full, 
            pre_processing_transforms, 
            transforms
        )

        # Create RTS Matrices for Point Clouds
        transform_paras = PoseEstDataset.create_RTS_matrices(transform_paras)

        # Convert gts to tensors
        gt_partial_volar, gt_partial_dorsal, gt_full = (torch.tensor(arr, dtype=torch.float32) for arr in pose_pcd_gts)

        return transformed_data, (gt_partial_volar, gt_partial_dorsal, gt_full), transform_paras

    def __getitem__(self, idx):
        """
        Override method of BaseDataset
        Get the item at index idx.
        :param idx: index of the sample
        :return: transformed data (partial_volar, partial_dorsal, full), 
            pose_pcd_gts (ground truth point clouds for pose estimation), 
            and transformation parameters
        """
        sample_partial_volar, sample_partial_dorsal, sample_full = self.get_pcd_sample(idx)

        transformed_data, pose_pcd_gts, transform_paras = PoseEstDataset.get_pose_estimation_data(
            sample_partial_volar, sample_partial_dorsal, sample_full, self.pre_processing_transforms, self.transforms)

        return transformed_data, pose_pcd_gts, transform_paras