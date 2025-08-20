
import numpy as np
from sklearn.decomposition import PCA

from base.scaphoid_utils.logger import print_log


from base.scaphoid_datasets.BaseDataset import BaseDataset
from base.scaphoid_datasets.PoseEstDataset import PoseEstDataset
# from base.scaphoid_datasets.RotationDataset import RotationDataset
import base.scaphoid_utils.transformations as t



    
class ScaphoidDataset(BaseDataset):
    """
    Dataset used during Point Cloud Completion Task
    """

    def __init__(self, subset: str, config: dict, transforms: list=None, strict=True, transform_with='volar', 
                 debug=False, logger=None, model_name='ScaphoidPointAttN'):
        
        super().__init__(subset, config, transforms, strict=strict, transform_with=transform_with, 
                         debug=debug, logger=logger)
        self.model_name = model_name
        print_log(f"ScaphoidDataset initialized with model: {self.model_name}", logger, color='green')
        print_log("#"*61, logger=logger, color='green')
    
    def __getitem__(self, idx):
        sample_partial_volar, sample_partial_dorsal, sample_full = self.get_pcd_sample(idx)
        inds = self.get_indices_sample(idx)

        if self.model_name == 'ScaphoidPointAttN' or self.model_name == 'ScaphoidAdaPoinTr':
            _, transformed_data, transform_paras = BaseDataset.get_completion_data(
                sample_partial_volar, 
                sample_partial_dorsal, 
                sample_full, 
                self.pre_processing_transforms, 
                self.transforms
            )
            transformed_data = (sample_partial_volar, sample_partial_dorsal, sample_full)

            # adds RTS matrices -> can be used to reverse the transformation later even when not needed for training the model
            transform_paras = PoseEstDataset.create_RTS_matrices(transform_paras)   

            output = (transformed_data, 0, inds, transform_paras)

        # elif self.model_name == 'ScaphoidRotationPointAttN':
            
        #     pre_processed_data, (sample_partial_volar, sample_partial_dorsal, sample_full), transform_paras = BaseDataset.get_completion_data(sample_partial_volar, sample_partial_dorsal, sample_full, self.pre_processing_transforms, self.transforms)
        #     transformed_data = (sample_partial_volar, sample_partial_dorsal, sample_full)

        #     new_sample_partial_volar, new_sample_partial_dorsal, new_sample_full = self.get_pcd_sample(idx)
        #     rotation_gt, transform_paras = RotationDataset.get_rotation_data(new_sample_partial_volar, new_sample_partial_dorsal, new_sample_full, transform_paras, self.transform_with)

        #     output = (transformed_data, rotation_gt, inds, transform_paras)


        elif self.model_name == 'CompletionPoseEstPointAttN' or self.model_name == 'PoseEstPointAttN':
            pre_processing_transforms = PoseEstDataset._get_pre_processing_transforms(self.subset, ['GeneralDemeaning'])
            transformed_data, pose_pcd_gts, transform_paras = PoseEstDataset.get_pose_estimation_data(
                sample_partial_volar, 
                sample_partial_dorsal, 
                sample_full, 
                pre_processing_transforms, 
                self.transforms
            )
            output = (transformed_data, pose_pcd_gts, inds, transform_paras)

        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented for ScaphoidDataset.")
        
        
        return output