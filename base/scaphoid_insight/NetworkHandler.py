import os
from typing import Dict
import torch
from pathlib import Path


from base.scaphoid_utils import parser
from base.scaphoid_utils.config import ConfigHandler
from base.scaphoid_utils.misc import worker_init_fn
from base.scaphoid_utils.logger import Logger, print_log
import base.scaphoid_utils.transformations as t
import base.scaphoid_utils.builder as b

from base.scaphoid_datasets.ScaphoidDataset import ScaphoidDataset

from base.scaphoid_models.ScaphoidPointAttN import ScaphoidPointAttN
from base.scaphoid_models.PoseEstPointAttN import CompletionPoseEstPointAttN, PoseEstPointAttN
from base.scaphoid_models.ScaphoidAdaPoinTr import ScaphoidAdaPoinTr
from tqdm import tqdm
import base.scaphoid_models.ScaphoidPointAttN as module


dataset_to_use = 'test'  # 'train', 'val', 'test'

EXPERIMENT_FOLDER = '/home/valantano/mt/repository/base/mvpcc_experiments'
CONFIG_FOLDER = '/home/valantano/mt/repository/base/cfgs/Scaphoid_models'


class Network:

    def __init__(self, config_folder='BaselineCfgs', config_name='ScaphoidPointAttN_Baseline_Min_dorsal.yaml', 
                 exp_name='FBaseMinDorsal0'):
        self.logger = None
        self.args = None
        self.config = None
        self.load_config(config_folder, config_name, exp_name)  # loads above parameters

        self.active_dataset = None

        self.active_dataloader = None
        self.data_loader_workaround = None  # workaround to avoid reloading dataset every time
        self.load_dataset(self.config) # loads above parameters


        self.model = None
        self.collector = None
        # self.load_model(self.args, self.config) # loads above parameters

        self.idx = 0
        self.scale = 1.0
        self.transform_paras = None

        

    def get_in_out(self) -> dict[str, any]:
        """Get input/output data from the collector."""
        if self.collector is None:
            raise ValueError("Collector is None. Please process a sample first.")
        return self.collector.in_out

    def reload(self, 
               config_folder: str = 'SeedAttnMatcherCfgs', 
               config_name: str = 'ScaphoidPointAttN_SeedAttnMatcher_volar', 
               exp_name: str = '+SAMV') -> None:
        """
        Reload configuration, dataset, and model.
        
        Args:
            config_folder: Configuration folder name
            config_name: Configuration file name
            exp_name: Experiment name
        """
        self.load_config(config_folder, config_name, exp_name)
        self.load_dataset(self.config)
        self.load_model(self.args, self.config)

        
    def load_config(self, config_folder='SeedAttnMatcherCfgs', config_name='ScaphoidPointAttN_SeedAttnMatcher_volar', 
                    exp_name='+SAMV'):
        self.config_base_folder = Path(CONFIG_FOLDER)
        self.config_folder = os.path.join(self.config_base_folder, config_folder)
        self.config_name = config_name
        config = os.path.join(self.config_folder, config_name)
        self.exp_name = exp_name

        args = parser.get_args()
        args.config_folder = os.path.join(self.config_base_folder, '..')
        args.config = config
        args.exp_name = self.exp_name

        args.resume = True
        args.distributed = False
        args.test = True

        exp_batch_path = os.path.join(EXPERIMENT_FOLDER, Path(args.config).parent.stem)
        args.experiment_path = os.path.join(exp_batch_path, Path(args.config).stem, args.exp_name)
        args.tfboard_path = os.path.join(exp_batch_path, Path(args.config).stem, 'TFBoard', args.exp_name)
        args.log_name = args.experiment_path
        print_log(f"Experiment Path: {args.config}", color='blue')
        print_log(f"Experiment Path: {args.experiment_path}", color='blue')

        
        config_handler = ConfigHandler(args.config_folder, args.resume)
        config = config_handler.get_config(args)  # if args.resume then ignore args.config and use config in experiment path
        config.dataset.train.others.bs = 1

        self.logger = Logger(args.log_name)
        self.args = args
        self.config = config
        
        
    def load_dataset(self, config: any = None) -> None:
        """
        Load dataset based on configuration.
        
        Args:
            config: Configuration object containing dataset parameters
        """
        transforms = None
        transform_with = None
        try:
            transforms = config.dataset.transforms
            transform_with = config.dataset.transform_with
            # transforms = [
            #     'CoupledRescale'
            # ]
        except:
            pass
        
        if self.active_dataset is None or self.active_dataset.model_name != config.model_name:
            STRICT = False
            DEBUG = False
            if dataset_to_use == 'train':
                self.active_dataset = ScaphoidDataset('train', config.dataset.train._base_, transforms, STRICT, 
                                                      transform_with, DEBUG, self.logger, config.model_name)
            if dataset_to_use == 'val':
                self.active_dataset = ScaphoidDataset('val', config.dataset.val._base_, transforms, STRICT, 
                                                      transform_with, DEBUG, self.logger, config.model_name)
            if dataset_to_use == 'test':
                self.active_dataset = ScaphoidDataset('test', config.dataset.test._base_, transforms, STRICT, 
                                                      transform_with, DEBUG, self.logger, config.model_name)

        else:
            self.active_dataset.change_transforms(transforms, transform_with)
        
        print_log(f"Loaded dataset {self.active_dataset.__class__.__name__} with {len(self.active_dataset)} samples \
                  and {self.active_dataset.transform_with} {transform_with}", self.logger, color='blue')

        
        self.active_dataloader = torch.utils.data.DataLoader(self.active_dataset, batch_size=1, shuffle=False, 
                                                             drop_last=False, num_workers=int(4), 
                                                             worker_init_fn=worker_init_fn)

    
    def load_model(self, args, config=None):
        print_log(f"############### Loading Model: {config.model_name} #################", self.logger, color='blue')
        if config.model_name == 'ScaphoidAdaPoinTr':
            base_model = ScaphoidAdaPoinTr(config.model)
            base_model = torch.nn.DataParallel(base_model).cuda()
        elif config.model_name == 'ScaphoidPointAttN':
            base_model = ScaphoidPointAttN(config)
            base_model = torch.nn.DataParallel(base_model).cuda()
        elif config.model_name == 'CompletionPoseEstPointAttN':
            base_model = CompletionPoseEstPointAttN(config)
            base_model = torch.nn.DataParallel(base_model).cuda()
        elif config.model_name == 'PoseEstPointAttN':
            base_model = PoseEstPointAttN(config)
            base_model = torch.nn.DataParallel(base_model).cuda()
        else:
            raise NotImplementedError(f'Train phase does not support {config.model_name}')
        
        print_log(f"############### Loading Optimizer and Scheduler #################", self.logger, color='blue')
        args.text = True
        _, _ = b.resume_model(base_model, args, config, self.logger, )

        # b.partly_resume_model(base_model, args, self.logger)

        # base_model.module.N_POINTS = 16384
        base_model.eval()
        module.set_store_attn_weights(True)

        self.model = base_model
        self.collector = module.collector


    def process_sample(self, idx_wanted):
        print_log(f"Processing sample {idx_wanted}", self.logger, color='yellow')
        dataset_name = self.config.dataset.val._base_.NAME
        if dataset_name != 'ScaphoidDataset':
            raise NotImplementedError(f'Train phase do not support {dataset_name}')
            

        self.collector.clear()
        with torch.no_grad():
            for idx, data in enumerate(self.active_dataloader):
                if idx != idx_wanted:
                    continue
                

                transformed_data, approach_dependent_gts, ind, transform_paras = data
                transform_with = self.config.dataset.transform_with

                try:
                    scale = transform_paras['transform_full'][2]['rescale'][2]
                    self.scale = scale
                except:
                    scale = transform_paras['transform_full'][0]['rescale'][2]
                    self.scale = scale

                self.transform_paras = transform_paras

                net_input_volar = transformed_data[0].cuda()
                net_input_dorsal = transformed_data[1].cuda()
                full_pcd = transformed_data[2].cuda()


                if self.config.model_name == 'ScaphoidAdaPoinTr':

                    sparse_pc, dense1 = self.model(net_input_volar, net_input_dorsal, use=transform_with)
                    seeds = sparse_pc
                    in_out = self.collector.in_out
                    
                    in_out.update({
                        'volar': net_input_volar.transpose(2, 1).contiguous(),
                        'dorsal': net_input_dorsal.transpose(2, 1).contiguous(),
                        'sparse': sparse_pc.transpose(2, 1).contiguous(),
                        'points': dense1.transpose(2, 1).contiguous(),
                    })
                    
                
                elif self.config.model_name == 'ScaphoidPointAttN':
                    dense1, dense, sparse, seeds = self.model(net_input_volar.transpose(2, 1).contiguous(), 
                                                              net_input_dorsal.transpose(2, 1).contiguous())
                    in_out = self.collector.in_out
                    
                elif self.config.model_name == 'CompletionPoseEstPointAttN' or self.config.model_name == 'PoseEstPointAttN':
                    gt_RTS_mat_v = transform_paras['volar_RTS'].cuda()
                    in_out = self.model(net_input_volar.transpose(2, 1).contiguous(), 
                                        net_input_dorsal.transpose(2, 1).contiguous(), gt_RTS_mat_v[:, 3].contiguous())
                    self.collector.add_in_out(in_out)

                    if self.config.model_name == 'CompletionPoseEstPointAttN':
                        in_out['input_concat'] = torch.cat([in_out['volar_aligned'], in_out['dorsal_aligned']], dim=2)
                        if "GT_augment_RTS" in in_out:
                            pose_gt_full = approach_dependent_gts[2].cuda()
                            pose_gt_full = t.apply_RTS_transformation(pose_gt_full, in_out['GT_augment_RTS'])
                            full_pcd = pose_gt_full
                        in_out['input_concat-gt'] = None

                    elif self.config.model_name == 'PoseEstPointAttN':
                        pred_RTS_mat = in_out['6d_pose']
                        scale_params = pred_RTS_mat[:, 3, 0], pred_RTS_mat[:, 3, 1], pred_RTS_mat[:, 3, 2]

                        if transform_with == 'volar':
                            aligned_volar = t.apply_reverse_RTS_transformation(net_input_volar, pred_RTS_mat)

                            in_out.update({
                                'volar_aligned': aligned_volar.transpose(2, 1).contiguous(),
                                'to_align': t.descale_point_cloud(net_input_volar.transpose(2, 1).contiguous(), scale_params).contiguous(),
                                'to_align-gt': None,        # enable comparison between to_align and gt
                                'volar_aligned-gt': None,   # enable comparison between volar_aligned and gt
                                'gt_pose': approach_dependent_gts[2].transpose(2, 1).contiguous()
                            })

                        elif transform_with == 'dorsal':
                            aligned_dorsal = t.apply_reverse_RTS_transformation(net_input_dorsal, pred_RTS_mat)
                            # partly_aligned_dorsal = t.apply_S_of_RTS_transformation(net_input_dorsal, pred_RTS_mat, reverse=True)
                            # partly_aligned_dorsal = t.apply_R_of_RTS_transformation(partly_aligned_dorsal, pred_RTS_mat, reverse=True)
                            print_log(f"Aligned dorsal shape: {aligned_dorsal.shape}", self.logger, color='blue')

                            in_out.update({
                                'dorsal_aligned': aligned_dorsal.transpose(2, 1).contiguous(),
                                'to_align': t.descale_point_cloud(net_input_dorsal.transpose(2, 1).contiguous(), scale_params).contiguous(),
                                'to_align-gt': None,        # enable comparison between to_align and gt
                                'dorsal_aligned-gt': None,  # enable comparison between dorsal_aligned and gt
                                'gt_pose': approach_dependent_gts[2].transpose(2, 1).contiguous()
                            })

                else:
                    raise NotImplementedError(f'Train phase does not support {self.config.model_name}')
                
                in_out.update({
                    'gt': full_pcd.transpose(2, 1).contiguous(),
                    'points-gt': None,  # enable comparison between points and gt
                })
                
                print_log(f"{full_pcd.shape}", self.logger, color='blue')
                
                break

        print_log(f"Finished processing sample {idx_wanted}", self.logger, color='blue')

    def reverse_transforms(self, pcd, pcd_name):
        if pcd_name == 'volar':
            RTS = self.transform_paras['volar_RTS']
        elif pcd_name == 'dorsal':
            RTS = self.transform_paras['dorsal_RTS']
        else:
            print(self.active_dataset.transform_with)
            try:
                RTS = self.transform_paras[f'{self.active_dataset.transform_with}_RTS']
            except KeyError:
                 RTS = self.transform_paras[f'volar_RTS']

        pcd_reversed = t.apply_reverse_RTS_transformation(pcd, RTS.cuda())
        pcd_reversed *= self.scale.cuda()
        return pcd_reversed


    def set_sample_id(self, sample_id):
        """
        Set the sample ID to process.
        """
        if sample_id < 0 or sample_id >= len(self.active_dataloader):
            raise ValueError(f"Sample ID {sample_id} is out of range. Valid range is 0 to \
                             {len(self.active_dataloader) - 1}.")
        
        self.idx = sample_id
        self.process_sample(self.idx)

    def get_sample_id(self):
        """
        Get the current sample ID.
        """
        return self.idx
    
    def __len__(self):
        """
        Get the number of samples in the validation dataset.
        """
        return len(self.active_dataloader)

    def get_valid_sample_id(self, new_sample_id):
        new_sample_id = max(0, min(new_sample_id, len(self.active_dataloader) - 1))
        return new_sample_id
