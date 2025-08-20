import os
import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor

from base.scaphoid_datasets.Transforms import Compose
from base.scaphoid_utils.logger import print_log, Logger


STRICT_DATASET = 'StrictScaphoidDataset'
UNSTRICT_DATASET = 'UnstrictScaphoidDataset'
DEFAULT_TRANSFORMS = ['DecoupledDemeaning', 'DecoupledRandomRotation', 'CoupledRescale']  # Default augmentations from older runs
SUPPORTED_VERSIONS = [1.5, 2.0, 2.0]  # Supported dataset versions


class BaseDataset(data.Dataset):
    """
    Dataset on which PoseEstDataset, ScaphoidDataset, (and RotationDataset) are based.
    PoseEstDataset, ScaphoidDataset, and RotationDataset only reimplement the __getitem__ method.
    Contains the basic functionality to load the dataset, apply transforms, and get samples.
    """

    def __init__(self, subset: str, config: dict, transforms: list[str]=None, strict=False, transform_with='volar', 
                 debug=False, logger: Logger=None):
        """
        :param subset: subset of the dataset (train, valid, test)
        :param config: configuration dictionary containing DATA_PATH and N_POINTS
        :param transforms: list of transforms to apply 
            (like ['DecoupledDemeaning', 'DecoupledRandomRotation', 'DecoupledRescale'])
        :param strict: whether to load the strict dataset or the unstrict
        :param transform_with: with which point set the gt is transformed (like 'volar' or 'dorsal')
        :param debug: whether to enable debug mode
        :param logger: logger instance for logging
        """
        
        self.data_dir = config.DATA_PATH

        self.npoints = config.N_POINTS
        self.subset = subset     # should be either 'train', 'valid', or 'test'

        self.debug = debug
        self.logger=logger

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        dataset_type_path = STRICT_DATASET if strict else UNSTRICT_DATASET
        self.dataset_path = Path(os.path.join(BASE_DIR, '..', '..', self.data_dir, dataset_type_path)).resolve()

        file_list = self._get_h5_file_list(self.dataset_path, self.subset)

        print_log(f"{'#'*20} ScaphoidDataset {self.subset} {'#'*20}\n"
                  f"{self.dataset_path}\n"
                  f"ScaphoidDataset ({self.subset}): Found {len(file_list)} files in {self.data_dir}: {file_list}",
                  color='cyan', logger=logger)

        self.all_full, self.all_dorsal, self.all_volar, ind, versions = self.__load_h5_data(file_list, debug, logger)
        (
            self.all_volar_ind, self.all_dorsal_ind, self.all_articular_ind, self.all_distal_ind, self.all_proximal_ind,
        ) = ind


        print_log(f"ScaphoidDataset ({self.subset}): Loaded {len(self.all_full)} samples with versions: {versions}\n"
                  f"ScaphoidDataset using {transforms} with transform_with={transform_with}", 
                  logger=self.logger, color='cyan')

        assert np.all(np.isin(np.unique(versions), SUPPORTED_VERSIONS)), f"ScaphoidDataset: Unsupported version \
             detected in dataset (supported versions: {SUPPORTED_VERSIONS}): {np.unique(versions)}"


        self.transforms = self._get_transforms(self.subset, transforms, transform_with)
        self.transform_with = transform_with
        if subset == 'train':
            # pre_processing_transforms = ['GeneralRandomMirror','GeneralDemeaning']
            pre_processing_transforms = ['GeneralDemeaning']
        else:
            pre_processing_transforms = ['GeneralDemeaning']
        self.pre_processing_transforms = self._get_pre_processing_transforms(subset, pre_processing_transforms)

    def change_transforms(self, transforms: list[str]=None, transform_with=None):
        """
        Change the transforms without reloading the entire dataset.
        :param transforms: list of transforms to apply 
            (like ['DecoupledDemeaning', 'DecoupledRandomRotation', 'DecoupledRescale'])
        :param transform_with: with which point set the gt is transformed (like 'volar' or 'dorsal')
        :return: None
        """
        if transforms is None or transform_with is None:
            raise ValueError("Transforms and transform_with cannot be None")
        self.transforms = self._get_transforms(self.subset, transforms, transform_with)
        self.transform_with = transform_with

    @staticmethod
    def _get_h5_file_list(base_dir: str, subset: str) -> list[str]:
        if subset == 'train':  # TODO: fix without workaround
            subset = 'train'
        elif subset == 'test':
            subset = 'test'
        else:
            subset = 'valid'
        h5_file_list = glob.glob(os.path.join(base_dir, f'{subset}_data_*.h5'))
        return h5_file_list
    
    
            
    @staticmethod
    def __load_h5_data(h5_file_list: list[str], debug: bool, logger
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, 
                            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                            np.ndarray]:
        """
        Load h5 point dataset.
        :param h5_file_list: list of paths to h5 files
        :return all_full, all_partial: ndarray of point sets [num_sets, 1024, 3]
        """
        all_full, all_partial_dorsal, all_partial_volar = [], [], []
        volar_inds, dorsal_inds, articular_inds = [], [], []
        distal_inds, proximal_inds = [], []
        versions = []

        for i, h5_file in enumerate(h5_file_list, start=1):
            with h5py.File(h5_file, "r") as f:
                print_log(f"Loading {i}/{len(h5_file_list)} h5 files with keys: {list(f.keys())}", logger=logger)

                # Core point sets
                full = f["full"][:].astype(np.float32)
                partial_dorsal = f["partial_dorsal"][:].astype(np.float32)
                partial_volar = f["partial_volar"][:].astype(np.float32)

                # Indices
                volar_ind = f["volar_points_ind"][:].astype(int)
                dorsal_ind = f["dorsal_points_ind"][:].astype(int)
                articular_ind = f["articular_points_ind"][:].astype(int)
                distal_ind = f["distal_points_ind"][:].astype(int)
                proximal_ind = f["proximal_points_ind"][:].astype(int)

                # Version (default = 1.0 if missing)
                try:
                    version = f["version"][:].astype(np.float32)
                except KeyError:
                    version = np.array(1.0, dtype=np.float32)

            # Append
            all_full.append(full)
            all_partial_dorsal.append(partial_dorsal)
            all_partial_volar.append(partial_volar)

            volar_inds.append(volar_ind)
            dorsal_inds.append(dorsal_ind)
            articular_inds.append(articular_ind)
            distal_inds.append(distal_ind)
            proximal_inds.append(proximal_ind)
            versions.append(version)

            # Debug shortcut
            if debug:
                print_log(f"Debug mode: Loaded {len(all_full)} sample(s)", logger=logger, color="yellow")
                break

        # Concatenate lists into single arrays
        all_full = np.concatenate(all_full, axis=0)
        all_partial_dorsal = np.concatenate(all_partial_dorsal, axis=0)
        all_partial_volar = np.concatenate(all_partial_volar, axis=0)

        volar_inds = np.concatenate(volar_inds, axis=0)
        dorsal_inds = np.concatenate(dorsal_inds, axis=0)
        articular_inds = np.concatenate(articular_inds, axis=0)
        distal_inds = np.concatenate(distal_inds, axis=0)
        proximal_inds = np.concatenate(proximal_inds, axis=0)

        versions = np.array(versions).flatten()

        print_log(
            f"Loaded {len(all_full)} samples, ({len(all_partial_volar)} volar, {len(all_partial_dorsal)} dorsal)", 
            logger=logger, 
            color="cyan")

        return (
            all_full,
            all_partial_dorsal,
            all_partial_volar,
            (volar_inds, dorsal_inds, articular_inds, distal_inds, proximal_inds),
            versions,
        )
    

    def get_indices_sample(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Helper function to get indices of the labels for sample at index idx.
        :param idx: sample index
        :return: volar_ind, dorsal_ind, articular_ind, distal_ind, proximal_ind
        """
        volar_ind = torch.tensor(self.all_volar_ind[idx], dtype=torch.int64)
        dorsal_ind = torch.tensor(self.all_dorsal_ind[idx], dtype=torch.int64)
        articular_ind = torch.tensor(self.all_articular_ind[idx], dtype=torch.int64)
        distal_ind = torch.tensor(self.all_distal_ind[idx], dtype=torch.int64)
        proximal_ind = torch.tensor(self.all_proximal_ind[idx], dtype=torch.int64)

        return volar_ind, dorsal_ind, articular_ind, distal_ind, proximal_ind
    
    
    @staticmethod
    def get_completion_data(sample_partial_volar: np.ndarray, sample_partial_dorsal: np.ndarray, 
                            sample_full: np.ndarray, pre_processing_transforms: Compose, transforms: Compose
                            ) -> tuple[tuple[Tensor, Tensor, Tensor], dict]:
        """
        Get the data for completion task.
        :param sample_full: full point cloud
        :param sample_partial_volar: partial volar point cloud
        :param sample_partial_dorsal: partial dorsal point cloud
        :param transforms: transforms to apply
        :return: preprocessed_data, transformed_data, transform_paras
        """
        data_pre = {
            'partial_volar': sample_partial_volar,
            'partial_dorsal': sample_partial_dorsal,
            'full': sample_full,
            'transform_partial_volar': [],
            'transform_partial_dorsal': [],
            'transform_full': []
        }

        data_pre = pre_processing_transforms(data_pre)

        data = {
            "partial_volar": data_pre['partial_volar'].copy(),
            "partial_dorsal": data_pre['partial_dorsal'].copy(),
            "full": data_pre['full'].copy(),
            "transform_partial_volar": [],
            "transform_partial_dorsal": [],
            "transform_full": []
        }

        data = transforms(data)

        preprocessed_data = (data_pre['partial_volar'], data_pre['partial_dorsal'], data_pre['full'])
        transformed_data = (data['partial_volar'], data['partial_dorsal'], data['full'])
        transform_keys = ['transform_partial_volar', 'transform_partial_dorsal', 'transform_full']
        transform_paras = {key: data[key] for key in transform_keys}

        return preprocessed_data, transformed_data, transform_paras
    

    def get_pcd_sample(self, idx: int) -> np.ndarray:
        return self.all_volar[idx, :, :].copy(), self.all_dorsal[idx, :, :].copy(), self.all_full[idx, :, :].copy()


    def __getitem__(self, idx) -> tuple[
        tuple[Tensor, Tensor, Tensor], int, 
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor], dict]:
        """
        Get the item at index idx.
        :param idx: index of the sample
        :return: transformed data (partial_volar, partial_dorsal, full), placeholder, 
            indices of the labels (volar_ind, dorsal_ind, articular_ind,
            distal_ind, proximal_ind), and transformation parameters
        """
        sample_p_volar, sample_p_dorsal, sample_full = self.get_pcd_sample(idx)
        inds = self.get_indices_sample(idx)

        _, (sample_p_volar, sample_p_dorsal, sample_full), transform_paras = BaseDataset.get_completion_data(
            sample_p_volar, sample_p_dorsal, sample_full, self.pre_processing_transforms, self.transforms)
        transformed_data = (sample_p_volar, sample_p_dorsal, sample_full)

        return transformed_data, 0, inds, transform_paras

    @staticmethod
    def _get_transforms(subset: str, transform_list: list=None, transform_with='volar') -> Compose:
        """
        Get the transforms to apply to the dataset.
        Usually, the partial point clouds are independenlty demeaned, randomly rotated and dependently rescaled.
        The full point cloud is demeaned, rotated and rescaled based on the partial point cloud given by transform_with.
        :param subset: subset of the dataset (train, valid, test)
        :param transform_list: list of transforms to apply 
            (like ['DecoupledDemeaning', 'DecoupledRandomRotation', 'CoupledRescale'])
        :param transform_with: with which point set the gt is transformed (like 'volar' or 'dorsal')
        :return: Compose object with the transforms
        """
        if transform_list is None:
            transform_list = DEFAULT_TRANSFORMS
        if transform_with is None:
            transform_with = 'volar'

        transforms = [
            {
                'callback': f'{t}',
                'parameters': {
                    'input_keys': {
                        'source_volar': 'partial_volar',
                        'source_dorsal': 'partial_dorsal',
                        'target': 'full'
                    },
                    'transform_with': transform_with
                },
                'objects': ['partial_volar', 'partial_dorsal', 'full']
            } 
            for t in transform_list
        ]

        transforms.append({
            'callback': 'ToTensor',
            'objects': ['partial_volar', 'partial_dorsal', 'full']
        })

        return Compose(transforms)
    
    @staticmethod
    def _get_pre_processing_transforms(subset: str, transform_list: list=None):
        """
        Get the pre-processing transforms to apply to the dataset.
        The full and partial point clouds should be demeaned and rescaled based on the full point cloud to ensure 
        consistency in the canonical space.
        This is used for pre-processing before the main transforms.
        :param subset: subset of the dataset (train, valid, test)
        :param transform_list: list of transforms to apply (like ['GeneralDemeaning', 'GeneralRescale'])
        :return: Compose object with the transforms
        """
        if transform_list is None:
            transform_list = ['GeneralDemeaning', 'GeneralRescale']

        transforms = [
            {
                'callback': f'{t}',
                'parameters': {
                    'input_keys': {
                        'sources': ['full'],
                        'targets': ['partial_dorsal', 'partial_volar', 'full'],
                    }
                },
                'objects': ['partial_volar', 'partial_dorsal', 'full']
            } 
            for t in transform_list
        ]

        return Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.all_full)
