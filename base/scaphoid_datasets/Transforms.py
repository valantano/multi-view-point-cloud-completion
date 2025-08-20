import numpy as np
import os, sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA
import open3d as o3d
from abc import ABC, abstractmethod
import transforms3d

from base.scaphoid_utils.logger import print_log


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [GeneralDemeaning, GeneralRescale, GeneralRandomRotation, GeneralRandomMirror,
                                       CoupledDemeaning, DecoupledDemeaning, 
                                       StaticCoupledDemeaning, StaticDecoupledDemeaning, 
                                       CoupledRescale, DecoupledRescale, StaticDecoupledRescale, 
                                       CoupledRandomRotation, DecoupledRandomRotation, 
                                       StaticCoupledRandomRotation, StaticDecoupledRandomRotation, 
                                       FarthestPointSampling, NormalizeObjectPosePCA]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        data[k] = transform(v)
        return data

# class PCARotation(AbstractTransform):

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class FarthestPointSampling(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']
        input_keys = parameters['input_keys']
        self.source_key = input_keys['source']

    def __call__(self, data):
        raise NotImplementedError # Not adapted to volar and dorsal
        """
        Farthest point subsampling
        :param points: array of points (n, 3)
        :param n_points: target number of points
        :return: subsampled points
        """
        source = data[self.source_key]
        complete = o3d.geometry.PointCloud()
        complete.points = o3d.utility.Vector3dVector(source)
        subsampled = complete.farthest_point_down_sample(num_samples=self.n_points)
        source = np.asarray(subsampled.points)

        if source.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - source.shape[0], 3))
            source = np.concatenate([source, zeros])

        data[self.source_key] = source
        return data


class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

    
# class Demean:

#     def __call__(self, data):

#         if transform_with == dorsal:
class AbstractGeneralTransform(ABC):
    """
    Use sources to calculate transform parameters and apply transform to sources and targets.
    """

    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.source_keys = input_keys['sources']
        self.target_keys = input_keys['targets']

        # self.transform_with = parameters['transform_with']
        # if self.transform_with is None:
        #     raise ValueError("transform_with must be either 'volar' or 'dorsal'")

    def __call__(self, data):
        # Do debugging here.
        # print(type(self))
        # print(type(data), data)
        # print(self.source_key_volar, self.source_key_dorsal, self.target_key, self.transform_with)
        return self._transform(data)
    
    @abstractmethod
    def _transform(self, data):
        pass

class GeneralDemeaning(AbstractGeneralTransform):
    """
    Use sources to calculate mean and demean targets.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Demean targets with regard to source (corresponding to real application)
        """
        source_pcds = [data[key] for key in self.source_keys]

        means = [np.mean(pcd, axis=0, keepdims=True) for pcd in source_pcds]
        n_points = [pcd.shape[0] for pcd in source_pcds]
        combined_mean = np.sum([n_p * mean for n_p, mean in zip(n_points, means)], axis=0) / np.sum(n_points)

        for key in self.target_keys:
            data[key] = data[key] - combined_mean
            data[f"transform_{key}"].append({'demean': combined_mean})

        return data
    
class GeneralRandomRotation(AbstractGeneralTransform):
    """
    Use sources to calculate random rotation and apply it to sources and targets.
    """

    def __init__(self, parameters):
        super().__init__(parameters)


    def _transform(self, data):
        """
        Random rotation with magnitude rot_mag
        """
        source_pcds = [data[key] for key in self.source_keys]

        rand_rot = Rotation.from_matrix(special_ortho_group.rvs(3))
        axis_angle = Rotation.as_rotvec(rand_rot)
        # axis_angle *= 360 / 180.0
        # rand_rot = Rotation.from_rotvec(axis_angle)


        for key in self.target_keys:
            data[key] = rand_rot.apply(data[key])
            data[f"transform_{key}"].append({'rotation': axis_angle})

        return data
    
class GeneralRandomMirror(AbstractGeneralTransform):
    """
    Use sources to calculate random mirror and apply it to sources and targets.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Random mirror with magnitude rot_mag
        """
        source_pcds = [data[key] for key in self.source_keys]

        rnd_value = np.random.uniform(0, 1)
        for key in self.target_keys:
            data[key] = RandomMirrorPoints(parameters=None)(data[key], rnd_value)
            data[f"transform_{key}"].append({'mirror': rnd_value})

        return data
    
class RandomMirrorPoints(object):
    """
    Implementation from AdaPoinTr https://github.com/yuxumin/PoinTr/blob/master/datasets/data_transforms.py
    """
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        # Identity matrix (no transformation)
        trfm_mat = np.eye(3)

        # Reflection matrix for flipping along X-axis
        flip_x = np.diag([-1, 1, 1])

        # Reflection matrix for flipping along Z-axis
        flip_z = np.diag([1, 1, -1])

        # Apply reflection(s) based on rnd_value
        if rnd_value <= 0.25:
            trfm_mat = flip_x @ flip_z
        elif rnd_value <= 0.5:
            trfm_mat = flip_x
        elif rnd_value <= 0.75:
            trfm_mat = flip_z
        # else: trfm_mat remains identity (no flip)

        # Apply transformation to the point cloud
        ptcloud[:, :3] = ptcloud[:, :3] @ trfm_mat.T
        return ptcloud

    # def __call__(self, ptcloud, rnd_value):
    #     trfm_mat = transforms3d.zooms.zfdir2mat(1)
    #     trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
    #     trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
    #     if rnd_value <= 0.25:
    #         trfm_mat = np.dot(trfm_mat_x, trfm_mat)
    #         trfm_mat = np.dot(trfm_mat_z, trfm_mat)
    #     elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
    #         trfm_mat = np.dot(trfm_mat_x, trfm_mat)
    #     elif rnd_value > 0.5 and rnd_value <= 0.75:
    #         trfm_mat = np.dot(trfm_mat_z, trfm_mat)
    #     print_log(f"Random Mirror: {rnd_value} | Transform Matrix: {trfm_mat}", logger=None)

    #     ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
    #     return ptcloud
    
class GeneralRescale(AbstractGeneralTransform):
    """
    Use sources to calculate rescaling factors and apply them to targets.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):   # TODO: in real world no target given. How to rescale then?
        """
        Rescaling targets with regard to demeaned source (corresponding to real application)
        """
        source_pcds = [data[key] for key in self.source_keys]

        range_min = -0.5
        range_max = 0.5

        p_min = min([np.amin(pcd) for pcd in source_pcds])
        p_max = max([np.amax(pcd) for pcd in source_pcds])

        scale = (range_max - range_min) / (p_max - p_min)


        for key in self.target_keys:
            data[key] = data[key] * scale + range_min - (p_min * scale)
            data[f"transform_{key}"].append({'rescale': [p_min, range_min, scale]})

        return data



class AbstractTransform(ABC):

    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.source_key_volar = input_keys['source_volar']
        self.source_key_dorsal = input_keys['source_dorsal']
        self.target_key = input_keys['target']
        self.transform_with = parameters['transform_with']
        if self.transform_with is None:
            raise ValueError("transform_with must be either 'volar' or 'dorsal'")

    
    def __call__(self, data):
        # print(type(self))
        # print(type(data), data)
        # print(self.source_key_volar, self.source_key_dorsal, self.target_key, self.transform_with)
        return self._transform(data)
    
    @abstractmethod
    def _transform(self, data):
        pass

class CoupledDemeaning(AbstractTransform):
    """
    Demean volar and dorsal point clouds together.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Demean source and target with regard to source (corresponding to real application)
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        volar_mean = np.mean(source_volar, axis=0, keepdims=True)
        dorsal_mean = np.mean(source_dorsal, axis=0, keepdims=True)
        n_volar, n_dorsal = source_volar.shape[0], source_dorsal.shape[0]
        combined_mean = (n_volar * volar_mean + n_dorsal * dorsal_mean) / (n_volar + n_dorsal)
        
        data[self.source_key_volar] = source_volar - combined_mean
        data[self.source_key_dorsal] = source_dorsal - combined_mean
        data[self.target_key] = target - combined_mean

        data[f"transform_{self.source_key_volar}"].append({'demean': combined_mean})
        data[f"transform_{self.source_key_dorsal}"].append({'demean': combined_mean})
        data[f"transform_{self.target_key}"].append({'demean': combined_mean})
        return data
    
class StaticCoupledDemeaning(AbstractTransform):
    """
    Demean volar and dorsal point clouds together.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Demean source and target with regard to source (corresponding to real application)
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        volar_mean = np.mean(source_volar, axis=0, keepdims=True)
        dorsal_mean = np.mean(source_dorsal, axis=0, keepdims=True)
        n_volar, n_dorsal = source_volar.shape[0], source_dorsal.shape[0]
        combined_mean = (n_volar * volar_mean + n_dorsal * dorsal_mean) / (n_volar + n_dorsal)
        
        data[self.source_key_volar] = source_volar - combined_mean
        data[self.source_key_dorsal] = source_dorsal - combined_mean
        data[self.target_key] = target

        data[f"transform_{self.source_key_volar}"].append({'demean': combined_mean})
        data[f"transform_{self.source_key_dorsal}"].append({'demean': combined_mean})
        data[f"transform_{self.target_key}"].append({})
        return data
        
       
class DecoupledDemeaning(AbstractTransform):
    """
    Demean volar and dorsal point clouds independently.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Demean source and target with regard to source (corresponding to real application)
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        volar_mean = np.mean(source_volar, axis=0, keepdims=True)
        dorsal_mean = np.mean(source_dorsal, axis=0, keepdims=True)

        data[self.source_key_volar] = source_volar - volar_mean
        data[self.source_key_dorsal] = source_dorsal - dorsal_mean

        if type(target) is list:
            data[self.target_key] = [t - volar_mean if self.transform_with == 'volar' else t - dorsal_mean for t in target]
        else:
            data[self.target_key] = target - volar_mean if self.transform_with == 'volar' else target - dorsal_mean

        data[f"transform_{self.source_key_volar}"].append({'demean': volar_mean})
        data[f"transform_{self.source_key_dorsal}"].append({'demean': dorsal_mean})
        data[f"transform_{self.target_key}"].append({'demean': volar_mean if self.transform_with == 'volar' else dorsal_mean})
        return data
    
class StaticDecoupledDemeaning(AbstractTransform):
    """
    Demean volar and dorsal point clouds independently.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Demean source and target with regard to source (corresponding to real application)
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        volar_mean = np.mean(source_volar, axis=0, keepdims=True)
        dorsal_mean = np.mean(source_dorsal, axis=0, keepdims=True)
        
        data[self.source_key_volar] = source_volar - volar_mean
        data[self.source_key_dorsal] = source_dorsal - dorsal_mean
        data[self.target_key] = target

        data[f"transform_{self.source_key_volar}"].append({'demean': volar_mean})
        data[f"transform_{self.source_key_dorsal}"].append({'demean': dorsal_mean})
        data[f"transform_{self.target_key}"].append({})
        return data


class CoupledRandomRotation(AbstractTransform):
    """
    Rotate volar and dorsal point clouds together.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Random rotation with magnitude rot_mag
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        rand_rot = Rotation.from_matrix(special_ortho_group.rvs(3))
        axis_angle = Rotation.as_rotvec(rand_rot)
        # axis_angle *= 360 / 180.0
        # rand_rot = Rotation.from_rotvec(axis_angle)

        data[self.source_key_volar] = rand_rot.apply(source_volar)
        data[self.source_key_dorsal] = rand_rot.apply(source_dorsal)
        data[self.target_key] = rand_rot.apply(target)

        data[f"transform_{self.source_key_volar}"].append({'rotation': axis_angle})
        data[f"transform_{self.source_key_dorsal}"].append({'rotation': axis_angle})
        data[f"transform_{self.target_key}"].append({'rotation': axis_angle})
        return data
    
class StaticCoupledRandomRotation(AbstractTransform):
    """
    Rotate volar and dorsal point clouds together.
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Random rotation with magnitude rot_mag
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        rand_rot = Rotation.from_matrix(special_ortho_group.rvs(3))
        axis_angle = Rotation.as_rotvec(rand_rot)
        # axis_angle *= 360 / 180.0
        # rand_rot = Rotation.from_rotvec(axis_angle)

        data[self.source_key_volar] = rand_rot.apply(source_volar)
        data[self.source_key_dorsal] = rand_rot.apply(source_dorsal)
        data[self.target_key] = target

        data[f"transform_{self.source_key_volar}"].append({'rotation': axis_angle})
        data[f"transform_{self.source_key_dorsal}"].append({'rotation': axis_angle})
        data[f"transform_{self.target_key}"].append({})
        return data
    

class DecoupledRandomRotation(AbstractTransform):
    """
    Rotate volar and dorsal point clouds independently.
    """

    def __init__(self, parameters):
        super().__init__(parameters)


    def _transform(self, data):
        """
        Random rotation with magnitude rot_mag
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        rand_rot_v = Rotation.from_matrix(special_ortho_group.rvs(3))
        rand_rot_d = Rotation.from_matrix(special_ortho_group.rvs(3))
        axis_angle_v, axis_angle_d = Rotation.as_rotvec(rand_rot_v), Rotation.as_rotvec(rand_rot_d)
        # axis_angle_v *= 360 / 180.0
        # axis_angle_d *= 360 / 180.0
        # rand_rot_v, rand_rot_d = Rotation.from_rotvec(axis_angle_v), Rotation.from_rotvec(axis_angle_d)
        
        data[self.source_key_volar] = rand_rot_v.apply(source_volar)
        data[self.source_key_dorsal] = rand_rot_d.apply(source_dorsal)
        if self.transform_with == 'volar':
            data[self.target_key] = rand_rot_v.apply(target)
            target_transform = axis_angle_v
        else:
            data[self.target_key] = rand_rot_d.apply(target)
            target_transform = axis_angle_d

        data[f"transform_{self.source_key_volar}"].append({'rotation': axis_angle_v})
        data[f"transform_{self.source_key_dorsal}"].append({'rotation': axis_angle_d})
        data[f"transform_{self.target_key}"].append({'rotation': target_transform})
        return data
    
class StaticDecoupledRandomRotation(AbstractTransform):
    """
    Rotate volar and dorsal point clouds independently, but dont rotate the target.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):
        """
        Random rotation with magnitude rot_mag
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        rand_rot_v = Rotation.from_matrix(special_ortho_group.rvs(3))
        rand_rot_d = Rotation.from_matrix(special_ortho_group.rvs(3))
        axis_angle_v, axis_angle_d = Rotation.as_rotvec(rand_rot_v), Rotation.as_rotvec(rand_rot_d)
        # axis_angle_v *= 360 / 180.0
        # axis_angle_d *= 360 / 180.0
        # rand_rot_v, rand_rot_d = Rotation.from_rotvec(axis_angle_v), Rotation.from_rotvec(axis_angle_d)
        
        data[self.source_key_volar] = rand_rot_v.apply(source_volar)
        data[self.source_key_dorsal] = rand_rot_d.apply(source_dorsal)
        data[self.target_key] = target

        data[f"transform_{self.source_key_volar}"].append({'rotation': axis_angle_v})
        data[f"transform_{self.source_key_dorsal}"].append({'rotation': axis_angle_d})
        data[f"transform_{self.target_key}"].append({})
        return data



class NormalizeObjectPosePCA(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.source_key = input_keys['source']
        self.target_key = input_keys['target']

    def _transform(self, data):
        """
        Normalize object pose along principal axes of source (corresponding to real application)
        """
        raise NotImplementedError # not adapted to volar and dorsal
        source = data[self.source_key]
        target = data[self.target_key]

        source_mean = np.mean(source, axis=0, keepdims=True)
        source -= source_mean
        target -= source_mean

        # align principle axes
        pca = PCA(n_components=3, svd_solver='full')
        pca.fit(source)
        source = pca.transform(source)
        target = pca.transform(target)

        data[self.source_key] = source
        data[self.target_key] = target
        return data

class CoupledRescale(AbstractTransform):
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):   # TODO: in real world no target given. How to rescale then?
        """
        Rescaling with regard to demeaned source (corresponding to real application)
        """
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        range_min = -0.5
        range_max = 0.5

        p_min = min(np.amin(source_volar), np.amin(source_dorsal))
        p_max = max(np.amax(source_volar), np.amax(source_dorsal))
        scale = (range_max - range_min) / (p_max - p_min)

        # dot here simple multiplication since scale is a scalar
        data[self.source_key_volar] = source_volar * scale + range_min - (p_min * scale)
        data[self.source_key_dorsal] = source_dorsal * scale + range_min - (p_min * scale)
        data[self.target_key] = target * scale + range_min - (p_min * scale)

        data[f"transform_{self.source_key_volar}"].append({'rescale': [p_min, range_min, scale]})
        data[f"transform_{self.source_key_dorsal}"].append({'rescale': [p_min, range_min, scale]})
        data[f"transform_{self.target_key}"].append({'rescale': [p_min, range_min, scale]})
        return data

class DecoupledRescale(AbstractTransform):
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):   # TODO: in real world no target given. How to rescale then?
        """
        Rescaling with regard to demeaned source (corresponding to real application)
        """
        raise Exception("DecoupledRescale should not be used since in real world CoupledRescale is possible.")
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]
        range_min = -0.5
        range_max = 0.5

        p_min_v = np.amin(source_volar)
        p_max_v = np.amax(source_volar)
        scale_v = (range_max - range_min) / (p_max_v - p_min_v)

        p_min_d = np.amin(source_dorsal)
        p_max_d = np.amax(source_dorsal)
        scale_d = (range_max - range_min) / (p_max_d - p_min_d)

        data[self.source_key_volar] = source_volar.dot(scale_v) + range_min - (p_min_v * scale_v)
        data[self.source_key_dorsal] = source_dorsal.dot(scale_d) + range_min - (p_min_d * scale_d)
        if self.transform_with == 'volar':
            data[self.target_key] = target.dot(scale_v) + range_min - (p_min_v * scale_v)
            target_transform = {'rescale': [p_min_v, range_min, scale_v]}
        else:
            data[self.target_key] = target.dot(scale_d) + range_min - (p_min_d * scale_d)
            target_transform = {'rescale': [p_min_d, range_min, scale_d]}

        data[f"transform_{self.source_key_volar}"].append({'rescale': [p_min_v, range_min, scale_v]})
        data[f"transform_{self.source_key_dorsal}"].append({'rescale': [p_min_d, range_min, scale_d]})
        data[f"transform_{self.target_key}"].append(target_transform)
        return data
    
class StaticDecoupledRescale(AbstractTransform):
    def __init__(self, parameters):
        super().__init__(parameters)

    def _transform(self, data):   # TODO: in real world no target given. How to rescale then?
        """
        Rescaling with regard to demeaned source (corresponding to real application)
        """
        raise Exception("DecoupledRescale should not be used since in real world CoupledRescale is possible.")
        source_volar = data[self.source_key_volar]
        source_dorsal = data[self.source_key_dorsal]
        target = data[self.target_key]

        range_min = -0.5
        range_max = 0.5

        p_min_v = np.amin(source_volar)
        p_max_v = np.amax(source_volar)
        scale_v = (range_max - range_min) / (p_max_v - p_min_v)

        p_min_d = np.amin(source_dorsal)
        p_max_d = np.amax(source_dorsal)
        scale_d = (range_max - range_min) / (p_max_d - p_min_d)

        data[self.source_key_volar] = source_volar.dot(scale_v) + range_min - (p_min_v * scale_v)
        data[self.source_key_dorsal] = source_dorsal.dot(scale_d) + range_min - (p_min_d * scale_d)
        data[self.target_key] = target

        data[f"transform_{self.source_key_volar}"].append({'rescale': [p_min_v, range_min, scale_v]})
        data[f"transform_{self.source_key_dorsal}"].append({'rescale': [p_min_d, range_min, scale_d]})
        data[f"transform_{self.target_key}"].append({})
        return data
    
# 'source_volar': 'partial_volar',
#                         'source_dorsal': 'partial_dorsal',
#                         'target': 'gt'
class ReverseTransforms:
    
    def __call__(self, point_clouds, transforms):
        """
        Reverse the transformations applied to the point cloud.
        :param point_clouds: point clouds to reverse [B, N, 3]
        :param transforms: list of transformations to reverse
        :return: reversed point cloud
        """
        # iterate over the transforms in reverse order
        reversed_pcs = []
        for i, point_cloud in enumerate(point_clouds):
            for transform in reversed(transforms):
                if 'demean' in transform:
                    point_cloud = self.reverse_demeaning(point_cloud, transform['demean'][i])
                elif 'rotation' in transform:
                    point_cloud = self.reverse_rotation(point_cloud, transform['rotation'][i])
                elif 'rescale' in transform:
                    p_min, range_min, scale = transform['rescale']
                    scale_vals = (p_min[i], range_min[i], scale[i])
                    point_cloud = self.reverse_rescale(point_cloud, scale_vals)
                elif len(transform.keys()) == 0:
                    pass
                else:
                    raise ValueError(f"Unknown transform: {transform}")
            reversed_pcs.append(point_cloud)


        return reversed_pcs
    
    

    def reverse_demeaning(self, to_be_reversed, mean):
        """
        Reverse demean
        :param to_be_reversed: point cloud to reverse
        :param mean: mean values to add back
        :return: reversed point cloud
        """
        if type(to_be_reversed) != type(mean):
            mean = np.array(mean.squeeze())

        return to_be_reversed + mean

    def reverse_rotation(self, to_be_reversed, axis_angle):
        """
        Reverse rotation
        :param to_be_reversed: point cloud to reverse
        :param axis_angle: rotation axis and angle
        :return: reversed point cloud
        """
        axis_angle *= -1
        if type(to_be_reversed) != type(axis_angle):
            axis_angle = np.array(axis_angle)
        rot = Rotation.from_rotvec(axis_angle)
        return rot.apply(to_be_reversed)
    
    def reverse_rescale(self, to_be_reversed, scale_vals):
        """
        Reverse rescale
        :param to_be_reversed: point cloud to reverse
        :param scale_vals: scale values containing p_min, range_min, and scale
        :return: reversed point cloud
        """
        # return to_be_reversed
        p_min, range_min, scale = scale_vals
        if type(to_be_reversed) != type(p_min):
            p_min = np.array(p_min)
            range_min = np.array(range_min)
            scale = np.array(scale)
        if scale == 0:
            raise ValueError("Scale factor cannot be zero.")
        return (to_be_reversed - range_min + (p_min * scale)) / scale