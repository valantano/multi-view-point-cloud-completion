import torch
from torch import Tensor
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group

from base.scaphoid_utils.logger import print_log

def apply_affine_transformation(pcd, affine_matrix):
    """
    Apply affine transformation to a point cloud
    :param pcd: point cloud to be transformed with shape (B, N, 3)
    :param affine_matrix: affine matrix to be applied with shape (B, 4, 4)
    :return: transformed point cloud with shape (B, N, 3)
    """
    if isinstance(pcd, torch.Tensor):
        B, N, C = pcd.shape
        device = pcd.device
        affine_matrix = affine_matrix
        pcd = torch.cat((pcd, torch.ones((B, N, 1)).to(device)), axis=2)        # add homogeneous coordinate
        pcd = torch.bmm(pcd, affine_matrix.transpose(1,2))        # batch matrix multiplication
        pcd = pcd[:, :, :3]        # remove homogeneous coordinate

    # elif isinstance(pcd, np.ndarray):
    #     affine_matrix = affine_matrix
    #     pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)        # add homogeneous coordinate
    #     pcd = pcd @ affine_matrix.T
    #     pcd = pcd[:, :3]        # remove homogeneous coordinate
    else:
        raise TypeError("pcd must be either torch.Tensor or np.ndarray")

    return pcd

def get_reverse_affine_matrix(affine_matrix):
    """
    Get reverse affine matrix
    :param affine_matrix: affine matrix to be reversed with shape (4, 4)
    :return: reverse affine matrix with shape (4, 4)
    """
    if isinstance(affine_matrix, torch.Tensor):
        # inverse affine transformation
        reverse_affine_matrix = torch.linalg.inv(affine_matrix)
    elif isinstance(affine_matrix, np.ndarray):
        # inverse affine transformation
        reverse_affine_matrix = np.linalg.inv(affine_matrix)
    else:
        raise TypeError("affine_matrix must be either torch.Tensor or np.ndarray")

    return reverse_affine_matrix

def get_affine_matrix_without_scaling(rescale, rotation, translation):
    if all(isinstance(x, torch.Tensor) for x in [rotation, translation]):

        R = torch.tensor(Rotation.from_rotvec(rotation).as_matrix())
        rescale = torch.tensor(rescale)

        S = torch.eye(3)
        RS = R.type(torch.float32) @ S.type(torch.float32)
        T = - RS @ translation
        affine_matrix = torch.eye(4)
        affine_matrix[:3, :3] = RS
        affine_matrix[:3, 3] = T
        return affine_matrix

    elif all(isinstance(x, np.ndarray) for x in [rotation, translation]):
        R = Rotation.from_rotvec(rotation).as_matrix()

        S = np.eye(3)
        RS = R @ S
        T = - RS @ translation
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = RS
        affine_matrix[:3, 3] = T
        return affine_matrix
    
    else:
        raise TypeError("rescale, rotation and translation must be either torch.Tensor or np.ndarray")
    

def get_affine_matrix(rescale, rotation, translation):

    if all(isinstance(x, torch.Tensor) for x in [rotation, translation]):

        R = torch.tensor(Rotation.from_rotvec(rotation).as_matrix())
        rescale = torch.tensor(rescale)
        p_min, range_min, scale = rescale[0], rescale[1], rescale[2]

        S = torch.eye(3) * scale
        RS = R.type(torch.float32) @ S.type(torch.float32)
        T = range_min - (p_min * scale) - RS @ translation
        affine_matrix = torch.eye(4)
        affine_matrix[:3, :3] = RS
        affine_matrix[:3, 3] = T
        return affine_matrix

    elif all(isinstance(x, np.ndarray) for x in [rotation, translation]):
        R = Rotation.from_rotvec(rotation).as_matrix()
        p_min, range_min, scale = rescale[0], rescale[1], rescale[2]

        S = np.eye(3) * scale
        RS = R @ S
        T = range_min - (p_min * scale) - RS @ translation
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = RS
        affine_matrix[:3, 3] = T
        return affine_matrix
    
    else:
        raise TypeError("rescale, rotation and translation must be either torch.Tensor or np.ndarray")
    
def rescale_point_cloud(pcd: Tensor, rescale_params: Tensor) -> Tensor:
    p_min, range_min, scale = rescale_params
    p_min, range_min, scale = p_min.view(-1, 1, 1), range_min.view(-1, 1, 1), scale.view(-1, 1, 1)
    return pcd * scale + range_min - (p_min * scale)

def descale_point_cloud(pcd: Tensor, rescale_params: Tensor) -> Tensor:
    p_min, range_min, scale = rescale_params
    p_min, range_min, scale = p_min.view(-1, 1, 1), range_min.view(-1, 1, 1), scale.view(-1, 1, 1)
    return (pcd - range_min + (p_min * scale)) / scale


def create_RTS_matrix(pred_RT_mat, gt_RTS_mat):
    """
    Helper function to create a 4x4 RTS matrix from a 3x4 RT matrix which was predicted by the model and the scaling 
    component of the ground truth RTS matrix.
    :param pred_RT_mat: predicted rotation and translation matrix with shape (B, 3, 4)
    :param gt_RTS_mat: ground truth rotation, translation and scaling matrix with shape (B, 4, 4)
    :return: pred_RTS_mat: predicted rotation, translation and scaling matrix with shape (B, 4, 4)
    """
    B = pred_RT_mat.shape[0]
    pred_RTS_mat = torch.zeros((B, 4, 4), dtype=torch.float32).to(pred_RT_mat.device)
    pred_RTS_mat[:, :3, :3] = pred_RT_mat[:, :3, :3].type(torch.float32)
    pred_RTS_mat[:, :3, 3] = pred_RT_mat[:, :3, 3].type(torch.float32)
    pred_RTS_mat[:, 3] = gt_RTS_mat[:, 3]  # scale
    return pred_RTS_mat


def apply_R_of_RTS_transformation(pcd: Tensor, RTS: Tensor, reverse=False) -> Tensor:
    """
    Apply rotation to a point cloud using the rotation part of the RTS matrix
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RTS: rotation and translation matrix to be applied with shape (B, 4, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    R = RTS[:, :3, :3]  # rotation matrix
    
    if reverse:
        pcd = torch.bmm(pcd, R)
    else:
        pcd = torch.bmm(pcd, R.transpose(1, 2))  # batch matrix multiplication
    
    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd

def apply_T_of_RTS_transformation(pcd: Tensor, RTS: Tensor, reverse=False) -> Tensor:
    """
    Apply translation to a point cloud using the translation part of the RTS matrix
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RTS: rotation and translation matrix to be applied with shape (B, 4, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    T = RTS[:, :3, 3]   # translation vector
    
    if reverse:
        pcd = pcd - T.unsqueeze(1)
    else:
        pcd = pcd + T.unsqueeze(1)

    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd

def apply_S_of_RTS_transformation(pcd: Tensor, RTS: Tensor, reverse=False) -> Tensor:
    """
    Apply scaling to a point cloud using the scaling part of the RTS matrix
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RTS: rotation and translation matrix to be applied with shape (B, 4, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    p_min, range_min, scale = RTS[:, 3, 0], RTS[:, 3, 1], RTS[:, 3, 2]  # rescale parameters
    B, N, C = pcd.shape

    if reverse:
        pcd = descale_point_cloud(pcd, (p_min, range_min, scale))  # descale point cloud
    else:
        pcd = rescale_point_cloud(pcd, (p_min, range_min, scale))  # rescale point cloud
    
    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd
    
def apply_RTS_transformation(pcd: Tensor, RTS: Tensor) -> Tensor:
    """
    Apply translation, rotation and scale to a point cloud
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RT: rotation and translation matrix to be applied with shape (B, 3, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    pcd = apply_T_of_RTS_transformation(pcd, RTS)
    pcd = apply_R_of_RTS_transformation(pcd, RTS)
    pcd = apply_S_of_RTS_transformation(pcd, RTS)

    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd

def apply_reverse_RTS_transformation(pcd: Tensor, RTS: Tensor, rescale_again: bool=False) -> Tensor:
    """
    Apply reverse scale, rotation and translation to a point cloud
    :param pcd: Tensor point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RT: Tensor rotation and translation matrix to be applied with shape (B, 3, 4)
    :param rescale_again: bool, whether to apply rescaling again after reverse transformation
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    pcd = apply_S_of_RTS_transformation(pcd, RTS, reverse=True) 
    pcd = apply_R_of_RTS_transformation(pcd, RTS, reverse=True)
    pcd = apply_T_of_RTS_transformation(pcd, RTS, reverse=True)

    if rescale_again:
        pcd = apply_S_of_RTS_transformation(pcd, RTS)

    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd

def apply_RT_transformation(pcd: Tensor, RT: Tensor) -> Tensor:
    """
    Apply rotation and translation to a point cloud
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RT: rotation and translation matrix to be applied with shape (B, 3, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RT, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    RT = RT[0].unsqueeze(0)
    pcd = pcd[0].unsqueeze(0)

    print_log(f"RT: {RT}", color='yellow')


    R = RT[:, :3, :3]  # rotation matrix
    T = RT[:, :3, 3]   # translation vector
    B, N, C = pcd.shape

    print_log(f"{T}, {T.shape}", color='yellow')

    pcd = pcd + T.unsqueeze(1)
    pcd = torch.bmm(pcd, R.transpose(1, 2))  # batch matrix multiplication
    
    

    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd
    

def apply_reverse_RT_transformation(pcd: Tensor, RT: Tensor) -> Tensor:
    """
    Apply reverse rotation and translation to a point cloud
    :param pcd: Tensor point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RT: Tensor rotation and translation matrix to be applied with shape (B, 3, 4)
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RT, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True

    R = RT[:, :3, :3]  # rotation matrix
    T = RT[:, :3, 3]   # translation vector
    B, N, C = pcd.shape
    pcd = torch.bmm(pcd, R)
    pcd = pcd - T.unsqueeze(1)

    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd

def augment_pcds(pcds: list[Tensor], rdm_rot: bool=True, demean: bool=True, scale_paras: Tensor=None) -> list[Tensor]:
    """
    Augment a list of point clouds with random rotation, translation (demeaning) and scaling.
    :param pcds: list of point clouds to be augmented, each with shape (B, C, N) or (B, N, C)
    :param rdm_rot: bool, whether to apply random rotation to the point clouds
    :param demean: bool, whether to apply demeaning (translation) to the point clouds
    :param rescale: bool, whether to apply rescaling to the point clouds
    :return: list of augmented point clouds
    """
    B, C, N = pcds[0].shape

    if pcds[0].shape[1] == 3:
        mean_dim = 2
    elif pcds[0].shape[2] == 3:
        mean_dim = 1
    else:
        raise ValueError("Point cloud must have shape (B, N, 3) or (B, 3, N)")

    RTS = torch.zeros((B, 4, 4), dtype=torch.float32, device=pcds[0].device)
    if rdm_rot:
        rot_matrices = torch.tensor(special_ortho_group.rvs(dim=3, size=B), dtype=torch.float32, device=pcds[0].device)
        RTS[:, :3, :3] = rot_matrices
    else:
        RTS[:, :3, :3] = torch.eye(3, dtype=torch.float32, device=pcds[0].device).unsqueeze(0).repeat(B, 1, 1)
    
    if demean:
        mean0 = pcds[0].mean(dim=mean_dim)
        mean1 = pcds[1].mean(dim=mean_dim)
        mean_total = (mean0 + mean1) / 2
        RTS[:, :3, 3] = mean_total
        # print_log(f"demean: {RTS[:, :3, 3]}", color='yellow')
    else:
        pass

    if scale_paras is not None:
        RTS[:, 3] = scale_paras

    
    augmented_pcds = [apply_RTS_transformation(pcd, RTS).contiguous() for pcd in pcds]

    return augmented_pcds, RTS


def get_RTS(rotation, translation, rescale):
    p_min, range_min, scale = rescale

    if all(isinstance(x, torch.Tensor) for x in [rotation, translation]):

        R = torch.tensor(Rotation.from_rotvec(rotation).as_matrix())

        RTS = torch.zeros((4, 4), dtype=torch.float32)
        RTS[:3, :3] = R.type(torch.float32)
        RTS[:3, 3] = translation.type(torch.float32)
        RTS[3] = torch.tensor([p_min, range_min, scale, 1], dtype=torch.float32)

        return RTS

    elif all(isinstance(x, np.ndarray) for x in [rotation, translation]):
        R = Rotation.from_rotvec(rotation).as_matrix()

        RTS = np.zeros((4, 4), dtype=np.float32)
        RTS[:3, :3] = R
        RTS[:3, 3] = translation.astype(np.float32)
        RTS[3] = np.array([p_min, range_min, scale, 1], dtype=np.float32)

        return RTS

    else:
        raise TypeError("rescale, rotation and translation must be either torch.Tensor or np.ndarray")


def apply_RTS_rescale_transformation(pcd: Tensor, RTS: Tensor) -> Tensor:
    """
    Helper function to only apply the scaling component of the RTS matrix to a point cloud.
    Mainly used for debugging and visualization purposes. 
    :param pcd: point cloud to be transformed with shape (B, N, 3) or (B, 3, N)
    :param RT: rotation and translation matrix to be applied with shape (B, 3, 4)
    
    :return: transformed point cloud with shape (B, N, 3)
    """
    if not isinstance(RTS, torch.Tensor):
        raise TypeError("RT must be a torch.Tensor")
    
    transposed = False
    if pcd.shape[-1] != 3:
        pcd = pcd.transpose(1, 2)  # ensure pcd is of shape (B, N, 3)
        transposed = True


    p_min, range_min, scale = RTS[:, 3, 0], RTS[:, 3, 1], RTS[:, 3, 2]  # rescale parameters
    B, N, C = pcd.shape

    pcd = rescale_point_cloud(pcd, (p_min, range_min, scale))  # rescale point cloud
    
    if transposed:
        pcd = pcd.transpose(1, 2)  # revert back to original shape

    return pcd