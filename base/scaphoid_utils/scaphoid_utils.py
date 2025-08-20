import os

import torch
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_euler_angles(R_batch, degrees=False):
    """
    Convert a batch of rotation matrices to Euler angles.
    :param R_batch: torch.Tensor of shape [B, 3, 3]
    :param degrees: whether to return angles in degrees
    :return: torch.Tensor of shape [B, 3] (Euler angles)
    """
    R_np = R_batch.detach().cpu().numpy()  # Convert to numpy for scipy
    r = R.from_matrix(R_np)
    eulers = r.as_euler('xyz', degrees=degrees)  # You can choose 'xyz', 'zyx', etc.
    return torch.from_numpy(eulers).to(R_batch.device)

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' %
              args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)