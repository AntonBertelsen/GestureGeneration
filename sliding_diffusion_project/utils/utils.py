import torch
import os
import glob
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def convert_6d_to_matrix(rot_6d_batch: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices."""

    input_dim = len(rot_6d_batch.shape)

    if input_dim == 2:
        # If input is 2D, add a batch dimension
        rot_6d_batch = rot_6d_batch.unsqueeze(0)

    batch_size = rot_6d_batch.shape[0]
    num_frames = rot_6d_batch.shape[1]
    
    # Extract columns
    col1 = rot_6d_batch[:, :, 0:3]  # Shape: (batch_size, num_frames, 3)
    col2 = rot_6d_batch[:, :, 3:6]  # Shape: (batch_size, num_frames, 3)
    
    # Normalize columns (vectorized)
    col1_norm = torch.linalg.norm(col1, axis=2, keepdims=True)
    col2_norm = torch.linalg.norm(col2, axis=2, keepdims=True)
    col1 = col1 / col1_norm
    col2 = col2 / col2_norm

    # Compute cross product for third column (vectorized)
    col3 = torch.linalg.cross(col1, col2)
    
    # Stack into rotation matrices
    matrices = torch.zeros((batch_size, num_frames, 3, 3), device=rot_6d_batch.device)
    matrices[:, :, :, 0] = col1
    matrices[:, :, :, 1] = col2
    matrices[:, :, :, 2] = col3

    if input_dim == 2:
        # If input was 2D, remove the batch dimension
        matrices = matrices.squeeze(0)
    
    return matrices

def convert_matrix_to_6d(rot_matrix_batch: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, _, _ = rot_matrix_batch.shape
    rot_6d = rot_matrix_batch.permute(0, 1, 3, 2)[:, :, :2, :].reshape(batch_size, num_frames, 6)
    return rot_6d

def get_latest_model_path(directory: str = "trained_models", return_folder = False) -> str:
    # Find the newest folder
    folders = sorted(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)
    if not folders:
        raise FileNotFoundError("No model folders found in the specified directory.")

    latest_folder = folders[-1]

    # Find the newest file in that folder
    files = sorted(glob.glob(os.path.join(latest_folder, '*.pth')), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No model files found in the latest folder.")

    if return_folder:
        return files[-1], latest_folder.split(os.path.sep)[-1]
    else:
        return files[-1]
    
def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )


def get_rest_pose(num_joints: int, device: torch.device) -> torch.Tensor:
    
    length = 3 + (num_joints-1) * 6  # 3 for root position, 6 for each joint (3D position + 3D rotation)

    # each joint after the first 3 values (root position) has 6 values (6d rotation). They should all have the same values, [1,0,0,0,1,0] repeated for each joint
    rest_pose = torch.zeros(length, device=device)
    rest_pose[0:3] = torch.tensor([0.0, 0.0, 0.0], device=device)  # Root position
    for i in range(1, num_joints):
        rest_pose[3 + (i - 1) * 6:3 + i * 6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)  # Joint position and rotation
    return rest_pose