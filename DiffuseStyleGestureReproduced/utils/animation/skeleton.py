import numpy as np
import torch
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R
from utils.utils import convert_6d_to_matrix, convert_matrix_to_6d

class Skeleton:
    def __init__(self):
        # Joint structure
        self.joints = []  # List of joints in order of appearance
        self.joint_channels = {}  # Dict mapping joint name to channel names (What channels are defined in the bvh for this joint, i.e. position, rotation)
        self.joint_position_indices = {}  # Maps joint name to position channel indices
        self.joint_rotation_indices = {}  # Maps joint name to rotation channel indices
        self.joint_offsets = {}  # Maps joint name to its offset
        self.joint_parent = {}  # Maps joint name to its parent's name
        self.end_sites = {}  # Maps joint name to end site offset
        self.target_joints = None  # Set of joints to extract, if None all joints are used
        
        # Extraction/insertion mappings
        self._pos_extraction_map = []  # [(output_idx, input_idx)] for positions
        self._rot_extraction_map = []  # [(output_start_idx, [x_idx, y_idx, z_idx])] for rotations
        self.bone_to_indices_map = {}  # Maps bone names to their indices in the output array

        # Bone categories (Used for loss weights)
        self.bone_categories = {}

        self.mean_pose = None  # Mean pose for normalization
        self.std_pose = None  # Standard deviation for normalization

        # The world position mean and std are used to normalize the world positions which are used as a regularisation term in the loss function and also used for calculating FGD.
        # I feel a little weird about normalising the world positions, since to me it seems like part of the idea of world positions is that bones that move a lot are punished more / more important.
        # But this is what is done in https://github.com/genea-workshop/genea_numerical_evaluations/tree/2022 as far as I can tell.
        self.world_pos_mean_pose = None  # Mean pose for world positions 
        self.world_pos_std_pose = None  # Standard deviation for world positions

        self.device = torch.device('cpu')  # Default device
    
    def set_device(self, device: torch.device) -> None:
        """Set the device for tensor operations."""
        self.device = device
        self.mean_pose = self.mean_pose.to(device) if self.mean_pose is not None else None
        self.std_pose = self.std_pose.to(device) if self.std_pose is not None else None
        self.world_pos_mean_pose = self.world_pos_mean_pose.to(device) if self.world_pos_mean_pose is not None else None
        self.world_pos_std_pose = self.world_pos_std_pose.to(device) if self.world_pos_std_pose is not None else None

    def set_mean_std(self, mean_pose: np.ndarray, std_pose: np.ndarray, world_pos_mean_pose: np.ndarray = None, world_Pos_mean_std: np.ndarray = None) -> None:
        """Set the mean and standard deviation for normalization."""
        self.mean_pose = torch.tensor(mean_pose, dtype=torch.float32, device=self.device)
        self.std_pose = torch.tensor(std_pose, dtype=torch.float32, device=self.device)
        if world_pos_mean_pose is not None and world_Pos_mean_std is not None:
            self.world_pos_mean_pose = torch.tensor(world_pos_mean_pose, dtype=torch.float32, device=self.device)
            self.world_pos_std_pose = torch.tensor(world_Pos_mean_std, dtype=torch.float32, device=self.device)

    def add_joint(self, joint_name: str, parent_name: Optional[str] = None) -> None:
        """Add a joint to the skeleton."""
        self.joints.append(joint_name)
        self.joint_channels[joint_name] = []
        self.joint_position_indices[joint_name] = []
        self.joint_rotation_indices[joint_name] = []
        self.joint_parent[joint_name] = parent_name
    
    def set_joint_offset(self, joint_name: str, offset: List[float]) -> None:
        """Set joint offset."""
        self.joint_offsets[joint_name] = offset
    
    def add_end_site(self, parent_name: str, offset: List[float]) -> None:
        """Add an end site to a joint."""
        end_site_name = f"EndSite_{parent_name}"
        self.end_sites[end_site_name] = offset
    
    def set_joint_channels(self, joint_name: str, channels: List[str], 
                          channel_start_idx: int) -> None:
        """Set the channels for a joint."""
        self.joint_channels[joint_name] = channels
        
        # Track position and rotation channel indices
        for j, channel in enumerate(channels):
            if 'position' in channel:
                self.joint_position_indices[joint_name].append(channel_start_idx + j)
            elif 'rotation' in channel:
                self.joint_rotation_indices[joint_name].append(channel_start_idx + j)
    
    def set_target_joints(self, target_joints: Optional[List[str]]) -> None:
        """Set the target joints for extraction."""
        self.target_joints = target_joints
        self._precompute_mappings()

    def set_bone_categories(self, bone_categories: Dict[str, str]) -> None:
        """Set the bone categories for loss weights."""
        self.bone_categories = bone_categories
    
    def construct_bone_weighting_vector(self, category_weighting) -> torch.Tensor:
        num_features = self.get_channel_count()
        bone_index_weighted_by_category_vector = torch.ones(num_features)
        bone_index_weighted_by_category_vector = bone_index_weighted_by_category_vector.to(self.device)

        # Assign weights based on the categories
        for category, weight in category_weighting.items():
            # Check if the category exists in the skeleton info
            if self.bone_categories is None or category not in self.bone_categories:
                print(f"Warning: Category '{category}' not found in skeleton info. Skipping.")
                continue
            for bone_name in self.bone_categories[category]:
                bone_indices = self.bone_to_indices_map[bone_name]
                # Check if bone exists in the skeleton info
                if bone_indices is None:
                    print(f"Warning: Bone '{bone_name}' not found in skeleton info. Skipping.")
                    continue
                for index in bone_indices:
                    bone_index_weighted_by_category_vector[index] = weight
        # Normalize the vector to ensure it sums to 1
        bone_index_weighted_by_category_vector = (bone_index_weighted_by_category_vector / bone_index_weighted_by_category_vector.sum()) * num_features
        return bone_index_weighted_by_category_vector

    def _precompute_mappings(self) -> None:
        """Precompute mappings for fast extraction/insertion."""
        # Reset mappings
        self._pos_extraction_map = []
        self._rot_extraction_map = []
        
        output_idx = 0
        
        # Process each joint
        for joint_name in self.joints:
            # Skip if not in target joints
            if self.target_joints is not None and joint_name not in self.target_joints:
                continue
            
            # Handle body_world: extract positions only
            if joint_name == 'body_world':
                self._pos_extraction_map.append((output_idx, self.joint_position_indices[joint_name]))
                self.bone_to_indices_map[joint_name] = [output_idx, 
                                                       output_idx + 1, 
                                                       output_idx + 2]
                output_idx += 3
            
            # For non-body_world joints, extract rotations
            if joint_name != 'body_world' and len(self.joint_rotation_indices[joint_name]) == 3:
                # Each rotation group is mapped to 6 output values (6D representation)
                self._rot_extraction_map.append((output_idx, self.joint_rotation_indices[joint_name]))
                self.bone_to_indices_map[joint_name] = [output_idx, 
                                                       output_idx + 1, 
                                                       output_idx + 2, 
                                                       output_idx + 3, 
                                                       output_idx + 4,
                                                       output_idx + 5]
                output_idx += 6
    
    def get_channel_count(self) -> int:
        """Get the number of channels in the extracted data."""
        count = 0
        
        for joint_name in self.joints:
            if self.target_joints is not None and joint_name not in self.target_joints:
                continue
                
            if joint_name == 'body_world':
                # Count position channels
                count += len(self.joint_position_indices[joint_name])
            else:
                # Each rotation becomes 6 values
                if len(self.joint_rotation_indices[joint_name]) == 3:
                    count += 6
        
        return count
    
    def _euler_to_6d_batch(self, euler_batch: np.ndarray) -> np.ndarray:
        """Convert Euler angles to 6D rotation representation."""
        # Create rotation objects (handles all frames at once)
        rot = R.from_euler('ZXY', euler_batch, degrees=True)
        
        # Get rotation matrices
        matrices = rot.as_matrix()  # Shape: (num_frames, 3, 3)
        
        # Extract first two columns and combine
        col1 = matrices[:, :, 0]  # Shape: (num_frames, 3)
        col2 = matrices[:, :, 1]  # Shape: (num_frames, 3)
        
        # Combine into 6D representation
        return np.hstack([col1, col2])  # Shape: (num_frames, 6)
    
    def _matrix_to_euler_batch(self, matrix_batch: torch.Tensor) -> np.ndarray:
        """Convert rotation matrices to Euler angles."""
        # matrix_batch is shape (batch_size, num_frames, 3, 3)
        # Reshape to (batch_size * num_frames, 3, 3) for conversion
        batch_size, num_frames, _, _ = matrix_batch.shape
        
        matrix_batch = matrix_batch.reshape(-1, 3, 3)  # Shape: (batch_size * num_frames, 3, 3)

        # Convert to rotation objects
        rot = R.from_matrix(matrix_batch)
        
        # Convert to Euler angles in ZXY order
        euler_zxy = rot.as_euler('ZXY', degrees=True)

        # Reshape back to (batch_size, num_frames, 3)
        euler_zxy = euler_zxy.reshape(batch_size, num_frames, 3)
        
        return euler_zxy
    
    def _6d_to_euler_batch(self, rot_6d_batch: torch.Tensor) -> np.ndarray:
        """Convert 6D rotation representation to Euler angles."""
        # Convert 6D to rotation matrices
        rot_matrices = convert_6d_to_matrix(rot_6d_batch)
        
        # Convert to Euler angles
        euler_angles = self._matrix_to_euler_batch(rot_matrices)
        
        return euler_angles
    
    def normalize_poses(self, pose: torch.Tensor) -> torch.Tensor:
        """Normalize the pose using mean and std."""
        if self.mean_pose is not None and self.std_pose is not None:
            return (pose - self.mean_pose) / self.std_pose
        return pose
    
    def denormalize_poses(self, pose: torch.Tensor) -> torch.Tensor:
        """Denormalize the pose using mean and std."""
        if self.mean_pose is not None and self.std_pose is not None:
            return pose * self.std_pose.to(pose.dtype) + self.mean_pose.to(pose.dtype)
        return pose
    
    def normalize_world_positions(self, world_positions: torch.Tensor) -> torch.Tensor:
        """Normalize the world positions using mean and std."""
        if self.world_pos_mean_pose is not None and self.world_pos_std_pose is not None:
            return (world_positions - self.world_pos_mean_pose) / self.world_pos_std_pose
        return world_positions
    
    def denormalize_world_positions(self, world_positions: torch.Tensor) -> torch.Tensor:
        """Denormalize the world positions using mean and std."""
        if self.world_pos_mean_pose is not None and self.world_pos_std_pose is not None:
            return world_positions * self.world_pos_std_pose + self.world_pos_mean_pose
        return world_positions
    
    def calculate_world_positions(self, frame_data: torch.Tensor, return_rotations=False) -> torch.Tensor:
        """Calculate world positions using forward kinematics."""

        # Check if the input is a npy array or a torch tensor (The data proccessing pipeline uses npy arrays)
        if isinstance(frame_data, np.ndarray):
            frame_data = torch.tensor(frame_data, dtype=torch.float32, device=self.device)

        # Check if there is a batch dimension
        has_batch_dim = len(frame_data.shape) == 3
        if not has_batch_dim:
            # Add batch dimension
            frame_data = frame_data.unsqueeze(0)

        batch_size, num_frames, _ = frame_data.shape
        
        # Dictionary to store world positions
        world_positions = {}
        
        # Dictionary to store world rotations (as matrices)
        world_rotations = {}
        
        # Get root position
        for joint_name in self.joints:
            if joint_name == 'body_world' and joint_name in self.bone_to_indices_map:
                root_idx = self.bone_to_indices_map[joint_name][0]
                # Extract the root position for all batches and frames
                root_pos = frame_data[:, :, root_idx:root_idx+3]
                world_positions[joint_name] = root_pos
                # Root has identity rotation by default
                identity = torch.eye(3, device=frame_data.device)
                world_rotations[joint_name] = identity.repeat(batch_size, num_frames, 1, 1)
                break
        
        # Process joints in hierarchy order
        processed = {'body_world'}
        
        while len(processed) < len(self.joints):
            for joint_name in self.joints:
                # Skip if already processed
                if joint_name in processed:
                    continue
                
                # Process joint if its parent has been processed
                parent_name = self.joint_parent.get(joint_name)

                if parent_name in processed:
                    # Get parent world rotation and position
                    parent_rot = world_rotations[parent_name]  # shape: (batch_size, num_frames, 3, 3)
                    parent_pos = world_positions[parent_name]  # shape: (batch_size, num_frames, 3)

                    # Get joint offset and expand for all batches and frames
                    offset = torch.tensor(self.joint_offsets.get(joint_name, [0, 0, 0]), 
                                          device=frame_data.device)
                    offset = offset.view(1, 1, 3).expand(batch_size, num_frames, 3)
                    
                    # Get joint local rotation (if available)
                    local_rot_batch = torch.eye(3, device=frame_data.device).repeat(batch_size, num_frames, 1, 1)
                    
                    if joint_name in self.bone_to_indices_map:
                        rot_indices = self.bone_to_indices_map[joint_name]
                        if len(rot_indices) == 6:  # 6D rotation
                            start_idx = rot_indices[0]
                            # Extract 6D rotation for all batches and frames
                            rot_6d = frame_data[:, :, start_idx:start_idx+6]
                            matrices = convert_6d_to_matrix(rot_6d)
                            # Reshape back to batch dimensions
                            local_rot_batch = matrices
                    
                    # Calculate world rotation
                    world_rot = torch.matmul(parent_rot, local_rot_batch)
                    world_rotations[joint_name] = world_rot
                    
                    # Calculate world position: parent_pos + (parent_rot * offset)
                    # Need to reshape for proper broadcasting in matmul
                    rotated_offset = torch.matmul(
                        parent_rot, 
                        offset.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    world_pos = parent_pos + rotated_offset
                    world_positions[joint_name] = world_pos
                    
                    # Mark as processed
                    processed.add(joint_name)
        
        # Compile a tensor of target joint world positions
        target_joint_names = self.target_joints if self.target_joints else self.joints
        world_positions_tensor = torch.zeros((batch_size, num_frames, len(target_joint_names), 3), device=frame_data.device)
        
        for i, joint_name in enumerate(target_joint_names):
            if joint_name in world_positions:
                world_positions_tensor[:, :, i, :] = world_positions[joint_name]
        
        flattened_world_positions = world_positions_tensor.reshape(world_positions_tensor.shape[0], world_positions_tensor.shape[1], -1) # shape: (batch_size, num_frames, 3 * num_joints)
        
        if not has_batch_dim:
            # Remove batch dimension if it was added
            flattened_world_positions = flattened_world_positions.squeeze(0)

        if return_rotations:
            # Filter world_rotations to only include target joints
            if self.target_joints is not None:
                filtered_rotations = {k: v for k, v in world_rotations.items() if k in self.target_joints}
                if not has_batch_dim:
                    # Remove batch dimension from rotations if it was added
                    filtered_rotations = {k: v.squeeze(0) for k, v in filtered_rotations.items()}
                return flattened_world_positions, filtered_rotations
            if not has_batch_dim:
                # Remove batch dimension from rotations if it was added
                world_rotations = {k: v.squeeze(0) for k, v in world_rotations.items()}
            return flattened_world_positions, world_rotations
        
        return flattened_world_positions

    def pose_to_websocket_format(self, pose):   
        # Use a dictionary with bone names as keys instead of an array
        pose_data = {}
        # Process root position (first 3 values if present)
        if len(self._pos_extraction_map) > 0:
            pose_data["body_world"] = {
                "position": {
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2])
                },
                # Default identity quaternion for root if not rotated
                "eulerAngles": {"x": 0.0, "y": 0.0, "z": 0.0}
            }
        
        # Process each joint's rotation
        for i, (out_start_idx, _) in enumerate(self._rot_extraction_map):
            # Get bone name for this rotation
            bone_name = list(self.bone_to_indices_map.keys())[i+1] # +1 because body_world is already processed
                
            # Get the 6D rotation representation
            rot_6d = pose[out_start_idx:out_start_idx+6].reshape(1,1,6)
            
            # Convert 6D representation to Euler angles
            euler_angles = self._6d_to_euler_batch(rot_6d)[0,0]
            
            # Add both quaternion and Euler angles to the frame data
            pose_data[bone_name] = {
                "eulerAngles": {
                    "x": float(euler_angles[1]),  # X is second in ZXY order
                    "y": float(euler_angles[2]),  # Y is third
                    "z": float(euler_angles[0])   # Z is first
                }
            }
        
        # Return single frame or array of frames
        return pose_data

    def get_joint_rotation_indices(self, joint_name: str) -> List[int]:
        """Get the rotation indices for a joint in the feature vector."""
        if joint_name not in self.bone_to_indices_map:
            raise ValueError(f"Joint '{joint_name}' not found in skeleton. Available joints: {list(self.bone_to_indices_map.keys())}")
        
        indices = self.bone_to_indices_map[joint_name]
        # For rotations, we expect 6 indices (6D representation)
        if len(indices) != 6:
            raise ValueError(f"Joint '{joint_name}' does not have 6D rotation representation. Found {len(indices)} indices.")
        
        return indices

    def get_joint_position_indices(self, joint_name: str) -> List[int]:
        """Get the position indices for a joint in the feature vector."""
        if joint_name != 'body_world':
            raise ValueError(f"Only 'body_world' joint has position channels. Got '{joint_name}'")
        
        if joint_name not in self.bone_to_indices_map:
            raise ValueError(f"Joint '{joint_name}' not found in skeleton.")
        
        indices = self.bone_to_indices_map[joint_name]
        # For positions, we expect 3 indices
        if len(indices) != 3:
            raise ValueError(f"Joint '{joint_name}' does not have 3D position representation. Found {len(indices)} indices.")
        
        return indices
    
    def calculate_distance_between_joints(self, joint_a: str, joint_b: str, reference_pose: torch.Tensor = None) -> float:
        """
        Calculate the direct distance between two joints in world space.
        """
        # Get target joints list
        target_joints = self.target_joints if self.target_joints else self.joints
        
        # Check if joints exist in the skeleton
        if joint_a not in target_joints or joint_b not in target_joints:
            available_joints = target_joints[:10]
            raise ValueError(f"Joint not found in skeleton. Available joints: {available_joints}...")
        
        # Get joint indices in the world positions array
        joint_a_idx = target_joints.index(joint_a)
        joint_b_idx = target_joints.index(joint_b)
        
        # Create REST pose if none provided (not zeros!)
        if reference_pose is None:
            # Create a valid rest pose instead of zeros
            reference_pose = torch.zeros((1, self.get_channel_count()), device=self.device)
            # For 6D rotations, use [1,0,0,0,1,0] which represents identity rotation
            for i, (out_start_idx, _) in enumerate(self._rot_extraction_map):
                reference_pose[0, out_start_idx] = 1.0
                reference_pose[0, out_start_idx+4] = 1.0
        
        # Calculate world positions
        world_positions = self.calculate_world_positions(reference_pose)
        
        # Handle different result shapes (with/without batch dimension)
        if len(world_positions.shape) == 2:  # [batch, joints*3]
            # Extract positions for the specific joints
            pos_a = world_positions[0, joint_a_idx*3:(joint_a_idx+1)*3]
            pos_b = world_positions[0, joint_b_idx*3:(joint_b_idx+1)*3]
        else:  # [joints*3]
            pos_a = world_positions[joint_a_idx*3:(joint_a_idx+1)*3]
            pos_b = world_positions[joint_b_idx*3:(joint_b_idx+1)*3]
        
        # Check for NaN values
        if torch.isnan(pos_a).any() or torch.isnan(pos_b).any():
            print(f"Warning: NaN detected in joint positions for {joint_a} or {joint_b}")
            return 0.0  # Return a default value
            
        # Calculate Euclidean distance
        distance = torch.norm(pos_b - pos_a).item()
        
        # Check for NaN distance
        if np.isnan(distance):
            print(f"Warning: NaN distance between {joint_a} and {joint_b}")
            return 0.0  # Return a default value
            
        return distance