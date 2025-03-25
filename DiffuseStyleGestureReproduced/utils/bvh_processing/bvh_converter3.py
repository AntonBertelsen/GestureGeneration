import numpy as np
import torch
import re
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict, Set, Optional, Union

class OffsetBVHParser:
    def __init__(self, file_path: str, target_joints: Optional[List[str]] = None):
        self.file_path = file_path
        self.target_joints = set(target_joints) if target_joints else None
        
        # BVH data
        self.hierarchy_text = ""
        self.motion_data = None
        self.frame_time = 0.0
        self.num_frames = 0
        
        # Joint data
        self.joints = []  # List of joints in order of appearance
        self.joint_channels = {}  # Dict mapping joint name to channel names
        self.joint_channel_start = {}  # Dict mapping joint name to starting channel index
        
        # Position channel tracking
        self.joint_position_indices = {}  # Maps joint name to position channel indices
        self.joint_rotation_indices = {}  # Maps joint name to rotation channel indices
        
        # Hierarchy data for forward kinematics
        self.joint_offsets = {}  # Maps joint name to its offset
        self.joint_parent = {}   # Maps joint name to its parent's name
        self.end_sites = {}      # Maps joint name to end site offset
        
        # Extraction/insertion mappings
        self._pos_extraction_map = []  # [(output_idx, input_idx)] for positions
        self._rot_extraction_map = []  # [(output_start_idx, [x_idx, y_idx, z_idx])] for rotations
        
        # Parse the file
        self._parse_bvh()
        
        # Precompute mappings for fast extraction/insertion
        self._precompute_mappings()
    
    def _parse_bvh(self) -> None:
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        parts = content.split('MOTION')
        self.hierarchy_text = parts[0].strip()
        motion_section = 'MOTION' + parts[1] if len(parts) > 1 else ""
        
        # Parse hierarchy using stack-based approach
        self._parse_hierarchy_stack()
        
        # Parse motion data
        self._parse_motion(motion_section)
    
    def _parse_hierarchy_stack(self) -> None:
        lines = self.hierarchy_text.split('\n')
        stack = []  # Stack to track current joint hierarchy
        current_joint = None
        channel_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('HIERARCHY'):
                continue
            elif line.startswith('ROOT') or line.startswith('JOINT'):
                # New joint
                parts = line.split()
                joint_name = parts[1]
                
                # Store joint
                self.joints.append(joint_name)
                self.joint_channels[joint_name] = []
                self.joint_channel_start[joint_name] = channel_index
                self.joint_position_indices[joint_name] = []
                self.joint_rotation_indices[joint_name] = []
                
                # Set parent relationship
                if stack:
                    parent_name = stack[-1]
                    self.joint_parent[joint_name] = parent_name
                else:
                    self.joint_parent[joint_name] = None
                
                # Make this the current joint
                current_joint = joint_name
                stack.append(joint_name)
                
            elif line.startswith('End Site') and current_joint:
                # End site belongs to the current joint
                # We'll process its offset in the OFFSET section
                pass
                
            elif line.startswith('OFFSET'):
                parts = line.split()
                offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                
                # Check if this is an end site offset
                if i > 0 and lines[i-1].strip().startswith('End Site'):
                    # Store end site offset with current joint
                    self.end_sites[current_joint] = offset
                else:
                    # Regular joint offset
                    self.joint_offsets[current_joint] = offset
                    
            elif line.startswith('CHANNELS'):
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                
                # Store channel information
                self.joint_channels[current_joint] = channels
                
                # Track position and rotation channel indices
                for j, channel in enumerate(channels):
                    if 'position' in channel:
                        self.joint_position_indices[current_joint].append(channel_index + j)
                    elif 'rotation' in channel:
                        self.joint_rotation_indices[current_joint].append(channel_index + j)
                
                # Update global channel index
                channel_index += num_channels
                
            elif line == '}':
                # End of current joint definition, pop from stack
                if stack:
                    stack.pop()
                    current_joint = stack[-1] if stack else None
    
    def _parse_motion(self, motion_text: str) -> None:
        lines = motion_text.strip().split('\n')
        
        # Get number of frames
        frames_match = re.match(r'Frames:\s+(\d+)', lines[1])
        if frames_match:
            self.num_frames = int(frames_match.group(1))
        
        # Get frame time
        frame_time_match = re.match(r'Frame Time:\s+(\d+\.\d+)', lines[2])
        if frame_time_match:
            self.frame_time = float(frame_time_match.group(1))
        
        # Parse motion data
        motion_data = []
        for i in range(3, len(lines)):
            if lines[i].strip():
                motion_data.append([float(x) for x in lines[i].strip().split()])
        
        self.motion_data = np.array(motion_data)
    
    def _precompute_mappings(self) -> None:
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
                for idx in self.joint_position_indices[joint_name]:
                    self._pos_extraction_map.append((output_idx, idx))
                    output_idx += 1
            
            # For non-body_world joints, extract rotations
            if joint_name != 'body_world' and len(self.joint_rotation_indices[joint_name]) == 3:
                # Each rotation group is mapped to 6 output values (6D representation)
                self._rot_extraction_map.append((output_idx, self.joint_rotation_indices[joint_name]))
                output_idx += 6
    
    def extract_channels(self) -> np.ndarray:
        # Initialize result array with exact size needed
        channel_count = self.get_channel_count()
    
        if channel_count == 0:
            raise ValueError("No channels to extract. Check if target_joints contains valid joint names.")
    
        result = np.zeros((self.num_frames, channel_count))
        
        # Extract position channels (direct copy)
        for out_idx, in_idx in self._pos_extraction_map:
            result[:, out_idx] = self.motion_data[:, in_idx]
        
        # Process rotation channels in batches
        for out_start_idx, in_indices in self._rot_extraction_map:
            # Extract Euler angles for this joint for all frames at once
            euler_batch = self.motion_data[:, in_indices]
            
            # Convert to 6D rotations using scipy (vectorized)
            rot_6d_batch = self._euler_to_6d_batch(euler_batch)
            
            # Store in result
            result[:, out_start_idx:out_start_idx+6] = rot_6d_batch
        
        return result
    
    def to_tensor(self) -> torch.Tensor:
        np_array = self.extract_channels()
        return torch.from_numpy(np_array).float()
    
    def update_motion_data(self, new_data: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.detach().cpu().numpy()
        
        # Update positions (direct copy)
        for out_idx, in_idx in self._pos_extraction_map:
            self.motion_data[:, in_idx] = new_data[:, out_idx]
        
        # Update rotations (convert from 6D to Euler)
        for out_start_idx, in_indices in self._rot_extraction_map:
            # Extract 6D rotations for all frames at once
            rot_6d_batch = new_data[:, out_start_idx:out_start_idx+6]
            
            # print the bone name
            print(f"Bone name: {self.joints[1+out_start_idx//6]}")
            # Convert back to Euler (vectorized)
            euler_batch = self._6d_to_euler_batch(rot_6d_batch)
            
            # Update motion data
            self.motion_data[:, in_indices] = euler_batch
        
        # Set position channels for non-root joints to their offset values
        self._set_non_root_positions_to_offsets()
    
    def _set_non_root_positions_to_offsets(self) -> None:
        for joint_name, pos_indices in self.joint_position_indices.items():
            # Skip root joint
            if joint_name == 'body_world' or not pos_indices:
                continue
            
            # Get joint offset
            offset = self.joint_offsets.get(joint_name, [0, 0, 0])
            
            # Set position channels to offset values for all frames
            for i, idx in enumerate(pos_indices):
                if i < len(offset):
                    self.motion_data[:, idx] = offset[i]
    
    def write_bvh(self, output_file: str) -> None:
        # Ensure non-root positions are set to offsets
        self._set_non_root_positions_to_offsets()
        
        # Construct the motion section
        motion_text = f"MOTION\nFrames: {self.num_frames}\nFrame Time: {self.frame_time}\n"
        
        # Format data with higher precision
        np.set_printoptions(precision=6, threshold=np.inf, suppress=True)
        for frame in self.motion_data:
            motion_text += " ".join([f"{val:.6f}" for val in frame]) + "\n"
        
        # Write to file - original hierarchy + updated motion
        with open(output_file, 'w') as f:
            f.write(self.hierarchy_text + "\n" + motion_text)
    
    def _euler_to_6d_batch(self, euler_batch: np.ndarray) -> np.ndarray:
        # Rearrange for scipy: from [x,y,z] to [z,x,y]
        zxy_batch = euler_batch[:, [2, 0, 1]]
        
        # Create rotation objects (handles all frames at once)
        rot = R.from_euler('zxy', zxy_batch, degrees=True)
        
        # Get rotation matrices
        matrices = rot.as_matrix()  # Shape: (num_frames, 3, 3)
        
        # Extract first two columns and combine
        col1 = matrices[:, :, 0]  # Shape: (num_frames, 3)
        col2 = matrices[:, :, 1]  # Shape: (num_frames, 3)
        
        # Combine into 6D representation
        return np.hstack([col1, col2])  # Shape: (num_frames, 6)
    
    def _6d_to_euler_batch(self, rot_6d_batch: np.ndarray) -> np.ndarray:
        num_frames = rot_6d_batch.shape[0]
        
        # Extract columns
        col1 = rot_6d_batch[:, 0:3]  # Shape: (num_frames, 3)
        col2 = rot_6d_batch[:, 3:6]  # Shape: (num_frames, 3)
        
        # Normalize columns (vectorized)
        col1_norm = np.linalg.norm(col1, axis=1, keepdims=True)
        col2_norm = np.linalg.norm(col2, axis=1, keepdims=True)
        col1 = col1 / col1_norm
        col2 = col2 / col2_norm
        
        # Compute cross product for third column (vectorized)
        col3 = np.cross(col1, col2)
        
        # Stack into rotation matrices
        matrices = np.zeros((num_frames, 3, 3))
        matrices[:, :, 0] = col1
        matrices[:, :, 1] = col2
        matrices[:, :, 2] = col3
        
        # Convert to rotation objects
        rot = R.from_matrix(matrices)
        
        # Convert to Euler angles in ZXY order
        euler_zxy = rot.as_euler('zxy', degrees=True)

        # detect large differences in z rotation for each bone
        for i in range(1, euler_zxy.shape[0]):
            # for _ in range(2): # Try twice. This is because we may have a jump of 360 degrees
            for j in range(euler_zxy.shape[1]):
                difference = euler_zxy[i, j] - euler_zxy[i-1, j]
                if abs(difference) > 160:
                    print(j, "difference at frame", i, ":", difference)
                    # print x y z for the 10 surrounding frames
                    for k in range(-5, 6):
                        print(euler_zxy[i+k, :])
                    rounded_diff = np.round(difference / 180) * 180
                    euler_zxy[i:, j] -= np.sign(difference) * rounded_diff
        # Rearrange from [z,x,y] back to [x,y,z]
        return euler_zxy[:, [1, 2, 0]]
    
    def get_all_joints(self) -> List[str]:
        return self.joints
    
    def set_target_joints(self, target_joints: List[str]) -> None:
        self.target_joints = set(target_joints) if target_joints else None
        self._precompute_mappings()
        
    def get_channel_count(self) -> int:
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
    

    def calculate_world_positions(self, frame_idx: int = 0) -> Dict[str, np.ndarray]:
        # Get the motion data for this frame
        frame_data = self.motion_data[frame_idx]
        
        # Create result dictionary
        world_positions = {}
        
        # Process joints in order
        for joint_name in self.joints:
            # Get joint's parent
            parent_name = self.joint_parent[joint_name]
            
            # Get joint's rotation
            rot_indices = self.joint_rotation_indices[joint_name]
            if rot_indices:
                rot_euler = frame_data[rot_indices]
                # Convert to matrix using ZXY order
                rot_matrix = R.from_euler('zxy', [rot_euler[2], rot_euler[0], rot_euler[1]], degrees=True).as_matrix()
            else:
                rot_matrix = np.eye(3)
            
            # Get offset
            offset = np.array(self.joint_offsets.get(joint_name, [0, 0, 0]))
            
            if parent_name is None:
                # Root joint - use global position
                pos_indices = self.joint_position_indices[joint_name]
                if pos_indices:
                    world_pos = frame_data[pos_indices]
                else:
                    world_pos = offset
                
                # Store results
                world_positions[joint_name] = world_pos
                
                # Calculate end site if present
                if joint_name in self.end_sites:
                    end_offset = np.array(self.end_sites[joint_name])
                    end_pos = world_pos + rot_matrix @ end_offset
                    world_positions[f"{joint_name}_end"] = end_pos
            else:
                # Child joint - combine with parent's transform
                if parent_name in world_positions:
                    parent_pos = world_positions[parent_name]
                    parent_rot_matrix = self._get_rotation_matrix(parent_name, frame_data)
                    
                    # Transform offset by parent's rotation
                    transformed_offset = parent_rot_matrix @ offset
                    
                    # Calculate world position
                    world_pos = parent_pos + transformed_offset
                    world_positions[joint_name] = world_pos
                    
                    # Calculate end site if present
                    if joint_name in self.end_sites:
                        world_rot_matrix = parent_rot_matrix @ rot_matrix
                        end_offset = np.array(self.end_sites[joint_name])
                        end_pos = world_pos + world_rot_matrix @ end_offset
                        world_positions[f"{joint_name}_end"] = end_pos
        
        return world_positions
    
    def _get_rotation_matrix(self, joint_name: str, frame_data: np.ndarray) -> np.ndarray:
        # Base case: root joint
        if self.joint_parent[joint_name] is None:
            rot_indices = self.joint_rotation_indices[joint_name]
            if rot_indices:
                rot_euler = frame_data[rot_indices]
                # Convert to matrix using ZXY order
                return R.from_euler('zxy', [rot_euler[2], rot_euler[0], rot_euler[1]], degrees=True).as_matrix()
            else:
                return np.eye(3)
        
        # Recursive case: combine with parent
        parent_name = self.joint_parent[joint_name]
        parent_matrix = self._get_rotation_matrix(parent_name, frame_data)
        
        # Get joint's rotation
        rot_indices = self.joint_rotation_indices[joint_name]
        if rot_indices:
            rot_euler = frame_data[rot_indices]
            # Convert to matrix using ZXY order
            joint_matrix = R.from_euler('zxy', [rot_euler[2], rot_euler[0], rot_euler[1]], degrees=True).as_matrix()
            return parent_matrix @ joint_matrix
        else:
            return parent_matrix