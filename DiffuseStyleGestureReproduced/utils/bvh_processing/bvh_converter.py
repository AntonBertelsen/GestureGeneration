import numpy as np
import torch
import re
from typing import List, Tuple, Dict, Set, Optional, Union
import base64
import io
import soundfile as sf
from utils.bvh_processing.skeleton import Skeleton

# This is a simplified BVH parser. It is inspired by the BVH parser from the Pymo library which is commonly used by Genea applicants.
# This version has fewer features, but it is more efficient and sufficient for our needs.

class BVHParser:
    def __init__(self, file_path: str, target_joints: Optional[List[str]] = None, bone_categories: Optional[List[str]] = None) -> None:
        self.file_path = file_path
        
        # BVH data
        self.hierarchy_text = ""
        self.motion_data = None
        self.frame_time = 0.0
        self.num_frames = 0
        
        # Create skeleton object
        self.skeleton = Skeleton()
        self.skeleton.set_target_joints(target_joints)

        self.skeleton.set_bone_categories(bone_categories)
        
        # Parse the file
        self._parse_bvh()
    
    def _parse_bvh(self) -> None:
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        parts = content.split('MOTION')
        self.hierarchy_text = parts[0].strip()
        motion_section = 'MOTION' + parts[1] if len(parts) > 1 else ""
        
        # Parse hierarchy using stack-based approach
        self._parse_hierarchy_stack()

        # Precompute mappings
        self.skeleton._precompute_mappings()
        
        # Parse motion data
        self._parse_motion(motion_section)
    
    def _parse_hierarchy_stack(self) -> None:
        lines = self.hierarchy_text.split('\n')
        stack = []  # Stack to track current joint hierarchy
        channel_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('HIERARCHY'):
                continue
            elif line.startswith('ROOT') or line.startswith('JOINT'):
                # New joint
                joint_name = line.split()[1]
                
                # Get parent name from stack
                parent_name = stack[-1] if stack else None
                
                # Add joint to skeleton
                self.skeleton.add_joint(joint_name, parent_name)
                
                stack.append(joint_name)
                
            elif line.startswith('End Site'):
                # Push a placeholder for End Site to stack
                stack.append('EndSite_' + stack[-1])  # Unique name to track it
                continue
                
            elif line.startswith('OFFSET'):
                offset = list(map(float, line.split()[1:4]))
                current = stack[-1]

                if current.startswith('EndSite_'):
                    parent_name = stack[-2]  # Get parent of end site
                    self.skeleton.add_end_site(parent_name, offset)
                else:
                    self.skeleton.set_joint_offset(current, offset)
                    
            elif line.startswith('CHANNELS'):
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                current = stack[-1]
                
                # Set channels for the joint
                self.skeleton.set_joint_channels(current, channels, channel_index)
                
                # Update global channel index
                channel_index += num_channels
                
            elif line == '}':
                # End of current joint definition, pop from stack
                if stack:
                    stack.pop()
    
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
    
    def to_features(self) -> np.ndarray:
        # Initialize result array with exact size needed
        channel_count = self.skeleton.get_channel_count()
    
        if channel_count == 0:
            raise ValueError("No channels to extract. Check if target_joints contains valid joint names.")
    
        result = np.zeros((self.num_frames, channel_count))
        
        # Extract position channels (direct copy)
        for out_idx, in_indices in self.skeleton._pos_extraction_map:
            # copy the three position channels for this joint
            result[:, out_idx:out_idx+3] = self.motion_data[:, in_indices]

        # Process rotation channels in batches
        for out_start_idx, in_indices in self.skeleton._rot_extraction_map:
            # Extract Euler angles for this joint for all frames at once
            euler_batch = self.motion_data[:, in_indices]

            # Convert to 6D rotations using the skeleton's method
            rot_6d_batch = self.skeleton._euler_to_6d_batch(euler_batch)
            
            # Store in result
            result[:, out_start_idx:out_start_idx+6] = rot_6d_batch
        
        return result
    
    # def update_motion_data(self, new_data: Union[np.ndarray, torch.Tensor]) -> None:
    #     if isinstance(new_data, torch.Tensor):
    #         new_data = new_data.detach().cpu().numpy()
    #    
    #     # Update positions (direct copy)
    #     for out_idx, in_idx in self._pos_extraction_map:
    #         self.motion_data[:, in_idx] = new_data[:, out_idx]
    #     
    #     # Update rotations (convert from 6D to Euler)
    #     for out_start_idx, in_indices in self._rot_extraction_map:
    #         # Extract 6D rotations for all frames at once
    #         rot_6d_batch = new_data[:, out_start_idx:out_start_idx+6]
    #         
    #         # Convert back to Euler (vectorized)
    #         euler_batch = self._6d_to_euler_batch(rot_6d_batch)
    #         
    #         # Update motion data
    #         self.motion_data[:, in_indices] = euler_batch
    #     
    #     # Set position channels for non-root joints to their offset values
    #     self._set_non_root_positions_to_offsets()
    
    # For some reason I can't quite understand, in the BVH format, you have to supply the offset values for joints in the skeleton definition. However, you also need to supply
    # the same position values in the motion section. So what is the point of the offset values? I don't get it, but this function copies the offset values to the position 
    # values for all joints except the root joint. (since the root joint has real position values)
    def _set_non_root_positions_to_offsets(self) -> None:
        """Set position channels for non-root joints to their offset values."""
        for joint_name, pos_indices in self.skeleton.joint_position_indices.items():
            # Skip root joint
            if joint_name == 'body_world' or not pos_indices:
                continue
            
            # Get joint offset
            offset = self.skeleton.joint_offsets.get(joint_name, [0, 0, 0])
            
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

    
    def get_all_joints(self) -> List[str]:
        """Get all joints in the skeleton."""
        return self.skeleton.joints
    
    def set_target_joints(self, target_joints: List[str]) -> None:
        """Set the target joints for extraction."""
        self.skeleton.set_target_joints(target_joints)