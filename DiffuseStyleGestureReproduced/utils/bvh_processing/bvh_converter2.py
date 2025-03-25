import os
import numpy as np
from utils.bvh_processing.pymo.parsers import BVHParser
from utils.bvh_processing.pymo.writers import BVHWriter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.bvh_processing.pymo.data import MocapData
        

# Target joints list
target_joints = ['body_world', 'b_root', 'b_r_foot', 'b_l_foot', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 
                'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 
                'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 
                'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 
                'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 
                'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 
                'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 
                'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 
                'b_neck0', 'b_head']

stats_file = 'dataset/genea2023_dataset/trn/main-agent/stats.npz'

class OptimizedBVHProcessor:
    """
    Optimized BVH processor with better performance on rotation handling
    """
    # Class variables to store statistics
    means = None
    stds = None
    
    @staticmethod
    def downsample(motion_data, tgt_fps=30):
        """Downsample motion data to target FPS using NumPy"""
        orig_fps = round(1.0 / motion_data.framerate)
        rate = orig_fps // tgt_fps

        if orig_fps % tgt_fps != 0:
            print(f"Warning: orig_fps ({orig_fps}) is not divisible by tgt_fps ({tgt_fps})")
        
        if rate == 1.0:
            return motion_data
        
        # Create downsampled motion data
        new_motion = motion_data.clone()
        
        # Downsample values (NumPy slicing)
        new_motion.values = motion_data.values[0:-1:rate].copy()
        new_motion.framerate = 1.0 / tgt_fps
        return new_motion
    
    @staticmethod
    def select_joints(motion_data, joint_list):
        """
        Select only specified joints from motion data
        Optimized to minimize data copying
        """
        # Get indices of selected channels
        selected_indices = []
        selected_channel_names = []
        
        for i, col_name in enumerate(motion_data.channel_names):
            if 'Nub' in col_name:
                continue
            
            # Split on the last underscore and use everything before it as the joint name
            parts = col_name.rsplit('_', 1)
            if len(parts) < 2:
                continue  # Skip column names without underscores

            # Skip position channels except for body_world
            if 'position' in parts[1] and parts[0] != 'body_world':
                continue
            
            joint_name = parts[0]
            
            if joint_name in joint_list:
                selected_indices.append(i)
                selected_channel_names.append(col_name)
        # Create new motion with only selected joints
        new_motion = motion_data.clone()
        
        # Remove joints not in the selected list
        new_motion.skeleton = {k: v for k, v in new_motion.skeleton.items() if k in joint_list}
        # Select only the relevant columns from the NumPy array
        new_motion.values = motion_data.values[:, selected_indices].copy()
        
        # Update channel names to reflect selection
        new_motion.channel_names = selected_channel_names
        
        return new_motion
        
    @staticmethod
    def to_rotation_matrices(motion_data):
        """
        Convert Euler angles to 6D rotation matrices more efficiently.
        Processes joints directly from skeleton and fills output array in-place.
        """
        # Clone motion data to maintain the original
        new_motion = motion_data.clone()
        
        # Get joints that are not end sites
        joints = [joint for joint in motion_data.skeleton if 'Nub' not in joint]
        
        # Calculate the size of the new array
        n_frames = motion_data.values.shape[0]
        n_channels = 0
        new_channel_names = []
        
        # Count channels and prepare channel names
        for joint in joints:
            rot_order = motion_data.skeleton[joint].get('order', '')
            if rot_order:
                # 6 values for rotation matrix (first two rows)
                n_channels += 6
                new_channel_names.extend([f'{joint}_r{i}' for i in range(1, 7)])
            
            # Check if this joint has position channels
            pos_channels = [f'{joint}_{axis}position' for axis in ['X', 'Y', 'Z']]
            for pos_channel in pos_channels:
                if pos_channel in motion_data.channel_names:
                    n_channels += 1
                    new_channel_names.append(pos_channel)
        
        # Create the output array
        new_values = np.zeros((n_frames, n_channels))
        
        # Fill the output array
        col_idx = 0
        for joint in joints:
            rot_order = motion_data.skeleton[joint].get('order', '')
            
            # Process position channels
            for axis in ['X', 'Y', 'Z']:
                pos_channel = f'{joint}_{axis}position'
                if pos_channel in motion_data.channel_names:
                    idx = motion_data.channel_names.index(pos_channel)
                    new_values[:, col_idx] = motion_data.values[:, idx]
                    col_idx += 1
            
            # Process rotation channels
            if rot_order:
                # Get rotation channel indices
                x_col = f'{joint}_{rot_order[0]}rotation'
                y_col = f'{joint}_{rot_order[1]}rotation'
                z_col = f'{joint}_{rot_order[2]}rotation'
                
                if all(col in motion_data.channel_names for col in [x_col, y_col, z_col]):
                    x_idx = motion_data.channel_names.index(x_col)
                    y_idx = motion_data.channel_names.index(y_col)
                    z_idx = motion_data.channel_names.index(z_col)
                    
                    # Extract Euler angles
                    euler_angles = np.column_stack([
                        motion_data.values[:, x_idx],
                        motion_data.values[:, y_idx],
                        motion_data.values[:, z_idx]
                    ])
                    
                    # Convert to rotation matrices
                    rotmats = R.from_euler(rot_order.lower(), euler_angles, degrees=True).as_matrix()
                    
                    # Store the first two rows (6 values)
                    new_values[:, col_idx:col_idx+3] = rotmats[:, 0, :]  # First row
                    new_values[:, col_idx+3:col_idx+6] = rotmats[:, 1, :]  # Second row
                    col_idx += 6
        
        # Update the motion data
        new_motion.values = new_values
        new_motion.channel_names = new_channel_names
        
        return new_motion
    
    @staticmethod
    def from_rotation_matrices(motion_data):
        """
        Convert 6D rotation matrices back to Euler angles.
        """
        # Create new motion data object
        new_motion = motion_data.clone()
        
        skeleton = motion_data.skeleton
        
        # Get joints (excluding end sites)
        joints = [joint for joint in skeleton if 'Nub' not in joint]
        
        # Calculate output array size
        n_frames = motion_data.values.shape[0]
        euler_channels = []
        channel_names = []
        
        # Process each joint
        for joint in joints:
            rot_order = skeleton[joint].get('order', '')
            if not rot_order:
                continue
            
            # Process position channels
            for axis in ['X', 'Y', 'Z']:
                pos_channel = f'{joint}_{axis}position'
                if pos_channel in motion_data.channel_names:
                    idx = motion_data.channel_names.index(pos_channel)
                    euler_channels.append(motion_data.values[:, idx].reshape(-1, 1))
                    channel_names.append(pos_channel)
            
            # Process rotation channels
            r_cols = [f'{joint}_r{i}' for i in range(1, 7)]
            if all(col in motion_data.channel_names for col in r_cols):
                # Get indices of the 6D rotation components
                indices = [motion_data.channel_names.index(col) for col in r_cols]
                
                # Extract rotation matrix components for all frames
                rotmat_data = motion_data.values[:, indices]
                
                # Reshape to proper 6D representation
                rotmats = np.zeros((n_frames, 3, 3))
                rotmats[:, 0, :] = rotmat_data[:, 0:3]  # First row
                rotmats[:, 1, :] = rotmat_data[:, 3:6]  # Second row
                
                # Compute third row as cross product
                rotmats[:, 2, 0] = rotmats[:, 0, 1] * rotmats[:, 1, 2] - rotmats[:, 0, 2] * rotmats[:, 1, 1]
                rotmats[:, 2, 1] = rotmats[:, 0, 2] * rotmats[:, 1, 0] - rotmats[:, 0, 0] * rotmats[:, 1, 2]
                rotmats[:, 2, 2] = rotmats[:, 0, 0] * rotmats[:, 1, 1] - rotmats[:, 0, 1] * rotmats[:, 1, 0]
                
                # Convert to Euler angles (batched)
                euler_angles = R.from_matrix(rotmats).as_euler(rot_order.lower(), degrees=True)
                
                # Store results for each axis
                for i, axis in enumerate(rot_order.upper()):
                    euler_channels.append(euler_angles[:, i].reshape(-1, 1))
                    channel_names.append(f'{joint}_{axis}rotation')
        
        # Combine all channels into one array
        new_motion.values = np.hstack(euler_channels) if euler_channels else np.array([])
        new_motion.channel_names = channel_names
        new_motion.skeleton = skeleton
        
        return new_motion
    
    @classmethod
    def calculate_dataset_statistics(cls, bvh_files, bvh_dir, stats_file=stats_file):
        """
        Efficiently calculate mean and std for each feature across all BVH files.
        Uses vectorized operations and minimizes object creation.
        """
        parser = BVHParser()
        
        # Initialize using running statistics to save memory
        feature_sum = None
        feature_sum_sq = None
        total_frames = 0

        cls.original_skeleton = None
        
        print(f"Calculating statistics across {len(bvh_files)} BVH files...")
        
        # Use tqdm for the progress bar
        for bvh_file in tqdm(bvh_files, desc="(1/2) Calculating mean and std"):
            try:
                # Parse BVH
                bvh_path = os.path.join(bvh_dir, bvh_file)
                motion_data = parser.parse(bvh_path)
                
                if cls.original_skeleton is None:
                    cls.original_skeleton = motion_data.skeleton

                # Apply preprocessing
                motion_data = cls.downsample(motion_data)
                motion_data = cls.select_joints(motion_data, target_joints)
                motion_data = cls.to_rotation_matrices(motion_data)
                
                features = motion_data.values
                
                # Update running statistics
                if feature_sum is None:
                    feature_sum = np.sum(features, axis=0)
                    feature_sum_sq = np.sum(features**2, axis=0)
                else:
                    feature_sum += np.sum(features, axis=0)
                    feature_sum_sq += np.sum(features**2, axis=0)
                
                total_frames += features.shape[0]
                
            except Exception as e:
                print(f"Error processing {bvh_file}: {e}")
        
        # Calculate final statistics
        means = feature_sum / total_frames
        stds = np.sqrt(feature_sum_sq / total_frames - means**2)
        
        # Prevent division by zero in standardization
        stds[stds < 1e-6] = 1.0
        
        # Save statistics
        print("original skeleton", cls.original_skeleton.keys())
        np.savez(stats_file, means=means, stds=stds, skeleton=motion_data.skeleton, original_skeleton=cls.original_skeleton)
        print(f"Statistics saved to {stats_file}")
        
        # Store in class variables
        cls.means = means
        cls.stds = stds
        cls.skeleton = motion_data.skeleton
        
        return means, stds, motion_data.skeleton, cls.original_skeleton
    
    @classmethod
    def load_statistics(cls, stats_file=stats_file):
        """Load pre-computed statistics"""
        data = np.load(stats_file, allow_pickle=True)
        cls.means = data['means']
        cls.stds = data['stds']
        # Convert skeleton from numpy array to dictionary
        cls.original_skeleton = data['original_skeleton'].item()
        cls.skeleton = data['skeleton'].item()
        return cls.means, cls.stds, cls.skeleton, cls.original_skeleton
    
    @classmethod
    def bvh_to_features(cls, bvh_path, stats_file=stats_file, standardize=True):
        """
        Convert BVH file to standardized features efficiently
        """
        # Load statistics if needed and not already loaded
        if standardize and (cls.means is None or cls.stds is None):
            cls.load_statistics(stats_file)
        
        # Parse BVH
        parser = BVHParser()
        motion_data = parser.parse(bvh_path)
        
        # Process motion data
        motion_data = cls.downsample(motion_data)
        motion_data = cls.select_joints(motion_data, target_joints)
        motion_data = cls.to_rotation_matrices(motion_data)
            
        # Get features as numpy array
        features = motion_data.values
        
        # Standardize if requested
        if standardize:
            features = (features - cls.means) / cls.stds
        
        # print the minimum and maximum values of the features
        print("min", np.min(features))
        print("max", np.max(features))

        print("std after", np.std(features))
        print("mean after", np.mean(features))

        outlier_percentage = np.mean((features < -3) | (features > 3)) * 100
        print(f"Outliers beyond ±3 std: {outlier_percentage:.2f}%")

        return features
    
    @staticmethod
    def restore_full_skeleton(motion_data, original_skeleton):
        """
        Restore the full skeleton hierarchy by properly handling joint positions and rotations.
        """
        # Store current values
        current_channels = motion_data.channel_names
        current_values = motion_data.values
        n_frames = current_values.shape[0]
        
        # Make a deep copy of the original skeleton
        restored_skeleton = {}
        for joint_name, joint_data in original_skeleton.items():
            restored_skeleton[joint_name] = joint_data.copy()
        
        # Collect channels following BVH conventions
        all_channels = []
        
        for joint in restored_skeleton:
            if 'Nub' in joint:
                continue
                
            # Add position channels for the root joint only
            if restored_skeleton[joint].get('parent') is None:
                for axis in ['X', 'Y', 'Z']:
                    all_channels.append(f'{joint}_{axis}position')
            
            # Add rotation channels for all joints
            rot_order = restored_skeleton[joint].get('order', '')
            if rot_order:
                for axis in rot_order:
                    all_channels.append(f'{joint}_{axis}rotation')
        
        # Create a new array with all channels
        new_values = np.zeros((n_frames, len(all_channels)))
        
        # Fill in values from the current motion data
        for i, channel in enumerate(all_channels):
            if channel in current_channels:
                idx = current_channels.index(channel)
                new_values[:, i] = current_values[:, idx]
        
        # Update the motion data
        motion_data.values = new_values
        motion_data.channel_names = all_channels
        
        # Update the 'channels' field in each joint to match the BVH structure
        for joint in restored_skeleton:
            joint_channels = []
            
            # Position channels only for root
            if restored_skeleton[joint].get('parent') is None:
                for axis in ['X', 'Y', 'Z']:
                    joint_channels.append(f'{axis}position')
            
            # Rotation channels for all non-end site joints
            if 'Nub' not in joint:
                rot_order = restored_skeleton[joint].get('order', '')
                if rot_order:
                    for axis in rot_order:
                        joint_channels.append(f'{axis}rotation')
            
            # Update the joint's channels
            restored_skeleton[joint]['channels'] = joint_channels
        
        # Set the skeleton with updated channel definitions
        motion_data.skeleton = restored_skeleton
        
        # Find the root joint
        for joint, props in motion_data.skeleton.items():
            if props.get('parent') is None:
                motion_data.root_name = joint
                break
        
        return motion_data

    @classmethod
    def features_to_bvh(cls, features, bvh_file, stats_file=stats_file, standardized=True):
        """
        Convert features back to BVH file
        """
        # Load statistics if needed
        if standardized and (cls.means is None or cls.stds is None or cls.original_skeleton is None):
            tuple = cls.load_statistics(stats_file)
            print(tuple)
            print("skeleton", cls.original_skeleton.keys())
        
        # Unstandardize features if needed
        if standardized:
            features = features * cls.stds + cls.means
        
        # Create feature names if not already created
        if not hasattr(cls, 'feature_names'):
            cls.feature_names = []
            for joint in target_joints:
                if 'Nub' not in joint:
                    # Add position channels for body_world
                    if joint == 'body_world':
                        for axis in ['X', 'Y', 'Z']:
                            cls.feature_names.append(f'{joint}_{axis}position')
                    
                    # Add rotation channels for all joints
                    for i in range(1, 7):
                        cls.feature_names.append(f'{joint}_r{i}')
        
        # Create MocapData object with the features
        motion_data = MocapData()
        motion_data.skeleton = cls.original_skeleton
        motion_data.values = features
        motion_data.channel_names = cls.feature_names[:features.shape[1]]
        motion_data.framerate = 1.0 / 30.0  # Assuming 30 FPS
        
        # Find the root joint from the skeleton
        root_joint = None
        for joint, props in motion_data.skeleton.items():
            if props.get('parent') is None:
                root_joint = joint
                break
        # Set the root_name attribute
        motion_data.root_name = root_joint
        
        # Convert back to Euler angles
        motion_data = cls.from_rotation_matrices(motion_data)

        print("root joint", root_joint)

        # Zero out root rotation for correct body orientation
        root_channels = ["body_world_Xrotation", "body_world_Yrotation", "body_world_Zrotation"]
        for channel in root_channels:
            if channel in motion_data.channel_names:
                idx = motion_data.channel_names.index(channel)
                motion_data.values[:, idx] = 0
        
        # Restore the full skeleton structure with default values for missing joints
        motion_data = cls.restore_full_skeleton(motion_data, cls.original_skeleton)


        print(motion_data.skeleton.keys())
        print(cls.original_skeleton.keys())
        # Write BVH file
        writer = BVHWriter()
        with open(bvh_file, 'w') as f:
            writer.write(motion_data, f)
        
        print(f"BVH file written to {bvh_file}")