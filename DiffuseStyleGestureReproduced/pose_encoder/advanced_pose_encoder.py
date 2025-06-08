import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from utils.WnB_trackable import WnBTrackable
import torch.nn.functional as F
from pose_encoder.ik_two_bone import IKChain2Bone
from utils.utils import convert_6d_to_matrix, convert_matrix_to_6d

class SimpleEncoder(nn.Module):
    """Simple VAE encoder/decoder."""
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        hidden = max(32, min(128, input_dim))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden//2),
            nn.GELU()
        )
        
        self.mu = nn.Linear(hidden//2, z_dim)
        self.logvar = nn.Linear(hidden//2, z_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden//2),
            nn.GELU(),
            nn.Linear(hidden//2, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim)
        )
    
    def encode(self, x, return_logvar=False):
        h = self.encoder(x)
        mu = self.mu(h)
        if return_logvar:
            return mu, self.logvar(h)
        return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x, return_logvar=True)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z
    
    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class AdvancedPoseEncoder(nn.Module, WnBTrackable):
    def __init__(self, pose_dim=345, component_definitions=None, device=None, checkpoint_path=None, skeleton=None):
        super().__init__()
        if skeleton is None:
            raise ValueError("Skeleton must be provided")
        
        self.pose_dim = pose_dim
        self.device = device if device is not None else torch.device('cpu')
        self.skeleton = skeleton
        
        self.component_definitions = component_definitions
        
        # Resolve components using skeleton's data
        self.components = self._resolve_components_with_skeleton()
        
        # Create joint name to index mapping for world positions
        target_joints = self.skeleton.target_joints if self.skeleton.target_joints else self.skeleton.joints
        self.joint_name_to_world_pos_idx = {
            joint_name: idx for idx, joint_name in enumerate(target_joints)
        }
        
        # Calculate dimensions
        self.preserved_dim = sum(comp['dim'] for comp in self.components.values() if comp['type'] == 'preserve')
        self.encoded_dim = sum(comp['z_dim'] for comp in self.components.values() if comp['type'] == 'encode')
        self.ik_dim = sum(4 for comp in self.components.values() if comp['type'] == 'ik')
        self.world_preserved_after_ik_dim = sum(6 for comp in self.components.values() if comp['type'] == 'world_preserve_after_ik')
        
        self.total_z_dim = self.preserved_dim + self.encoded_dim + self.ik_dim + self.world_preserved_after_ik_dim
        
        # Create encoders for auto-encoded components only
        self.encoders = nn.ModuleDict()
        for name, comp in self.components.items():
            if comp['type'] == 'encode':
                input_dim = len(comp['indices'])
                self.encoders[name] = SimpleEncoder(input_dim, comp['z_dim'])
        
        # Create IK chains for each IK component
        self.ik_chains = {}
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                chain_parent = comp['chain_parent']
                twist_joint = comp['twist_joint']
                bone1_len = comp['bone_lengths'][0]
                bone2_len = comp['bone_lengths'][1]
                
                # Determine forward and up directions
                bone1_forward_dir = torch.tensor(comp['bone1_forward_dir'], device=self.device)
                bone1_up_dir = torch.tensor(comp['bone1_up_dir'], device=self.device)
                bone2_forward_dir = torch.tensor(comp['bone2_forward_dir'], device=self.device)
                bone2_up_dir = torch.tensor(comp['bone2_up_dir'], device=self.device)
                
                self.ik_chains[name] = IKChain2Bone(
                    l1=bone1_len,
                    l2=bone2_len, 
                    chain_parent=chain_parent,
                    twist_joint=twist_joint,
                    device=self.device,
                    bone1_forward_dir=bone1_forward_dir,
                    bone1_up_dir=bone1_up_dir,
                    bone2_forward_dir=bone2_forward_dir,
                    bone2_up_dir=bone2_up_dir
                )
        
        # Print resolved components for debugging
        self._print_component_info()

        # Enhanced hyperparameters including skeleton info
        self.hyperparameter_dict_to_WnB_tracking = {
            "total_z_dim": self.total_z_dim,
            "preserved_dim": self.preserved_dim,
            "ik_dim": self.ik_dim,
            "encoded_dim": self.encoded_dim,
            "pose_dim": pose_dim,
            "ik_solver": "IKChain2Bone",
            "hints_in_latent": True,
            "separate_hand_encoders": True,
            "checkpoint_path": checkpoint_path,
            "architecture": "AdvancedPoseEncoder",
            "skeleton_joints": len(self.skeleton.joints),
            # Simplified IK tracking
            "ik_chains": {name: sum(comp['bone_lengths']) 
                         for name, comp in self.components.items() 
                         if comp['type'] == 'ik'}
        }
        
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(f"pose_encoder/models/{checkpoint_path}", map_location=device))
            print(f"Advanced Pose Encoder loaded from {checkpoint_path}")
        
        if device is not None:
            self.to(device)
    
    def _resolve_components_with_skeleton(self) -> Dict:
        """Resolve component definitions using skeleton's bone length data."""
        resolved_components = {}
        
        # Track which indices are handled by any component
        handled_indices = set()
        
        for name, comp_def in self.component_definitions.items():
            comp = comp_def.copy()
            
            if comp['type'] == 'preserve':
                # Get indices for preserved bones
                indices = self._get_bone_indices_from_skeleton(
                    comp_def['bone_names'], 
                    comp_def.get('use_position', False)
                )
                comp['indices'] = indices
                comp['dim'] = len(indices)
                
                # Add to handled indices
                handled_indices.update(indices)
            
            elif comp['type'] == 'encode':
                # Get indices for encoded bones
                indices = self._get_bone_indices_from_skeleton(comp_def['bone_names'], use_position=False)
                comp['indices'] = indices
                
                # Add to handled indices
                handled_indices.update(indices)

            elif comp['type'] == 'ik':
                # Directly calculate what we need - much simpler!
                bone_lengths = self._get_ik_bone_lengths(comp_def)
                joint_indices = self._get_ik_joint_indices(comp_def)
                twist_joint = comp_def.get('twist_joint', None)
                chain_parent = comp_def.get('chain_parent', None)
                bone1_forward_dir = comp_def.get('bone1_forward_dir', [0.0, 0.0, 1.0])
                bone1_up_dir = comp_def.get('bone1_up_dir', [0.0, 1.0, 0.0])
                bone2_forward_dir = comp_def.get('bone2_forward_dir', [0.0, 0.0, 1.0])
                bone2_up_dir = comp_def.get('bone2_up_dir', [0.0, 1.0, 0.0])
                
                comp['bone_lengths'] = bone_lengths
                comp['joint_indices'] = joint_indices
                comp['end_effector_joint'] = comp_def['end_effector']
                comp['chain_parent'] = chain_parent
                comp['twist_joint'] = twist_joint
                comp['bone1_forward_dir'] = bone1_forward_dir
                comp['bone1_up_dir'] = bone1_up_dir
                comp['bone2_forward_dir'] = bone2_forward_dir
                comp['bone2_up_dir'] = bone2_up_dir
                
                # Add chain info for debugging and visualization
                comp['chain_info'] = {
                    'chain_joints': comp_def['chain_joints'],
                    'end_effector': comp_def['end_effector'],
                    'bone_lengths': bone_lengths,
                    'total_reach': sum(bone_lengths)
                }
                
                # Add to handled indices
                handled_indices.update(joint_indices)
                
                # Validate we have enough bones for 2-bone IK
                if len(bone_lengths) < 2:
                    raise ValueError(f"IK chain '{name}' needs at least 2 bones, got {len(bone_lengths)}")
                
                # Use a fixed z_dim of 4 (3 position + 1 swivel)
                comp['z_dim'] = 4
        
            elif comp['type'] == 'world_preserve_after_ik':
                indices = self.skeleton.get_joint_rotation_indices(comp_def['bone_name'])
                comp['indices'] = indices
                comp['bone_name'] = comp_def['bone_name']
                comp['z_dim'] = 6  # 6D rotation

                handled_indices.update(indices)

            resolved_components[name] = comp
    
        # Find unhandled indices
        all_indices = set(range(self.pose_dim))
        self.unhandled_indices = list(all_indices - handled_indices)
        
        # Group unhandled indices by bone for better organization
        self.unhandled_bones = {}
        for idx in sorted(self.unhandled_indices):
            for bone_name, bone_indices in self.skeleton.bone_to_indices_map.items():
                if idx in bone_indices:
                    if bone_name not in self.unhandled_bones:
                        self.unhandled_bones[bone_name] = []
                    self.unhandled_bones[bone_name].append(idx)
                    break
        
        # Print unhandled bones info
        if self.unhandled_bones:
            print(f"\nUnhandled bones that will use identity rotation:")
            for bone_name, indices in self.unhandled_bones.items():
                index_range = f"{min(indices)}-{max(indices)}" if indices else "None"
                print(f"  {bone_name}: indices {index_range}")
        
        return resolved_components
    
    def _get_ik_bone_lengths(self, ik_comp_def: Dict) -> List[float]:
        """Get bone lengths for an IK chain using world positions."""
        chain_joints = ik_comp_def['chain_joints']
        bone_lengths = []
        
        # For each consecutive pair of joints in the chain
        for i in range(len(chain_joints) - 1):
            parent_joint = chain_joints[i]
            child_joint = chain_joints[i + 1]
            
            # Calculate direct distance in world space
            length = self.skeleton.calculate_distance_between_joints(parent_joint, child_joint)
            bone_lengths.append(length)
        
        # Add end effector length if specified
        if 'end_effector' in ik_comp_def:
            last_joint = chain_joints[-1]
            end_effector = ik_comp_def['end_effector']
            
            # Calculate direct distance in world space
            end_length = self.skeleton.calculate_distance_between_joints(last_joint, end_effector)
            bone_lengths.append(end_length)
        
        return bone_lengths

    def _get_ik_joint_indices(self, ik_comp_def: Dict) -> List[int]:
        """Get joint indices for an IK chain - simple and direct."""
        joint_indices = []
        
        for joint in ik_comp_def['chain_joints']:
            indices = self.skeleton.get_joint_rotation_indices(joint)
            joint_indices.extend(indices)
        
        return joint_indices
    
    def _get_bone_indices_from_skeleton(self, bone_names: List[str], use_position: bool = False) -> List[int]:
        """Get indices for a list of bone names from skeleton with better error handling."""
        indices = []
        
        for bone_name in bone_names:
            if use_position:
                if bone_name != 'body_world':
                    raise ValueError(f"Only 'body_world' has position channels, got '{bone_name}'")
                bone_indices = self.skeleton.get_joint_position_indices(bone_name)
            else:
                bone_indices = self.skeleton.get_joint_rotation_indices(bone_name)
            
            indices.extend(bone_indices)
        return indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a pose tensor into the latent space.
        
        Args:
            x: [batch_size, pose_dim] Pose tensor
            
        Returns:
            [batch_size, latent_dim] Encoded pose
        """
        batch_size = x.shape[0]
        z = torch.zeros(batch_size, self.total_z_dim, device=x.device)

         # Denormalize the pose for world position calculations
        # x_denorm = self.skeleton.denormalize_poses(x)
        x = self.skeleton.denormalize_poses(x)
        x_denorm = x.clone()
        
        # First, calculate world positions for the entire pose
        world_positions, world_rotations_dict = self.skeleton.calculate_world_positions(x_denorm, return_rotations=True)
        
        z_idx = 0
        for name, comp in self.components.items():
            if comp['type'] == 'preserve':
                indices = comp['indices']
                dim = len(indices)
                z[:, z_idx:z_idx+dim] = x[:, indices] # use z-normalized pose
                z_idx += dim

        for name, comp in self.components.items():    
            if comp['type'] == 'encode':
                indices = comp['indices']
                encoder = self.encoders[name]
                dim = comp['z_dim']
                z[:, z_idx:z_idx+dim] = encoder.encode(x[:, indices]) # use z-normalized pose
                z_idx += dim
        
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                # Get the chain origin joint world position
                origin_joint = comp['chain_info']['chain_joints'][0]
                origin_idx = self.joint_name_to_world_pos_idx[origin_joint]
                origin_pos = world_positions[:, origin_idx*3:origin_idx*3+3]  # Get 3D position

                # Get the chain elbow joint world position
                elbow_joint = comp['chain_info']['chain_joints'][1]
                elbow_idx = self.joint_name_to_world_pos_idx[elbow_joint]
                elbow_pos = world_positions[:, elbow_idx*3:elbow_idx*3+3]  # Get 3D position

                # Get the target end effector position
                end_effector_joint = comp['chain_info']['end_effector']
                end_effector_idx = self.joint_name_to_world_pos_idx[end_effector_joint]
                end_effector_pos = world_positions[:, end_effector_idx*3:end_effector_idx*3+3]  # Get 3D position

                # print(f"Origin joint '{origin_joint}' at index {origin_idx} with position {origin_pos}")
                # Get the rotations of the first two joints in the chain
                joint1_rot_6d = x_denorm[:, comp['joint_indices'][:6]]  # First 6 indices for first joint
                joint2_rot_6d = x_denorm[:, comp['joint_indices'][6:12]]  # Next 6 indices for second joint

                # Convert 6D rotations to rotation matrices
                joint1_rot = convert_6d_to_matrix(joint1_rot_6d.unsqueeze(0)).squeeze(0) # expects frames, so we need to squeeze and unsqueeze
                joint2_rot = convert_6d_to_matrix(joint2_rot_6d.unsqueeze(0)).squeeze(0)

                # Get the parent of the origin joint
                parent_joint = self.ik_chains[name].chain_parent

                # Get the world rotation for the parent of the origin joint
                parent_rot = world_rotations_dict[parent_joint]

                # Handle twist bone if defined
                twist_rot = None
                if 'twist_joint' in comp and comp['twist_joint'] is not None:
                    twist_joint = comp['twist_joint']
                    twist_indices = self.skeleton.get_joint_rotation_indices(twist_joint)
                    twist_rot_6d = x_denorm[:, twist_indices]
                    twist_rot = convert_6d_to_matrix(twist_rot_6d.unsqueeze(0)).squeeze(0)

                # Convert rotations to IK parameters
                end_effector_pos, elbow_pos, swivel = self.ik_chains[name].fk_to_ik(
                    origin_pos, joint1_rot, joint2_rot, parent_rot, twist_rot
                )

                # end_effector_pos, swivel = self.ik_chains[name].world_positions_to_ik(
                #     origin_pos, elbow_pos, end_effector_pos
                # )
                
                # Store the world space target position and swivel
                z[:, z_idx:z_idx+3] = end_effector_pos  # Store world space target
                z[:, z_idx+3:z_idx+4] = torch.clamp(swivel, -1.0, 1.0)  # Clamp swivel to [-1, 1]
                z_idx += 4

        for name, comp in self.components.items():
            if comp['type'] == 'world_preserve_after_ik':
                # Get the world space rotation for this bone
                bone_name = comp['bone_name']
                indices = comp['indices']

                # print("WORLD PRESERVE AFTER IK")
                # print(f"  Bone name: {bone_name}")
                # print(f"  Indices: {indices}")

                # Get the world rotation matrix for this bone
                world_rotation = world_rotations_dict[bone_name]

                # print(f"  World rotation matrix: {world_rotation}")

                # Store the world space rotation in 6D format
                world_rotation_6d = convert_matrix_to_6d(world_rotation.unsqueeze(0)).squeeze(0)

                # print (f"  World rotation 6D: {world_rotation_6d}")
                z[:, z_idx:z_idx+6] = world_rotation_6d
                # print("Storing identity rotation for world preserve after IK component at indices", comp['indices'], " in z indices", z_idx, "to", z_idx+6)
                # z[:, z_idx:z_idx+6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=x.device)  # Temporary identity rotation
                z_idx += 6
                
        return z
    
    def decode(self, z):
        """Decode latent space back to pose with properly positioned IK chains."""
        batch_shape = z.shape[:-1]
        output = torch.zeros(batch_shape + (self.pose_dim,), device=z.device)
        
        start_idx = 0
        
        # 1. Preserved components (direct copy)
        for name, comp in self.components.items():
            if comp['type'] == 'preserve':
                size = comp['dim']
                data = z[..., start_idx:start_idx+size]
                output[..., comp['indices']] = data
                start_idx += size
        
        # 2. Auto-encoded components
        for name, comp in self.components.items():
            if comp['type'] == 'encode':
                size = comp['z_dim']
                latent = z[..., start_idx:start_idx+size]
                decoded = self.encoders[name].decode(latent)
                output[..., comp['indices']] = decoded
                start_idx += size
        
        # # preserved components and auto-encoded components work with normalized poses. But to do IK we need unnormalized poses
        # So we denormalize everything here, and then we will normalize again at the end when we're done with IK
        # output = self.skeleton.denormalize_poses(output)

        for idx in self.unhandled_indices:
            # TODO: Uderstand what is happening here. Why does identity not work, but this combination does?
            remainder = idx % 6
            if remainder == 0:  # First element of first column (0,1,0)
                output[..., idx] = 0.0
            elif remainder == 1:  # Second element of first column
                output[..., idx] = 1.0
            elif remainder == 2:  # Third element of first column
                output[..., idx] = 0.0
            elif remainder == 3:  # First element of second column (1,0,0)
                output[..., idx] = 1.0
            elif remainder == 4:  # Second element of second column
                output[..., idx] = 0.0
            elif remainder == 5:  # Third element of second column
                output[..., idx] = 0.0

        # 3. Set identity rotations for IK joints (temporary placeholders)
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                indices = comp['joint_indices']
                if len(indices) >= 12:
                    # Set identity rotation for first joint
                    output[..., indices[0]] = 1.0  # First element of 6D rotation = [1,0,0,0,1,0]
                    output[..., indices[4]] = 1.0  # Fifth element of 6D rotation
                    
                    # Set identity rotation for second joint
                    output[..., indices[6]] = 1.0  # First element of 6D rotation
                    output[..., indices[10]] = 1.0  # Fifth element of 6D rotation
            
        # 4. Calculate world positions with pre-ik pose.
        # We need to do this so we can get the origin positions and world rotations for IK chains
        world_positions, world_rotations_dict = self.skeleton.calculate_world_positions(output, return_rotations=True)
        
        # 5. Process IK components with proper origins
        start_idx = self.preserved_dim + self.encoded_dim  # Reset index for IK components
        
        world_parent_rotations_after_ik_dict = {}

        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                # Extract position and swivel
                target_pos = z[..., start_idx:start_idx+3]
                swivel = z[..., start_idx+3:start_idx+4]
                start_idx += 4
                
                # Get chain origin joint name and find its world position
                origin_joint = comp['chain_info']['chain_joints'][0]
                origin_idx = self.joint_name_to_world_pos_idx[origin_joint]
                origin_pos = world_positions[:, origin_idx*3:origin_idx*3+3]  # Get 3D position

                # Get the chain origin world rotation
                origin_rot = world_rotations_dict[origin_joint]
                
                # Get the IK chain for this component
                ik_chain = self.ik_chains[name]
                
                # Convert IK parameters to joint rotations with local target
                local_rot1, local_rot2, elbow_pos, target_pos, world_rot1, world_rot2 = ik_chain.ik_to_fk(
                    origin_pos, 
                    target_pos, 
                    swivel * torch.pi * 0.25, # scale from [-1, 1] to radians
                    origin_rot # Supply the origin rotation to return local rotations
                )

                # Store the world rotations for later use
                world_parent_rotations_after_ik_dict[comp['chain_info']['end_effector']] = world_rot2

                # Convert joint positions to 6D rotations
                rot1_6d = convert_matrix_to_6d(local_rot1.unsqueeze(0)).squeeze(0)  # expects frames, so we need to unsqueeze and squeeze
                rot2_6d = convert_matrix_to_6d(local_rot2.unsqueeze(0)).squeeze(0)
                
                # Place results in output tensor
                indices = comp['joint_indices']
                if len(indices) >= 12:
                    output[..., indices[:6]] = rot1_6d
                    output[..., indices[6:12]] = rot2_6d

        for name, comp in self.components.items():
            if comp['type'] == 'world_preserve_after_ik':
                world_space_rot_6d = z[..., start_idx:start_idx+6]
                start_idx += 6
                bone_name = comp['bone_name']

                # print("DECODING WORLD PRESERVE AFTER IK")
                # print(f"  Bone name: {bone_name}")
                # print(f"  World space rotation 6D: {world_space_rot_6d}")
                
                # convert 6D rotation to matrix
                world_space_rot = convert_6d_to_matrix(world_space_rot_6d.unsqueeze(0)).squeeze(0)

                # print(f"  World space rotation matrix: {world_space_rot}")

                # Get the world rotation of the parent of this bone after IK - This is so we can convert world space rotations to local rotations
                parent_rot = world_parent_rotations_after_ik_dict[bone_name]
                
                # Get the inverse of the parent rotation
                parent_rot_inv = parent_rot.transpose(-1, -2)

                # Convert the world rotation to local rotation
                local_rot = torch.matmul(parent_rot_inv, world_space_rot)

                # Convert local rotation to 6D
                local_rot_6d = convert_matrix_to_6d(local_rot.unsqueeze(0)).squeeze(0)

                # Get the indices for this bone
                indices = comp['indices']
                
                # print(f"  Storing local rotation 6D at indices {indices[:6]}")
                output[..., indices[:6]] = local_rot_6d

        
        # Now we need to normalize the output pose again
        output = self.skeleton.normalize_poses(output)

        return output
    
    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking
    
    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)

    def _print_component_info(self):
        """Print detailed information about resolved components."""
        print("\n=== Pose Encoder Component Information ===")
        
        for name, comp in self.components.items():
            print(f"\n{name} ({comp['type']}):")
            
            if comp['type'] == 'preserve':
                print(f"  Indices: {comp['indices']}")
                print(f"  Dimensions: {comp['dim']}")
                
            elif comp['type'] == 'ik':
                chain_info = comp['chain_info']
                print(f"  Chain joints: {chain_info['chain_joints']}")
                print(f"  End effector: {chain_info['end_effector']}")
                print(f"  Bone lengths: {[f'{l:.3f}' for l in chain_info['bone_lengths']]}")
                print(f"  Total reach: {chain_info['total_reach']:.3f}")
                print(f"  Joint indices: {comp['joint_indices']}")
                print(f"  Dimensions: {comp['z_dim']}")
                
            elif comp['type'] == 'encode':
                print(f"  Input dimensions: {len(comp['indices'])}")
                print(f"  Latent dimensions: {comp['z_dim']}")
                if comp['indices']:
                    print(f"  Indices range: [{min(comp['indices'])}, {max(comp['indices'])}]")
                else:
                    print("  No indices found")
        
        print(f"\nTotal latent dimensions: {self.total_z_dim}")
        print(f"  Preserved: {self.preserved_dim}")
        print(f"  IK: {self.ik_dim}")
        print(f"  Encoded: {self.encoded_dim}")
