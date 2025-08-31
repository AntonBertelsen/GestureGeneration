import time
import torch
import torch.nn as nn
from typing import Dict, List
from pose_encoder.ik_two_bone import IKChain2Bone
from pose_encoder.batched_ik_registry import BatchedIKRegistry
import utils.utils as utils

from pose_encoder.pose_encoder import PoseEncoder

class SimpleEncoder(nn.Module):
    """Simple VAE encoder/decoder."""
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        hidden = max(32, min(128, input_dim))
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     nn.GELU(),
        #     nn.Linear(hidden, hidden//2),
        #     nn.GELU()
        # )
        
        self.mu = nn.Linear(input_dim, z_dim)
        self.logvar = nn.Linear(input_dim, z_dim)
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, hidden//2),
        #     nn.GELU(),
        #     nn.Linear(hidden//2, hidden),
        #     nn.GELU(),
        #     nn.Linear(hidden, input_dim)
        # )
        self.decoder = nn.Linear(z_dim, input_dim)
    
    def encode(self, x, return_logvar=False):
        # h = self.encoder(x)
        mu = self.mu(x)
        if return_logvar:
            return mu, self.logvar(x)
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

class AdvancedPoseEncoder(PoseEncoder):
    def __init__(self, pose_dim=345, component_definitions=None, device=None, skeleton=None):
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

        # Replace individual IK chains with registry
        self.ik_registry = BatchedIKRegistry(device)
        self.chain_ids = {}
        
        # Register each IK chain
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
                
                # Register with the registry
                chain_id = self.ik_registry.register_chain(
                    name=name,
                    l1=bone1_len,
                    l2=bone2_len,
                    bone1_forward_dir=bone1_forward_dir,
                    bone1_up_dir=bone1_up_dir,
                    bone2_forward_dir=bone2_forward_dir,
                    bone2_up_dir=bone2_up_dir,
                    twist_joint=twist_joint,
                    chain_parent=chain_parent
                )
                
                # Store chain ID for later reference
                self.chain_ids[name] = chain_id
        
        # Print resolved components for debugging
        # self._print_component_info()

        # Enhanced hyperparameters including skeleton info
        # Store full configuration for reproducibility
        self.hyperparameter_dict_to_WnB_tracking = {
            # Core parameters
            "type": "advanced_pose_encoder",
            "pose_dim": pose_dim,
            # Component configuration (critical for recreation)
            "component_definitions": self.component_definitions
        }    
        
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
        # if self.unhandled_bones:
        #     print(f"\nUnhandled bones that will use identity rotation:")
        #     for bone_name, indices in self.unhandled_bones.items():
        #         index_range = f"{min(indices)}-{max(indices)}" if indices else "None"
        #         print(f"  {bone_name}: indices {index_range}")
        
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
    
    def encode(self, x: torch.Tensor, using_normalized_poses = True) -> torch.Tensor:
        """
        Encode a pose tensor into the latent space.
        
        Args:
            x: [batch_size, num_frames, pose_dim] Pose tensor
            
        Returns:
            [batch_size, num_frames, latent_dim] Encoded pose
        """
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        z = torch.zeros(batch_size, num_frames, self.total_z_dim, device=x.device, dtype=x.dtype)

        if using_normalized_poses:
            # Denormalize the pose for world position calculations
            x_denorm = self.skeleton.denormalize_poses(x)
        else: 
            x_denorm = x
            x = self.skeleton.normalize_poses(x)  # Normalize for encoding
        
        # First, calculate world positions for the entire pose
        world_positions, world_rotations_dict = self.skeleton.calculate_world_positions(x_denorm, return_rotations=True)
        
        z_idx = 0
        for name, comp in self.components.items():
            if comp['type'] == 'preserve':
                indices = comp['indices']
                dim = len(indices)
                z[:, :, z_idx:z_idx+dim] = x[:, :, indices] # use z-normalized pose
                z_idx += dim

        for name, comp in self.components.items():    
            if comp['type'] == 'encode':
                indices = comp['indices']
                encoder = self.encoders[name]
                dim = comp['z_dim']
                z[:, :, z_idx:z_idx+dim] = encoder.encode(x[:, :, indices]) # use z-normalized pose
                z_idx += dim

        # Process IK components
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                # Get joint positions directly from world positions
                origin_joint = comp['chain_info']['chain_joints'][0]
                origin_idx = self.joint_name_to_world_pos_idx[origin_joint]
                origin_pos = world_positions[:, :, origin_idx*3:origin_idx*3+3]
                
                elbow_joint = comp['chain_info']['chain_joints'][1]
                elbow_idx = self.joint_name_to_world_pos_idx[elbow_joint]
                elbow_pos = world_positions[:, :, elbow_idx*3:elbow_idx*3+3]
                
                end_effector_joint = comp['chain_info']['end_effector']
                end_effector_idx = self.joint_name_to_world_pos_idx[end_effector_joint]
                end_effector_pos = world_positions[:, :, end_effector_idx*3:end_effector_idx*3+3]
                
                # Get chain ID for this component
                chain_id = self.chain_ids[name]
                
                # Calculate IK parameters using the batched registry
                ik_results = self.ik_registry.world_positions_to_ik(
                    chain_id,
                    origin_pos,
                    elbow_pos,
                    end_effector_pos
                )
                
                # Extract results
                target_pos = ik_results['target_positions']
                swivel = ik_results['swivel_angles']
                
                # Z-normalize end effector position
                mean = self.skeleton.world_pos_mean_pose[end_effector_idx*3:end_effector_idx*3+3]
                std = self.skeleton.world_pos_std_pose[end_effector_idx*3:end_effector_idx*3+3]
                normalized_target_pos = (target_pos - mean) / std
                
                # Store in latent vector
                z[:, :, z_idx:z_idx+3] = normalized_target_pos
                z[:, :, z_idx+3:z_idx+4] = swivel
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
                world_rotation_6d = utils.convert_matrix_to_6d(world_rotation)

                # print (f"  World rotation 6D: {world_rotation_6d}")
                z[:, :, z_idx:z_idx+6] = world_rotation_6d
                # print("Storing identity rotation for world preserve after IK component at indices", comp['indices'], " in z indices", z_idx, "to", z_idx+6)
                # z[:, z_idx:z_idx+6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=x.device)  # Temporary identity rotation
                z_idx += 6
                
        return z
    
    def decode(self, z):
        """Decode latent space back to pose with properly positioned IK chains."""
        batch_shape = z.shape[:-1]

        profiling_stages = {}
        start = time.time()

        output = torch.zeros(batch_shape + (self.pose_dim,), device=z.device, dtype=z.dtype)
        
        profiling_stages['initialization'] = time.time() - start
        start = time.time()
        
        start_idx = 0
        # 1. Preserved components (direct copy)
        for name, comp in self.components.items():
            if comp['type'] == 'preserve':
                size = comp['dim']
                data = z[..., start_idx:start_idx+size]
                output[..., comp['indices']] = data
                start_idx += size

        profiling_stages['preserved_components'] = time.time() - start
        start = time.time()

        # 2. Auto-encoded components
        for name, comp in self.components.items():
            if comp['type'] == 'encode':
                size = comp['z_dim']
                latent = z[..., start_idx:start_idx+size]
                decoded = self.encoders[name].decode(latent).to(z.dtype)
                output[..., comp['indices']] = decoded
                start_idx += size

        profiling_stages['auto_encoded_components'] = time.time() - start
        start = time.time()

        # preserved components and auto-encoded components work with normalized poses. But to do IK we need unnormalized poses
        # So we denormalize everything here, and then we will normalize again at the end when we're done with IK
        output = self.skeleton.denormalize_poses(output)

        profiling_stages['denormalization'] = time.time() - start
        start = time.time()

        # TODO: Find a better way to do this (future you here - dont)
        for idx in self.unhandled_indices:
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

        # if self.unhandled_indices:  # Only process if there are unhandled indices
            # Since unhandled indices always occur in batches of 6,
            # we can efficiently set the pattern [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
            # Note: positions 0, 2, 4, 5 will be 0.0 by default from initialization
            
            # For all indices where remainder is 1 (second element of each batch)
            # ones_indices = [idx for idx in self.unhandled_indices if idx % 6 == 1 or idx % 6 == 3]
            # print(ones_indices)
            # output[..., ones_indices] = 1.0

        profiling_stages['unhandled_indices'] = time.time() - start
        start = time.time()

        # 3. Set identity rotations for IK joints (temporary placeholders)
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                indices = comp['joint_indices']
                output[..., indices] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=z.device, dtype=z.dtype)  # Temporary identity rotation

        profiling_stages['identity_rotations'] = time.time() - start
        start = time.time()

        # 4. Calculate world positions with pre-ik pose.
        # We need to do this so we can get the origin positions and world rotations for IK chains

        # get every ik chain's origin joint name
        joints_to_calculate_world_positions_for = []
        for name, comp in self.components.items():
            if comp['type'] == 'ik':
                origin_joint = comp['chain_info']['chain_joints'][0]
                joints_to_calculate_world_positions_for.append(origin_joint)
                # print("ik chain origin joint:", origin_joint)

        # print("Joints to calculate for:", joints_to_calculate_world_positions_for)

        world_positions, world_rotations_dict = self.skeleton.calculate_world_positions_new(output, joints_to_calculate=joints_to_calculate_world_positions_for, return_rotations=True)

        # print("world rotations dict:", world_rotations_dict)
        # print("world positions shape:", world_positions.shape)

        profiling_stages['world_positions'] = time.time() - start
        start = time.time()

        # 5. Process IK components with proper origins
        start_idx = self.preserved_dim + self.encoded_dim  # Reset index for IK components

        # In the decode method, replace IK processing:
        # Process all IK components in a single batch operation
        start_idx = self.preserved_dim + self.encoded_dim
        start_idx, world_parent_rotations_after_ik_dict = self.ik_registry.process_all_chains_decode(
            z=z,
            output=output,
            world_positions=world_positions,
            world_rotations_dict=world_rotations_dict,
            components=self.components,
            start_idx=start_idx,
            joint_name_to_world_pos_idx=self.joint_name_to_world_pos_idx,
            skeleton=self.skeleton,
            chain_ids=self.chain_ids
        )

        profiling_stages['ik_processing'] = time.time() - start
        start = time.time()

        for name, comp in self.components.items():
            if comp['type'] == 'world_preserve_after_ik':
                world_space_rot_6d = z[..., start_idx:start_idx+6]
                start_idx += 6
                bone_name = comp['bone_name']
                
                # convert 6D rotation to matrix
                world_space_rot = utils.convert_6d_to_matrix(world_space_rot_6d)

                # Get the world rotation of the parent of this bone after IK - This is so we can convert world space rotations to local rotations
                parent_rot = world_parent_rotations_after_ik_dict[bone_name]
                
                # Get the inverse of the parent rotation
                parent_rot_inv = parent_rot.transpose(-1, -2)

                # Convert the world rotation to local rotation
                local_rot = torch.matmul(parent_rot_inv, world_space_rot)

                # Convert local rotation to 6D
                local_rot_6d = utils.convert_matrix_to_6d(local_rot)

                # Get the indices for this bone
                indices = comp['indices']
                
                # print(f"  Storing local rotation 6D at indices {indices[:6]}")
                output[..., indices[:6]] = local_rot_6d.to(z.dtype)

        profiling_stages['world_preserve_after_ik'] = time.time() - start

        # Now we need to normalize the output pose again
        output = self.skeleton.normalize_poses(output)

        profiling_stages['final_normalization'] = time.time() - start

        # Print profiling information
        print("\n=== Pose Encoder Profiling Information ===")
        for stage, duration in profiling_stages.items():
            print(f"{stage:30}: {duration * 1000:.2f} ms")

        return output
    
    def construct_component_weighting_vector(self, category_weighting: Dict[str, float]) -> torch.Tensor:
        """
        Construct a weighting vector for the components based on the provided category weights.
        
        Args:
            category_weighting: Dictionary mapping component categories to their weights.
        
        Returns:
            A tensor of shape (total_z_dim,) containing the weights for each component.
        """
        weights = torch.ones(self.total_z_dim, device=self.device)
        start_idx = 0
        
        for name, comp in self.components.items():
            if comp['type'] in category_weighting:
                weight = category_weighting[comp['type']]
                size = comp['z_dim']
                weights[start_idx:start_idx+size] = weight
                start_idx += size
        return weights

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

    @staticmethod
    def load_from_checkpoint(checkpoint_name, device = utils.get_device()):
        checkpoint_path = f"pose_encoder/models/{checkpoint_name}.pth"
            
        # Load the saved data
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # New format with complete configuration
        state_dict = checkpoint['state_dict']
        pose_dim = checkpoint['hyperparameters']['pose_dim']
        component_definitions = checkpoint['hyperparameters']['component_definitions']
        skeleton = checkpoint['skeleton']
        
        # Create new model instance
        pose_encoder_model = AdvancedPoseEncoder(
            pose_dim                = pose_dim,
            component_definitions   = component_definitions,
            device                  = device,
            skeleton                = skeleton
        )

        # Add the checkpoint name to the hyperparameter tracking
        pose_encoder_model.hyperparameter_dict_to_WnB_tracking['checkpoint_name'] = checkpoint_name
        
        # Load the state dict
        pose_encoder_model.load_state_dict(state_dict)
            
        return pose_encoder_model