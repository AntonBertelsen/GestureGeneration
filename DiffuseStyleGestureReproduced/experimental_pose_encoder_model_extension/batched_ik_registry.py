import torch
import torch.nn.functional as F
import utils.utils as utils

class BatchedIKRegistry(torch.nn.Module):
    """
    Stores all IK chain configurations directly in optimized tensor format
    and performs operations on multiple chains simultaneously.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._eye3 = torch.eye(3, device=device)
        
        # Registry for chains (maps chain_name to chain_id)
        self.chain_registry = {}
        self.next_chain_id = 0

        self.twist_joints = []
        self.chain_parents = []
        
        # Tensors to store chain data (will grow as chains are registered)
        self.bone_lengths = torch.zeros((0, 2), device=device)
        self.bone1_forward_dirs = torch.zeros((0, 3), device=device)
        self.bone1_up_dirs = torch.zeros((0, 3), device=device)
        self.bone2_forward_dirs = torch.zeros((0, 3), device=device)
        self.bone2_up_dirs = torch.zeros((0, 3), device=device)
        
    def register_chain(self, name, l1, l2, 
                      bone1_forward_dir, bone1_up_dir,
                      bone2_forward_dir, bone2_up_dir,
                      twist_joint, chain_parent):
        """
        Register a new IK chain with the registry.
        Returns the chain_id for future reference.
        """
        if name in self.chain_registry:
            return self.chain_registry[name]
            
        chain_id = self.next_chain_id
        self.chain_registry[name] = chain_id
        self.next_chain_id += 1
        
        # Grow tensors to accommodate the new chain
        self.bone_lengths = torch.cat([
            self.bone_lengths,
            torch.tensor([[l1, l2]], device=self.device)
        ], dim=0)
        
        self.bone1_forward_dirs = torch.cat([
            self.bone1_forward_dirs,
            bone1_forward_dir.view(1, 3)
        ], dim=0)
        
        self.bone1_up_dirs = torch.cat([
            self.bone1_up_dirs,
            bone1_up_dir.view(1, 3)
        ], dim=0)
        
        self.bone2_forward_dirs = torch.cat([
            self.bone2_forward_dirs,
            bone2_forward_dir.view(1, 3)
        ], dim=0)
        
        self.bone2_up_dirs = torch.cat([
            self.bone2_up_dirs,
            bone2_up_dir.view(1, 3)
        ], dim=0)

        self.twist_joints.append(twist_joint)
        self.chain_parents.append(chain_parent)
        
        return chain_id
    
    def ik_to_fk(self, chain_ids, start_pos, target_pos, swivel_angle, parent_world_rot=None):
        """
        Batch process multiple IK chains with different configurations.
        
        Args:
            chain_ids: [...] IDs for the chains to use (can be a scalar or tensor)
            start_pos: [..., 3] Starting positions
            target_pos: [..., 3] Target positions 
            swivel_angle: [..., 1] Swivel angles
            parent_world_rot: [..., 3, 3] Optional parent rotations
            
        Returns:
            Tuple of rotations and positions
        """
        # Handle scalar chain_id case
        use_single_chain = not isinstance(chain_ids, torch.Tensor)
        if use_single_chain:
            # Use the same chain for all inputs
            chain_id = chain_ids
            batch_shape = start_pos.shape[:-1]
            chain_ids = torch.full(batch_shape, chain_id, device=start_pos.device)
        
        # Prepare bone config tensors based on chain_ids
        # Use advanced indexing to get the right config for each item
        batch_shape = start_pos.shape[:-1]
        flat_chain_ids = chain_ids.reshape(-1)
        
        # Get bone lengths for each chain
        l1 = self.bone_lengths[flat_chain_ids, 0].reshape(*batch_shape, 1)
        l2 = self.bone_lengths[flat_chain_ids, 1].reshape(*batch_shape, 1)
        
        # Get directions for each chain
        bone1_forward = self.bone1_forward_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        bone1_up = self.bone1_up_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        bone2_forward = self.bone2_forward_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        bone2_up = self.bone2_up_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        
        # Rest of calculation is the same as the batched function from before
        eps = 1e-8
        max_reach = l1 + l2
        
        # Vector calculation
        vec = target_pos - start_pos
        dist_squared = torch.sum(vec * vec, dim=-1, keepdim=True)
        dist = torch.sqrt(dist_squared.clamp(min=eps))
        
        # Handle constraints
        with torch.no_grad():
            too_far = dist > max_reach
            too_close = dist < 1e-6
            
        safe_dir = vec / dist.clamp(min=eps)
        scaled_vecs = safe_dir * max_reach
        
        default_dir = torch.zeros_like(vec)
        default_dir[..., 2] = 1.0
        default_vec = F.normalize(default_dir, dim=-1) * 1e-6
        
        vec = torch.where(too_far, scaled_vecs, vec)
        vec = torch.where(too_close, default_vec, vec)
        
        dist = torch.norm(vec, dim=-1, keepdim=True)
        dir = vec / dist.clamp(min=eps)
        
        # Law of cosines
        cos_arg = (l1**2 + dist**2 - l2**2) / (2 * l1 * dist).clamp(min=eps)
        cos_theta = torch.clamp(cos_arg, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        
        # Orthonormal frame
        cross1 = torch.cross(bone1_up, dir, dim=-1)
        right = F.normalize(cross1, dim=-1)
        up_proj = F.normalize(torch.cross(dir, right, dim=-1), dim=-1)
        
        # Swivel
        swivel_cos = torch.cos(swivel_angle)
        swivel_sin = torch.sin(swivel_angle)
        elbow_dir = swivel_cos * up_proj + swivel_sin * right
        
        # Joint positions
        joint1 = start_pos + l1 * (cos_theta * dir + torch.sin(theta) * elbow_dir)
        joint2 = start_pos + vec  # End effector
        
        # Calculate bone directions
        bone1_dir = F.normalize(joint1 - start_pos, dim=-1)
        bone2_dir = F.normalize(joint2 - joint1, dim=-1)
        
        # Calculate rotation matrices
        rot1_world = self._fast_compute_rotation_batched(bone1_forward, bone1_dir, bone1_up)
        rot2_world = self._fast_compute_rotation_batched(bone2_forward, bone2_dir, bone2_up)
        
        # Handle parent rotations
        if parent_world_rot is not None:
            # Convert world rotations to local
            rot1_local = torch.matmul(parent_world_rot.transpose(-2, -1), rot1_world)
            rot2_local = torch.matmul(rot1_world.transpose(-2, -1), rot2_world)
            return rot1_local, rot2_local, joint1, joint2, rot1_world, rot2_world
        
        return rot1_world, rot2_world, joint1, joint2
    
    def _fast_compute_rotation_batched(self, v1, v2, up_hint):
        """Vectorized rotation computation that handles arbitrary batch dimensions."""
        batch_shape = v1.shape[:-1]
        
        # Normalize inputs
        v1 = F.normalize(v1, dim=-1)
        v2 = F.normalize(v2, dim=-1)
        
        # Calculate rotation axis and angle
        cos_angle = torch.sum(v1 * v2, dim=-1).clamp(-0.9999, 0.9999)
        axis = torch.cross(v1, v2, dim=-1)
        sin_angle = torch.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axis / sin_angle
        
        # Create efficient skew-symmetric matrices
        K = torch.zeros(*batch_shape, 3, 3, device=self.device)
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        
        K[..., 0, 1], K[..., 0, 2] = -z, y
        K[..., 1, 0], K[..., 1, 2] = z, -x
        K[..., 2, 0], K[..., 2, 1] = -y, x
        
        # Calculate rotation using Rodrigues formula
        angle = torch.acos(cos_angle)
        sin_angles = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
        one_minus_cos = (1 - cos_angle).unsqueeze(-1).unsqueeze(-1)
        
        K_squared = torch.matmul(K, K)
        eye = torch.eye(3, device=self.device).reshape(*(1,)*len(batch_shape), 3, 3)
        eye = eye.expand(*batch_shape, 3, 3)
        
        result = eye + sin_angles * K + one_minus_cos * K_squared
        return result
    
    def world_positions_to_ik(self, chain_ids, start_pos, elbow_pos, end_effector_pos):
        """
        Calculate swivel angles from world joint positions.
        
        Args:
            chain_ids: [...] IDs for the chains to use
            start_pos: [..., 3] Starting positions
            elbow_pos: [..., 3] Elbow positions
            end_effector_pos: [..., 3] End effector positions
            world_rotations_dict: Dictionary of world rotations
            
        Returns:
            Dictionary with target positions and swivel angles
        """
        # Handle scalar chain_id case
        use_single_chain = not isinstance(chain_ids, torch.Tensor)
        if use_single_chain:
            chain_id = chain_ids
            batch_shape = start_pos.shape[:-1]
            chain_ids = torch.full(batch_shape, chain_id, device=start_pos.device)
        
        # Get chain config
        batch_shape = start_pos.shape[:-1]
        flat_chain_ids = chain_ids.reshape(-1)
        
        # Get bone directions
        bone1_forward = self.bone1_forward_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        bone1_up = self.bone1_up_dirs[flat_chain_ids].reshape(*batch_shape, 3)
        
        # Calculate direction vectors
        vec = end_effector_pos - start_pos
        dist = torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-8)
        dir = vec / dist
        
        # Create orthonormal frame
        cross1 = torch.cross(bone1_up, dir, dim=-1)
        right = F.normalize(cross1, dim=-1, eps=1e-8)
        up_proj = F.normalize(torch.cross(dir, right, dim=-1), dim=-1, eps=1e-8)
        
        # Project elbow positions onto plane perpendicular to dir
        to_elbow = elbow_pos - start_pos
        dots = torch.sum(to_elbow * dir, dim=-1, keepdim=True)
        to_elbow_proj = to_elbow - dots * dir
        
        # Normalize projection
        proj_norm = torch.norm(to_elbow_proj, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_proj = to_elbow_proj / proj_norm
        
        # Calculate swivel using projected elbow
        cos_swivel = torch.sum(normalized_proj * up_proj, dim=-1, keepdim=True)
        sin_swivel = torch.sum(normalized_proj * right, dim=-1, keepdim=True)
        swivel = torch.atan2(sin_swivel, cos_swivel)
        
        # Scale swivel from radians to normalized range [-1, 1]
        normalized_swivel = swivel / (torch.pi * 0.25)
        normalized_swivel = torch.clamp(normalized_swivel, -1.0, 1.0)
        
        return {
            'target_positions': end_effector_pos,
            'swivel_angles': normalized_swivel
        }
    
    
    def process_all_chains_decode(self, z, output, world_positions, world_rotations_dict, 
                              components, start_idx, joint_name_to_world_pos_idx,
                              skeleton, chain_ids):
        """
        Process all IK chains at once in a single batched operation for maximum performance.
        Reads directly from z and writes directly to output to minimize data transfers.
        
        Args:
            z: Latent vector with IK parameters
            output: Output pose tensor to write results to
            world_positions: Pre-calculated world positions
            world_rotations_dict: Dictionary of world rotations
            components: Component definitions
            start_idx: Starting index in z for IK data
            joint_name_to_world_pos_idx: Mapping from joint names to world position indices
            skeleton: Skeleton data for normalization
            chain_ids: Dictionary mapping component names to chain IDs
            
        Returns:
            Updated start_idx, world parent rotations dict
        """
        # Get all IK components
        ik_components = [(name, comp) for name, comp in components.items() if comp['type'] == 'ik']
        if not ik_components:
            return start_idx, {}
        
        batch_shape = z.shape[:-1]
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim
        world_parent_rotations_dict = {}
        
        # Pre-allocate tensors for all chains
        num_chains = len(ik_components)
        all_chain_ids = []
        all_origins = []
        all_targets = []
        all_swivels = []
        all_parent_rots = []
        all_indices = []      # To map results back to components
        all_end_names = []    # To track end effector names
        
        # Track source indices to map back into output tensor
        source_info = []  # List of (name, indices, component_idx)
        
        # 1. GATHER DATA FOR ALL CHAINS
        for chain_idx, (name, comp) in enumerate(ik_components):
            # Extract parameters from latent vector
            normalized_target_pos = z[..., start_idx:start_idx+3]
            swivel = z[..., start_idx+3:start_idx+4] 
            start_idx += 4
            
            # Denormalize target position
            end_effector_joint = comp['chain_info']['end_effector']
            end_effector_idx = joint_name_to_world_pos_idx[end_effector_joint]
            mean = skeleton.world_pos_mean_pose[end_effector_idx*3:end_effector_idx*3+3]
            std = skeleton.world_pos_std_pose[end_effector_idx*3:end_effector_idx*3+3]
            target_pos = normalized_target_pos * std + mean
            
            # Get origin position and rotation
            origin_joint = comp['chain_info']['chain_joints'][0]
            origin_idx = joint_name_to_world_pos_idx[origin_joint]
            origin_pos = world_positions[..., origin_idx*3:origin_idx*3+3]
            
            parent_rot = world_rotations_dict[self.chain_parents[chain_idx]]
            
            # Append to batch tensors
            chain_id = torch.tensor(chain_ids[name], device=z.device, dtype=torch.long)
            chain_id_repeated = chain_id.expand(batch_size)
            all_chain_ids.append(chain_id_repeated)

            all_origins.append(origin_pos)
            all_targets.append(target_pos)
            all_swivels.append(swivel * torch.pi * 0.25)  # Convert to radians
            all_parent_rots.append(parent_rot)
            
            # Store component info for mapping results
            source_info.append((name, comp['joint_indices'], end_effector_joint))
        
        # 2. CONCATENATE ALL TENSORS ALONG BATCH DIMENSION
        # This is the key optimization - create a single giant batch of all chains
        flat_chain_ids = torch.cat(all_chain_ids)
        flat_origins = torch.cat([pos.reshape(-1, 3) for pos in all_origins])
        flat_targets = torch.cat([pos.reshape(-1, 3) for pos in all_targets])
        flat_swivels = torch.cat([ang.reshape(-1, 1) for ang in all_swivels]) 
        flat_parent_rots = torch.cat([rot.reshape(-1, 3, 3) for rot in all_parent_rots])
        
        # 3. PERFORM BATCHED IK CALCULATION
        # Process all chains at once in a single operation
        flat_results = self.ik_to_fk(
            flat_chain_ids, 
            flat_origins, 
            flat_targets, 
            flat_swivels,
            flat_parent_rots
        )
        
        # Unpack results
        flat_rot1_local, flat_rot2_local, _, _, _, flat_world_rot2 = flat_results
        
        # 4. DISTRIBUTE RESULTS BACK TO COMPONENTS
        items_per_chain = flat_origins.shape[0] // num_chains
        
        for i, (name, indices, end_effector) in enumerate(source_info):
            # Extract this chain's results
            start_item = i * items_per_chain
            end_item = (i + 1) * items_per_chain
            
            # Get local rotations
            chain_rot1 = flat_rot1_local[start_item:end_item]
            chain_rot2 = flat_rot2_local[start_item:end_item]
            chain_world_rot2 = flat_world_rot2[start_item:end_item]
            
            # Reshape back to original batch dimensions
            local_rot1 = chain_rot1.reshape(*batch_shape, 3, 3)
            local_rot2 = chain_rot2.reshape(*batch_shape, 3, 3)
            world_rot2 = chain_world_rot2.reshape(*batch_shape, 3, 3)
            
            # Store world rotation for later use with world_preserve_after_ik
            world_parent_rotations_dict[end_effector] = world_rot2
            
            # Convert to 6D and write directly to output
            rot1_6d = utils.convert_matrix_to_6d(local_rot1)
            rot2_6d = utils.convert_matrix_to_6d(local_rot2)
            
            output[..., indices[:6]] = rot1_6d.to(output.dtype)
            output[..., indices[6:12]] = rot2_6d.to(output.dtype)
        
        return start_idx, world_parent_rotations_dict