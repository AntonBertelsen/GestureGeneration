import torch
import torch.nn.functional as F

class IKChain2Bone:
    def __init__(self, l1, l2, device,
                chain_parent = None,
                bone1_forward_dir=torch.tensor([1.0, 0.0, 0.0]), 
                bone1_up_dir=torch.tensor([0.0, 1.0, 0.0]),
                bone2_forward_dir=torch.tensor([1.0, 0.0, 0.0]),
                bone2_up_dir=torch.tensor([0.0, 1.0, 0.0]),
                twist_joint=None
    ):
        self.l1 = l1
        self.l2 = l2
        self.chain_parent = chain_parent
        self.twist_joint = twist_joint
        self.device = device
        self.bone1_forward_dir = F.normalize(bone1_forward_dir, dim=0).to(device)
        self.bone1_up_dir = F.normalize(bone1_up_dir, dim=0).to(device)
        self.bone2_forward_dir = F.normalize(bone2_forward_dir, dim=0).to(device)
        self.bone2_up_dir = F.normalize(bone2_up_dir, dim=0).to(device)

    def ik(self, start_pos, target_pos, swivel_angle):
        """
        Perform 2-bone IK and return joint positions.
        
        Args:
            start_pos: [B, 3] Starting position of the chain
            target_pos: [B, 3] Target position for end effector
            swivel_angle: [B, 1] or [B] Swivel angle in radians
            
        Returns:
            joint1: [B, 3] Middle joint position
            joint2: [B, 3] End effector position
        """
        batch_size = start_pos.shape[0]
    
        # Calculate vector and distance with safety clamp
        vec = target_pos - start_pos
        
        # Use a safer norm calculation with min bound
        eps = 1e-8
        dist_squared = torch.sum(vec * vec, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared.clamp(min=eps))
        max_reach = self.l1 + self.l2
        
        # Create safe masks (no gradients through the conditions)
        with torch.no_grad():
            too_far = dist > max_reach
            too_close = dist < 1e-6
        
        # Handle out-of-range cases with stable operations
        # For vectors that are too far
        safe_dir = F.normalize(vec, dim=1, eps=eps)
        scaled_vecs = safe_dir * max_reach
        
        # For vectors that are too close (degenerate case)
        default_dir = torch.zeros_like(vec)
        default_dir[:, 2] = 1.0
        default_vec = F.normalize(default_dir, dim=1, eps=eps) * 1e-6
        
        # Apply masks one at a time
        vec = torch.where(too_far, scaled_vecs, vec)
        vec = torch.where(too_close, default_vec, vec)
        
        # Recalculate distance and direction safely
        dist_squared = torch.sum(vec * vec, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared.clamp(min=eps))
        dir = F.normalize(vec, dim=1, eps=eps)
        
        # Law of cosines with safety bounds
        cos_arg = (self.l1**2 + dist**2 - self.l2**2) / (2 * self.l1 * dist).clamp(min=eps)
        cos_theta = torch.clamp(cos_arg, -0.999, 0.999)  # Avoid exact -1 and 1 for acos
        theta = torch.acos(cos_theta)
        
        # Safe orthonormal frame calculation
        up_dir_batch = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
        
        # Create safe cross products
        cross1 = torch.cross(up_dir_batch, dir, dim=1)
        right_norm = torch.norm(cross1, dim=1, keepdim=True).clamp(min=eps)
        right = cross1 / right_norm
        
        cross2 = torch.cross(dir, right, dim=1)
        up_norm = torch.norm(cross2, dim=1, keepdim=True).clamp(min=eps)
        up_proj = cross2 / up_norm

        # Swivel direction - 
        if swivel_angle.dim() == 1:
            swivel_angle = swivel_angle.unsqueeze(1)
        elbow_dir = torch.cos(swivel_angle) * up_proj + torch.sin(swivel_angle) * right

        # Elbow position
        joint1 = start_pos + self.l1 * (
            cos_theta * dir + torch.sin(theta) * elbow_dir
        )
        joint2 = start_pos + vec  # Clamped end

        return joint1, joint2

    def vec_to_rotmat(self, direction, use_bone1=True):
        """
        Align forward_dir (default +X) to `direction`.
        Returns a rotation matrix [B, 3, 3].
        """
        batch_size = direction.shape[0]
        direction = F.normalize(direction, dim=1)
        
        up_dir = self.bone1_up_dir if use_bone1 else self.bone2_up_dir
        up_dir_batch = up_dir.unsqueeze(0).expand(batch_size, 3)
        right = F.normalize(torch.cross(up_dir_batch, direction, dim=1), dim=1)
        up = F.normalize(torch.cross(direction, right, dim=1), dim=1)
        
        return torch.stack([direction, up, right], dim=2)

    def fk_rotations(self, start_pos, joint1_pos, joint2_pos):
        """
        Compute rotation matrices for each bone from joint positions,
        taking into account rest pose orientations.
        """
        batch_size = start_pos.shape[0]
        
        # Current bone directions (normalized)
        bone1_dir = F.normalize(joint1_pos - start_pos, dim=1)
        bone2_dir = F.normalize(joint2_pos - joint1_pos, dim=1)
        
        # Get rotation matrices for bone1
        bone1_rest = self.bone1_forward_dir.unsqueeze(0).expand(batch_size, 3)
        rot1 = self.rotation_between_vectors(bone1_rest, bone1_dir, 
                                            self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3))
        
        # Get rotation matrices for bone2
        bone2_rest = self.bone2_forward_dir.unsqueeze(0).expand(batch_size, 3)
        rot2 = self.rotation_between_vectors(bone2_rest, bone2_dir,
                                            self.bone2_up_dir.unsqueeze(0).expand(batch_size, 3))
        
        return rot1, rot2

    def rotation_between_vectors(self, v1, v2, up_hint):
        """
        Create rotation matrix that rotates from vector v1 to vector v2
        while preserving appropriate up direction - vectorized version
        """
        batch_size = v1.shape[0]
        
        # Normalize vectors
        v1 = F.normalize(v1, dim=1)
        v2 = F.normalize(v2, dim=1)
        
        # Compute rotation axis and angle
        cos_angle = torch.sum(v1 * v2, dim=1).clamp(-1.0, 1.0)
        
        # Create result tensor (identity matrices)
        result = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, 3, 3).clone()
        
        # Compute rotation axis (normalized cross product)
        axis = torch.cross(v1, v2, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        
        # Create masks for different cases
        near_parallel = cos_angle.abs() > 0.99999
        non_parallel = ~near_parallel
        valid_axis = (axis_norm > 1e-6).squeeze(-1)
        valid_non_parallel = non_parallel & valid_axis
        
        # Handle non-parallel cases (where cross product is valid)
        if valid_non_parallel.any():
            # Extract valid indices
            valid_indices = torch.where(valid_non_parallel)[0]
            
            # Extract data for valid cases
            valid_cos = cos_angle[valid_indices]
            valid_axis = axis[valid_indices] / axis_norm[valid_indices]
            valid_angles = torch.acos(valid_cos)
            
            # Create batch of cross-product matrices
            K = torch.zeros(len(valid_indices), 3, 3, device=self.device)
            x, y, z = valid_axis[:, 0], valid_axis[:, 1], valid_axis[:, 2]
            
            # Fill cross product matrices for all valid cases at once
            K[:, 0, 1] = -z
            K[:, 0, 2] = y
            K[:, 1, 0] = z
            K[:, 1, 2] = -x
            K[:, 2, 0] = -y
            K[:, 2, 1] = x
            
            # Rodrigues formula: I + sin(a)*K + (1-cos(a))*K^2
            eye = torch.eye(3, device=self.device).unsqueeze(0).expand(len(valid_indices), 3, 3)
            sin_angles = torch.sin(valid_angles).unsqueeze(-1).unsqueeze(-1)
            cos_angles = torch.cos(valid_angles).unsqueeze(-1).unsqueeze(-1)
            K_squared = torch.matmul(K, K)
            
            # Compute result for valid indices
            valid_results = eye + sin_angles * K + (1 - cos_angles) * K_squared
            
            # Assign back to result tensor
            result[valid_indices] = valid_results
        
        # Handle anti-parallel case
        anti_parallel = near_parallel & (cos_angle < 0)
        if anti_parallel.any():
            # Extract indices where vectors are anti-parallel
            anti_indices = torch.where(anti_parallel)[0]
            
            # Extract relevant vectors
            anti_v1 = v1[anti_indices]
            anti_up = up_hint[anti_indices]
            
            # Find perpendicular axes
            perp_axes = torch.cross(anti_v1, anti_up)
            perp_norms = torch.norm(perp_axes, dim=1, keepdim=True)
            
            # Create mask for valid perpendicular axes
            valid_perp = (perp_norms > 1e-6).squeeze(-1)
            
            if valid_perp.any():
                # Get indices where perpendicular axis is valid
                valid_perp_indices = anti_indices[valid_perp]
                
                # Normalize axes
                valid_axes = perp_axes[valid_perp] / perp_norms[valid_perp]
                x, y, z = valid_axes[:, 0], valid_axes[:, 1], valid_axes[:, 2]
                
                # Create rotation matrices (180° rotation around axis)
                R = torch.zeros(len(valid_perp_indices), 3, 3, device=self.device)
                
                # Fill rotation matrices
                R[:, 0, 0] = 1 - 2*(y*y + z*z)
                R[:, 0, 1] = 2*(x*y)
                R[:, 0, 2] = 2*(x*z)
                R[:, 1, 0] = 2*(x*y)
                R[:, 1, 1] = 1 - 2*(x*x + z*z)
                R[:, 1, 2] = 2*(y*z)
                R[:, 2, 0] = 2*(x*z)
                R[:, 2, 1] = 2*(y*z)
                R[:, 2, 2] = 1 - 2*(x*x + y*y)
                
                # Assign to result
                result[valid_perp_indices] = R
        
        return result

    def fk_positions(self, start_pos, rot1, rot2, parent_world_rot=None):
        """
        Given rotations and start_pos, reconstruct joint positions.
        
        Args:
            start_pos: [B, 3] Starting position
            rot1: [B, 3, 3] Local rotation matrix for first bone
            rot2: [B, 3, 3] Local rotation matrix for second bone
            parent_rot: [B, 3, 3] World rotation of parent (optional)
            
        Returns:
            joint1: [B, 3] Middle joint position
            joint2: [B, 3] End joint position
        """
        batch_size = start_pos.shape[0]
        
        # Default to identity matrix if parent_rot not provided
        if parent_world_rot is not None:
            # Convert local rotations to world rotations (following skeleton.py approach)
            rot1_world = torch.matmul(parent_world_rot, rot1)
            rot2_world = torch.matmul(rot1_world, rot2)
        else:
            # If no parent rotation, use local rotations as world rotations
            rot1_world = rot1
            rot2_world = rot2
        
        # Create bone offset vectors (like skeleton.py's static offsets)
        bone1_offset = self.bone1_forward_dir.unsqueeze(0).expand(batch_size, 3) * self.l1
        bone2_offset = self.bone2_forward_dir.unsqueeze(0).expand(batch_size, 3) * self.l2

        # Calculate first joint position using parent rotation
        rotated_offset1 = torch.matmul(rot1_world, bone1_offset.unsqueeze(-1)).squeeze(-1)
        joint1 = start_pos + rotated_offset1
        
        # Calculate second joint position using first joint's world rotation
        rotated_offset2 = torch.matmul(rot2_world, bone2_offset.unsqueeze(-1)).squeeze(-1)
        joint2 = joint1 + rotated_offset2
        
        return joint1, joint2

    # def ik_to_fk(self, start_pos, target_pos, swivel_angle, parent_world_rot=None):
    #     """Optimized version of IK to FK conversion"""
    #     batch_size = start_pos.shape[0]
        
    #     # Calculate IK joint positions - this part can be optimized further
    #     joint1, joint2 = self.ik(start_pos, target_pos, swivel_angle)
        
    #     # Optimize rotation calculation by computing both bone directions at once
    #     bone1_dir = joint1 - start_pos
    #     bone2_dir = joint2 - joint1
        
    #     # Normalize both bone directions in one operation
    #     stacked_dirs = torch.cat([bone1_dir, bone2_dir], dim=0)
    #     stacked_dirs_norm = torch.norm(stacked_dirs, dim=1, keepdim=True).clamp(min=1e-8)
    #     stacked_dirs = stacked_dirs / stacked_dirs_norm
    #     bone1_dir, bone2_dir = stacked_dirs.chunk(2, dim=0)
        
    #     # Prepare rest pose directions as batched tensors
    #     bone1_rest = self.bone1_forward_dir.unsqueeze(0).expand(batch_size, 3)
    #     bone2_rest = self.bone2_forward_dir.unsqueeze(0).expand(batch_size, 3)
    #     bone1_up = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
    #     bone2_up = self.bone2_up_dir.unsqueeze(0).expand(batch_size, 3)
        
    #     # Compute rotations more efficiently (replacing rotation_between_vectors)
    #     rot1_world = self._fast_compute_rotation(bone1_rest, bone1_dir, bone1_up)
    #     rot2_world = self._fast_compute_rotation(bone2_rest, bone2_dir, bone2_up)
        
    #     # Handle parent rotation if provided
    #     if parent_world_rot is not None:
    #         # Use batch matrix multiply for all rotations at once
    #         rot1_local = torch.bmm(parent_world_rot.transpose(-2, -1), rot1_world)
    #         rot2_local = torch.bmm(rot1_world.transpose(-2, -1), rot2_world)
    #         return rot1_local, rot2_local, joint1, joint2, rot1_world, rot2_world

    #     return rot1_world, rot2_world, joint1, joint2

    
    def _fast_compute_rotation(self, v1, v2, up_hint):
        """Faster version of rotation_between_vectors for common cases"""
        batch_size = v1.shape[0]
        
        # Compute dot products for all vectors at once
        cos_angle = torch.sum(v1 * v2, dim=1).clamp(-0.9999, 0.9999)
        
        # Only handle the most common case - non-parallel vectors
        # This avoids expensive branching logic in the original function
        axis = torch.cross(v1, v2, dim=1)
        sin_angle = torch.norm(axis, dim=1)
        
        # Normalize axis all at once
        axis = axis / sin_angle.unsqueeze(-1).clamp(min=1e-8)
        
        # Compute rotation matrix using Rodrigues formula efficiently
        K = torch.zeros(batch_size, 3, 3, device=self.device)
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # Fill cross product matrices in one go using advanced indexing
        K[:, 0, 1] = -z
        K[:, 0, 2] = y
        K[:, 1, 0] = z
        K[:, 1, 2] = -x
        K[:, 2, 0] = -y
        K[:, 2, 1] = x
        
        # Vectorize Rodrigues formula: I + sin(θ)*K + (1-cos(θ))*K²
        angle = torch.acos(cos_angle)
        sin_angles = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
        one_minus_cos = (1 - cos_angle).unsqueeze(-1).unsqueeze(-1)
        
        # Compute K² for all matrices at once
        K_squared = torch.bmm(K, K)
        
        # Compute result using broadcasting
        eye = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, 3, 3)
        result = eye + sin_angles * K + one_minus_cos * K_squared
        
        return result


    def ik_to_fk(self, start_pos, target_pos, swivel_angle, parent_world_rot=None):
        """Optimized version that avoids redundant calculations"""
        return self.combined_ik_rotations(start_pos, target_pos, swivel_angle, parent_world_rot)


    def combined_ik_rotations(self, start_pos, target_pos, swivel_angle, parent_world_rot=None):
        """
        Combined IK and rotation calculation that avoids redundant computations.
        Returns everything in one pass: joint positions and rotations.
        """
        batch_size = start_pos.shape[0]
        eps = 1e-8
        
        # ---- IK CALCULATION ----
        # Calculate vector and distance with safety clamp
        vec = target_pos - start_pos
        
        # Compute distance and handle edge cases in one pass
        dist_squared = torch.sum(vec * vec, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared.clamp(min=eps))
        max_reach = self.l1 + self.l2
        
        # Create masks for edge cases (no gradients through conditions)
        with torch.no_grad():
            too_far = dist > max_reach
            too_close = dist < 1e-6
        
        # Handle edge cases, normalizing only once
        safe_dir = vec / dist.clamp(min=eps)
        scaled_vecs = safe_dir * max_reach
        
        # Handle too_close case (degenerate)
        default_dir = torch.zeros_like(vec)
        default_dir[:, 2] = 1.0
        default_dir_norm = torch.norm(default_dir, dim=1, keepdim=True).clamp(min=eps)
        default_vec = default_dir / default_dir_norm * 1e-6
        
        # Apply masks
        vec = torch.where(too_far, scaled_vecs, vec)
        vec = torch.where(too_close, default_vec, vec)
        
        # Recalculate final distance and direction (necessary for accuracy)
        dist_squared = torch.sum(vec * vec, dim=1, keepdim=True)
        dist = torch.sqrt(dist_squared.clamp(min=eps))
        dir = vec / dist.clamp(min=eps)
        
        # Law of cosines
        cos_arg = (self.l1**2 + dist**2 - self.l2**2) / (2 * self.l1 * dist).clamp(min=eps)
        cos_theta = torch.clamp(cos_arg, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        
        # Create orthonormal frame - we'll reuse these for rotations
        up_dir_batch = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
        
        # Cross products - reused for both IK and rotation calculation
        cross1 = torch.cross(up_dir_batch, dir, dim=1)
        right_norm = torch.norm(cross1, dim=1, keepdim=True).clamp(min=eps)
        right = cross1 / right_norm
        
        cross2 = torch.cross(dir, right, dim=1)
        up_norm = torch.norm(cross2, dim=1, keepdim=True).clamp(min=eps)
        up_proj = cross2 / up_norm
        
        # Swivel direction
        if swivel_angle.dim() == 1:
            swivel_angle = swivel_angle.unsqueeze(1)
        elbow_dir = torch.cos(swivel_angle) * up_proj + torch.sin(swivel_angle) * right
        
        # Joint positions
        joint1 = start_pos + self.l1 * (cos_theta * dir + torch.sin(theta) * elbow_dir)
        joint2 = start_pos + vec  # Clamped end
        
        # ---- ROTATION CALCULATION ----
        # Bone directions are already computed during IK
        bone1_dir = joint1 - start_pos
        bone1_dir_norm = torch.norm(bone1_dir, dim=1, keepdim=True).clamp(min=eps)
        bone1_dir = bone1_dir / bone1_dir_norm
        
        bone2_dir = joint2 - joint1
        bone2_dir_norm = torch.norm(bone2_dir, dim=1, keepdim=True).clamp(min=eps)
        bone2_dir = bone2_dir / bone2_dir_norm
        
        # Rest pose directions
        bone1_rest = self.bone1_forward_dir.unsqueeze(0).expand(batch_size, 3)
        bone2_rest = self.bone2_forward_dir.unsqueeze(0).expand(batch_size, 3)
        bone1_up = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
        bone2_up = self.bone2_up_dir.unsqueeze(0).expand(batch_size, 3)
        
        # Compute world rotations directly
        rot1_world = self._fast_compute_rotation(bone1_rest, bone1_dir, bone1_up)
        rot2_world = self._fast_compute_rotation(bone2_rest, bone2_dir, bone2_up)
        
        # Handle parent rotation
        if parent_world_rot is not None:
            rot1_local = torch.bmm(parent_world_rot.transpose(-2, -1), rot1_world)
            rot2_local = torch.bmm(rot1_world.transpose(-2, -1), rot2_world)
            return rot1_local, rot2_local, joint1, joint2, rot1_world, rot2_world
        
        return rot1_world, rot2_world, joint1, joint2

    def fk_to_ik(self, start_pos, rot1, rot2, parent_world_rot=None, twist_rot=None):
        """
        Convert FK parameters to IK parameters.
        
        Args:
            start_pos: [B, 3] Starting position
            rot1: [B, 3, 3] First bone rotation matrix
            rot2: [B, 3, 3] Second bone rotation matrix
            parent_world_rot: [B, 3, 3] World rotation of parent (optional)
            twist_rot: [B, 3, 3] Twist bone rotation matrix (optional)
            
        Returns:
            target_pos: [B, 3] End effector position
            elbow_pos: [B, 3] Middle joint position
            swivel_angle: [B, 1] The swivel angle
        """
        # Combine rot1 with twist_rot if provided
        if twist_rot is not None:
            # Since twist is a child of the first bone in the hierarchy, 
            # multiply rot1 * twist_rot to get combined effect
            combined_rot1 = torch.matmul(rot1, twist_rot)
        else:
            combined_rot1 = rot1

        # Get joint positions from rotations
        elbow_pos, target_pos = self.fk_positions(start_pos, combined_rot1, rot2, parent_world_rot)

        swivel_angles = self.calculate_swivel(start_pos, elbow_pos, target_pos)
        
        return target_pos, elbow_pos, swivel_angles
    
    def world_positions_to_ik(self, start_pos, elbow_pos, target_pos):
        """
        Convert FK parameters to IK parameters.
        
        Args:
            start_pos: [B, 3] Starting position
            rot1: [B, 3, 3] First bone rotation matrix
            rot2: [B, 3, 3] Second bone rotation matrix
            
        Returns:
            target_pos: [B, 3] End effector position
            swivel_angle: [B, 1] The swivel angle
        """        
        swivel_angles = self.calculate_swivel(start_pos, elbow_pos, target_pos)
        
        return target_pos, swivel_angles
    
    def calculate_swivel(self, start_pos, elbow_pos, target_pos):

        batch_size = start_pos.shape[0]

        # Calculate swivel angle for each item in batch
        vec = target_pos - start_pos
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        # Create zero tensor for swivel angles
        swivel_angles = torch.zeros(batch_size, 1, device=self.device)
        
        # Handle non-degenerate cases
        valid_idx = (dist >= 1e-6).squeeze(1)
        if valid_idx.any():
            dir = torch.zeros_like(vec)
            dir[valid_idx] = vec[valid_idx] / dist[valid_idx]
            
            # Create orthonormal frame for each batch element - use bone1_up_dir
            bone1_up_dir_batch = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
            right = torch.zeros_like(vec)
            up_proj = torch.zeros_like(vec)
            
            # Calculate right and up vectors only for valid indices
            right[valid_idx] = F.normalize(torch.linalg.cross(bone1_up_dir_batch[valid_idx], dir[valid_idx]), dim=1)
            up_proj[valid_idx] = F.normalize(torch.linalg.cross(dir[valid_idx], right[valid_idx]), dim=1)
            
            # Project elbow positions onto plane perpendicular to dir
            to_elbow = elbow_pos - start_pos
            
            # Compute projection to plane perpendicular to direction
            dots = torch.sum(to_elbow * dir, dim=1, keepdim=True)
            to_elbow_proj = torch.zeros_like(to_elbow)
            to_elbow_proj[valid_idx] = to_elbow[valid_idx] - dots[valid_idx] * dir[valid_idx]
            
            # Find indices where projection is valid (not too close to zero)
            proj_norm = torch.norm(to_elbow_proj, dim=1)
            proj_valid_idx = (proj_norm >= 1e-6)
            
            if proj_valid_idx.any():
                # Normalize projection vectors
                normalized_proj = to_elbow_proj[proj_valid_idx] / proj_norm[proj_valid_idx].unsqueeze(1)
                
                # Calculate swivel angles via dot products
                cos_swivel = torch.sum(normalized_proj * up_proj[proj_valid_idx], dim=1)
                sin_swivel = torch.sum(normalized_proj * right[proj_valid_idx], dim=1)
                
                swivel_angles[proj_valid_idx] = torch.atan2(sin_swivel, cos_swivel).unsqueeze(1)
        return swivel_angles
    
    def extract_twist(self, bone_dir, rot_matrix):
        """Extract twist component (rotation around bone's own axis)"""
        batch_size = rot_matrix.shape[0]
        
        # Normalize bone direction
        bone_dir = F.normalize(bone_dir, dim=1)
        
        # Create a reference frame with bone_dir as forward
        up_ref = torch.zeros_like(bone_dir)
        # Use world up as starting point, but ensure it's not parallel to bone_dir
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        world_up = world_up.unsqueeze(0).expand(batch_size, 3)
        
        # Create perpendicular vector if needed
        parallel_mask = torch.abs(torch.sum(bone_dir * world_up, dim=1)) > 0.99
        if parallel_mask.any():
            world_up[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        
        # Create orthonormal frame
        right_ref = F.normalize(torch.cross(world_up, bone_dir, dim=1), dim=1)
        up_ref = F.normalize(torch.cross(bone_dir, right_ref, dim=1), dim=1)
        
        # Get the bone's reference up direction after rotation
        rotated_up = torch.matmul(rot_matrix, up_ref.unsqueeze(-1)).squeeze(-1)
        
        # Project onto plane perpendicular to bone direction
        proj_up = rotated_up - torch.sum(rotated_up * bone_dir, dim=1, keepdim=True) * bone_dir
        proj_up = F.normalize(proj_up, dim=1)
        
        # Calculate twist angle
        cos_twist = torch.sum(proj_up * up_ref, dim=1)
        # Cross product for direction
        cross = torch.cross(up_ref, proj_up, dim=1)
        sin_twist = torch.sum(cross * bone_dir, dim=1)
        
        twist_angle = torch.atan2(sin_twist, cos_twist).unsqueeze(1)
        return twist_angle