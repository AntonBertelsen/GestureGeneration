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

        vec = target_pos - start_pos
        dist = torch.norm(vec, dim=1, keepdim=True)
        max_reach = self.l1 + self.l2

        # Clamp distance
        too_far_idx = (dist > max_reach).squeeze(1)  # Convert to [B] for indexing
        if too_far_idx.any():
            vec[too_far_idx] = vec[too_far_idx] / dist[too_far_idx] * max_reach
            dist[too_far_idx] = max_reach
            
        too_close_idx = (dist < 1e-6).squeeze(1)  # Convert to [B] for indexing
        if too_close_idx.any():
            # Set a small default direction for degenerate cases
            default_dir = torch.zeros_like(vec)
            default_dir[:, 2] = 1.0  # Use Z direction as default
            vec[too_close_idx] = default_dir[too_close_idx] * 1e-6
            dist[too_close_idx] = 1e-6

        dir = vec / dist

        # Law of cosines
        cos_theta = (self.l1**2 + dist**2 - self.l2**2) / (2 * self.l1 * dist)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        # Orthonormal frame - use bone1_up_dir for swivel reference
        up_dir_batch = self.bone1_up_dir.unsqueeze(0).expand(batch_size, 3)
        right = F.normalize(torch.cross(up_dir_batch, dir, dim=1), dim=1)
        up_proj = F.normalize(torch.cross(dir, right, dim=1), dim=1)

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
        while preserving appropriate up direction
        
        Args:
            v1: [B, 3] Source direction (typically rest pose)
            v2: [B, 3] Target direction (typically current pose)
            up_hint: [B, 3] Up direction hint for disambiguation
        """
        batch_size = v1.shape[0]
        
        # Normalize vectors
        v1 = F.normalize(v1, dim=1)
        v2 = F.normalize(v2, dim=1)
        
        # Compute rotation axis and angle
        cos_angle = torch.sum(v1 * v2, dim=1).clamp(-1.0, 1.0)
        
        # Create result tensor
        result = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, 3, 3).clone()
        
        # Compute rotation axis (normalized cross product) for all items
        # This avoids the boolean indexing issue
        axis = torch.cross(v1, v2, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        
        # Create a mask for non-parallel vectors (where cross product is meaningful)
        near_parallel = cos_angle.abs() > 0.99999
        non_parallel = ~near_parallel
        
        # Handle non-parallel cases (where cross product is valid)
        if non_parallel.any():
            # Only normalize and compute for non-parallel vectors
            valid_axis = (axis_norm > 1e-6).squeeze(-1)
            valid_non_parallel = non_parallel & valid_axis
            
            if valid_non_parallel.any():
                # For valid cases, compute rotation using Rodrigues formula
                for i in range(batch_size):
                    if valid_non_parallel[i]:
                        angle = torch.acos(cos_angle[i])
                        axis_normalized = axis[i] / axis_norm[i]
                        
                        # Create cross-product matrix for axis vector
                        K = torch.zeros(3, 3, device=self.device)
                        x, y, z = axis_normalized[0], axis_normalized[1], axis_normalized[2]
                        
                        # Fill cross product matrix
                        K[0, 1] = -z
                        K[0, 2] = y
                        K[1, 0] = z
                        K[1, 2] = -x
                        K[2, 0] = -y
                        K[2, 1] = x
                        
                        # Rodrigues formula: I + sin(a)*K + (1-cos(a))*K^2
                        eye = torch.eye(3, device=self.device)
                        result[i] = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
        
        # Handle parallel case
        if near_parallel.any():
            # For parallel vectors (cos ≈ 1), keep identity matrix
            # For anti-parallel vectors (cos ≈ -1), rotate 180° around perpendicular axis
            anti_parallel = near_parallel & (cos_angle < 0)
            
            if anti_parallel.any():
                for i in range(batch_size):
                    if anti_parallel[i]:
                        # Find perpendicular axis using up_hint
                        perp_axis = torch.cross(v1[i], up_hint[i])
                        perp_norm = torch.norm(perp_axis)
                        
                        if perp_norm > 1e-6:
                            # Create 180° rotation around perpendicular axis
                            axis = perp_axis / perp_norm
                            x, y, z = axis[0], axis[1], axis[2]
                            
                            # 180° rotation matrix around axis
                            R = torch.zeros(3, 3, device=self.device)
                            
                            # Fill rotation matrix (180° rotation formula)
                            R[0, 0] = 1 - 2*(y*y + z*z)
                            R[0, 1] = 2*(x*y)
                            R[0, 2] = 2*(x*z)
                            R[1, 0] = 2*(x*y)
                            R[1, 1] = 1 - 2*(x*x + z*z)
                            R[1, 2] = 2*(y*z)
                            R[2, 0] = 2*(x*z)
                            R[2, 1] = 2*(y*z)
                            R[2, 2] = 1 - 2*(x*x + y*y)
                            
                            result[i] = R
        
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

    def ik_to_fk(self, start_pos, target_pos, swivel_angle, parent_world_rot=None):
        """
        Convenience method: IK → FK (rotations)
        
        Args:
            start_pos: [B, 3] Starting position
            target_pos: [B, 3] Target position
            swivel_angle: [B, 1] or [B] Swivel angle
            parent_world_rot: [B, 3, 3] World rotation of parent (optional)
            
        Returns:
            rot1: [B, 3, 3] First joint rotation
            rot2: [B, 3, 3] Second joint rotation
            joint1: [B, 3] Middle joint position
            joint2: [B, 3] End joint position
        """
        joint1, joint2 = self.ik(start_pos, target_pos, swivel_angle)
        rot1_world, rot2_world = self.fk_rotations(start_pos, joint1, joint2)

        if parent_world_rot is not None:
            # Local rotation = inverse(parent_world_rot) * world_rot
            rot1_local = torch.matmul(parent_world_rot.transpose(-2, -1), rot1_world)
            
            # Second bone's parent is the first bone
            rot2_local = torch.matmul(rot1_world.transpose(-2, -1), rot2_world)
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