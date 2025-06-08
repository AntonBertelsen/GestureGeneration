import torch
from typing import Tuple

class TwoBoneIKChain:
    def __init__(self, bone1_length: float, bone2_length: float, device=None, 
                 bone1_direction=None, bone2_direction=None):
        self.bone1_length = bone1_length
        self.bone2_length = bone2_length
        self.total_reach = bone1_length + bone2_length
        self.min_reach = abs(bone1_length - bone2_length)
        self.device = device if device is not None else torch.device('cpu')
        
        # Different bone directions for each joint
        self.bone1_direction = torch.tensor(bone1_direction or [0.0, 0.0, 1.0], device=self.device)
        self.bone2_direction = torch.tensor(bone2_direction or [0.0, 0.0, 1.0], device=self.device)

    def fk_to_ik_with_target(self, joint1_rot_6d: torch.Tensor, joint2_rot_6d: torch.Tensor, 
                    actual_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert FK representation to IK parameters using the actual target position.
        
        Args:
            joint1_rot_6d: [batch, 6] First joint 6D rotation
            joint2_rot_6d: [batch, 6] Second joint 6D rotation  
            actual_target: [batch, 3] Actual target position relative to chain origin
            
        Returns:
            target_pos: [batch, 3] Target position (same as input)
            swivel: [batch, 1] Swivel parameter (-1 to 1)
        """
        batch_size = joint1_rot_6d.shape[0]
        
        # Use the provided target instead of calculating from FK
        target_pos = actual_target
        target_distance = torch.norm(target_pos, dim=-1, keepdim=True)
        target_direction = target_pos / (target_distance + 1e-8)
        
        # Calculate what the middle joint position should be with current rotations
        joint1_rot_matrix = rotation_6d_to_matrix(joint1_rot_6d)
        
        # Use the actual bone1_direction instead of hardcoded Z+
        bone1_local = self.bone1_direction * self.bone1_length
        bone1_local = bone1_local.expand(batch_size, -1)
        
        middle_joint_pos = torch.bmm(joint1_rot_matrix, bone1_local.unsqueeze(-1)).squeeze(-1)
        
        # Calculate canonical position for zero swivel
        cos_angle1 = (self.bone1_length**2 + target_distance**2 - self.bone2_length**2) / (2 * self.bone1_length * target_distance + 1e-8)
        cos_angle1 = torch.clamp(cos_angle1, -1.0 + 1e-6, 1.0 - 1e-6)
        angle1 = torch.acos(cos_angle1)
        
        # Create canonical up vector (Y-axis, made perpendicular to target)
        canonical_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(batch_size, -1)
        canonical_up = canonical_up - torch.sum(canonical_up * target_direction, dim=-1, keepdim=True) * target_direction
        canonical_up = canonical_up / (torch.norm(canonical_up, dim=-1, keepdim=True) + 1e-8)
        
        # Calculate canonical middle joint position (swivel = 0)
        middle_along_target = self.bone1_length * torch.cos(angle1)
        middle_perp_distance = self.bone1_length * torch.sin(angle1)
        canonical_middle_pos = middle_along_target * target_direction + middle_perp_distance * canonical_up
        
        # Calculate actual perpendicular component
        middle_along_target_actual = torch.sum(middle_joint_pos * target_direction, dim=-1, keepdim=True)
        middle_perpendicular_actual = middle_joint_pos - middle_along_target_actual * target_direction
        
        # Calculate canonical perpendicular component
        canonical_middle_along_target = torch.sum(canonical_middle_pos * target_direction, dim=-1, keepdim=True)
        canonical_perpendicular = canonical_middle_pos - canonical_middle_along_target * target_direction
        
        # Calculate the angle between actual and canonical perpendicular vectors
        # This gives us the swivel angle
        perp_magnitude_actual = torch.norm(middle_perpendicular_actual, dim=-1, keepdim=True)
        perp_magnitude_canonical = torch.norm(canonical_perpendicular, dim=-1, keepdim=True)
        
        # Normalize the perpendicular vectors
        middle_perpendicular_actual_norm = middle_perpendicular_actual / (perp_magnitude_actual + 1e-8)
        canonical_perpendicular_norm = canonical_perpendicular / (perp_magnitude_canonical + 1e-8)
        
        # Calculate the signed angle between the two perpendicular vectors
        cos_swivel_angle = torch.sum(middle_perpendicular_actual_norm * canonical_perpendicular_norm, dim=-1, keepdim=True)
        cos_swivel_angle = torch.clamp(cos_swivel_angle, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Calculate cross product to determine sign
        cross_product = torch.cross(canonical_perpendicular_norm, middle_perpendicular_actual_norm, dim=-1)
        sign = torch.sign(torch.sum(cross_product * target_direction, dim=-1, keepdim=True))
        
        # Calculate the swivel angle
        swivel_angle = sign * torch.acos(cos_swivel_angle)
        
        # Convert to normalized swivel parameter [-1, 1]
        # Map the angle range [-π/4, π/4] to [-1, 1]
        swivel = swivel_angle / (torch.pi * 0.25)
        swivel = torch.clamp(swivel, -1.0, 1.0)
        
        return target_pos, swivel

    def ik_to_fk(self, target_pos: torch.Tensor, swivel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert IK parameters to FK rotations with different bone directions."""
        batch_size = target_pos.shape[0]
        target_distance = torch.norm(target_pos, dim=-1, keepdim=True)
        target_direction = target_pos / (target_distance + 1e-8)
        
        # Clamp target distance to reachable range
        target_distance = torch.clamp(target_distance, self.min_reach + 1e-6, self.total_reach - 1e-6)
        
        # Calculate joint angles using law of cosines
        cos_angle1 = (self.bone1_length**2 + target_distance**2 - self.bone2_length**2) / (2 * self.bone1_length * target_distance + 1e-8)
        cos_angle2 = (self.bone1_length**2 + self.bone2_length**2 - target_distance**2) / (2 * self.bone1_length * self.bone2_length + 1e-8)
        
        cos_angle1 = torch.clamp(cos_angle1, -1.0 + 1e-6, 1.0 - 1e-6)
        cos_angle2 = torch.clamp(cos_angle2, -1.0 + 1e-6, 1.0 - 1e-6)
        
        angle1 = torch.acos(cos_angle1)
        angle2 = torch.pi - torch.acos(cos_angle2)
        
        # Create swivel plane using bone1 direction
        bone1_dir_expanded = self.bone1_direction.expand(batch_size, -1)
        swivel_axis = torch.cross(bone1_dir_expanded, target_direction, dim=-1)
        swivel_axis_norm = torch.norm(swivel_axis, dim=-1, keepdim=True)
        
        # Handle case where target is aligned with bone direction
        nearly_aligned = swivel_axis_norm < 1e-6
        if nearly_aligned.any():
            # Use perpendicular axis as fallback
            fallback_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(batch_size, -1)
            fallback_perp = fallback_axis - torch.sum(fallback_axis * target_direction, dim=-1, keepdim=True) * target_direction
            fallback_perp = fallback_perp / (torch.norm(fallback_perp, dim=-1, keepdim=True) + 1e-8)
            
            swivel_axis = torch.where(nearly_aligned, fallback_perp, swivel_axis / (swivel_axis_norm + 1e-8))
        else:
            swivel_axis = swivel_axis / swivel_axis_norm
        
        # Create perpendicular direction in swivel plane
        perp_direction = torch.cross(target_direction, swivel_axis, dim=-1)
        
        # Apply swivel rotation
        swivel_angle = swivel.squeeze(-1) * torch.pi * 0.25
        cos_swivel = torch.cos(swivel_angle).unsqueeze(-1)
        sin_swivel = torch.sin(swivel_angle).unsqueeze(-1)
        
        # Calculate middle joint position with swivel
        middle_along_target = self.bone1_length * torch.cos(angle1)
        middle_perp_distance = self.bone1_length * torch.sin(angle1)
        middle_joint_pos = (middle_along_target * target_direction + 
                           middle_perp_distance * (cos_swivel * perp_direction + sin_swivel * swivel_axis))
        
        # First joint: align bone1_direction with middle joint direction
        joint1_forward = middle_joint_pos / (torch.norm(middle_joint_pos, dim=-1, keepdim=True) + 1e-8)
        joint1_rot_matrix = self.look_at_direction(joint1_forward, swivel_axis, self.bone1_direction)
        
        # Second joint: align bone2_direction with vector from middle to target
        bone2_world_direction = target_pos - middle_joint_pos
        bone2_world_direction = bone2_world_direction / (torch.norm(bone2_world_direction, dim=-1, keepdim=True) + 1e-8)
        
        # Transform to middle joint's local space (relative to joint1's rotation)
        joint1_rot_matrix_inv = joint1_rot_matrix.transpose(-2, -1)
        bone2_local_direction = torch.bmm(joint1_rot_matrix_inv, bone2_world_direction.unsqueeze(-1)).squeeze(-1)
        
        # Create rotation to align bone2_direction with desired local direction
        joint2_rot_matrix = self.look_at_direction(bone2_local_direction, swivel_axis, self.bone2_direction)
        
        # Convert to 6D representation
        joint1_rot_6d = matrix_to_rotation_6d(joint1_rot_matrix)
        joint2_rot_6d = matrix_to_rotation_6d(joint2_rot_matrix)
        
        return joint1_rot_6d, joint2_rot_6d

    def look_at_direction(self, target_dir: torch.Tensor, up_hint: torch.Tensor, forward_axis: torch.Tensor) -> torch.Tensor:
        """Create rotation matrix that aligns forward_axis with target_dir."""
        batch_size = target_dir.shape[0]
        forward_axis = forward_axis.expand(batch_size, -1)
        
        # Normalize target direction
        forward = target_dir / (torch.norm(target_dir, dim=-1, keepdim=True) + 1e-8)
        
        # Create right vector perpendicular to forward and up_hint
        right = torch.cross(forward, up_hint, dim=-1)
        right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-8)
        
        # Create actual up vector
        up = torch.cross(right, forward, dim=-1)
        
        # Build rotation matrix based on which axis is the forward axis
        if torch.allclose(forward_axis[0], torch.tensor([1., 0., 0.], device=self.device)):
            # X-axis forward: X=forward, Y=up, Z=right
            return torch.stack([forward, up, right], dim=-1)
        elif torch.allclose(forward_axis[0], torch.tensor([0., 0., 1.], device=self.device)):
            # Z-axis forward: X=right, Y=up, Z=forward
            return torch.stack([right, up, forward], dim=-1)
        elif torch.allclose(forward_axis[0], torch.tensor([0., 1., 0.], device=self.device)):
            # Y-axis forward: X=right, Y=forward, Z=up
            return torch.stack([right, forward, up], dim=-1)
        else:
            # Default Z-axis forward
            return torch.stack([right, up, forward], dim=-1)
    
def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rot_6d: [..., 6] rotation representation
        
    Returns:
        [..., 3, 3] rotation matrix
    """
    # Extract first and second columns
    x_col = rot_6d[..., :3]
    y_col = rot_6d[..., 3:6]
    
    # Normalize first column
    x_col = x_col / (torch.norm(x_col, dim=-1, keepdim=True) + 1e-8)
    
    # Make second column orthogonal to first
    y_col = y_col - torch.sum(x_col * y_col, dim=-1, keepdim=True) * x_col
    y_col = y_col / (torch.norm(y_col, dim=-1, keepdim=True) + 1e-8)
    
    # Third column is cross product
    z_col = torch.cross(x_col, y_col, dim=-1)
    
    # Stack to form rotation matrix
    return torch.stack([x_col, y_col, z_col], dim=-1)


def matrix_to_rotation_6d(rot_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation.
    
    Args:
        rot_matrix: [..., 3, 3] rotation matrix
        
    Returns:
        [..., 6] 6D rotation representation
    """
    # Extract first two columns and flatten
    batch_shape = rot_matrix.shape[:-2]
    x_col = rot_matrix[..., 0]
    y_col = rot_matrix[..., 1]
    return torch.cat([x_col, y_col], dim=-1)
