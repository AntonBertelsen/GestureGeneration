import torch
from typing import Callable, Union
from diffusion_process_super import *

class OutpaintDiffusion(Diffusion):

    def __init__(self,
                 num_timesteps: int,          # Number of denoising steps
                 sequence_length: int,        # Total length of sequence 
                 overlap_frames: int,         # Number of frames from previous generation to keep clean
                 noise_schedule: tuple[Callable[[int, int], float], dict[str, Union[str, int, float, bool]]],
                 device = None  
        ):
        self.num_timesteps = num_timesteps
        self.sequence_length = sequence_length
        self.overlap_frames = overlap_frames
        self.noise_schedule_hyper_params = noise_schedule[1]
        self.noise_schedule = noise_schedule[0]
        self.device = device

        # Create timestep matrix like in normal diffusion
        timesteps = torch.arange(0, num_timesteps, device=device).float() / (num_timesteps - 1)
        self.timestep_matrix = timesteps.unsqueeze(1).repeat(1, sequence_length)
        
        # Calculate beta values for each timestep
        self.beta_values = torch.tensor([self.noise_schedule(t, self.num_timesteps) 
                                       for t in range(self.num_timesteps)], 
                                      device=device)
        
        # Calculate alpha values and cumulative products
        self.alpha_values = 1 - self.beta_values
        self.alpha_hat_values = torch.cumprod(self.alpha_values, dim=0)
        
        # Precalculate square roots for efficiency
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hat_values)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hat_values)
        
        # Create overlap mask (1 for overlap region, 0 elsewhere)
        # This will be used during inference to keep the overlap region clean
        self.overlap_mask = torch.zeros((1, sequence_length, 1), device=device)
        self.overlap_mask[:, :overlap_frames] = 1.0

    def forward(self, sequence_tensor: torch.Tensor, timestep: torch.Tensor = 0) -> torch.Tensor:
        # Generate random noise
        noise = torch.randn_like(sequence_tensor, device=sequence_tensor.device)
        
        # Get noise scaling factors for current timestep
        sqrt_alpha_hat = self.sqrt_alpha_hats[timestep]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[timestep]
        
        # Reshape for proper broadcasting (from [bs] to [bs, 1, 1])
        sqrt_alpha_hat = sqrt_alpha_hat.view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(-1, 1, 1)
        
        # Apply noise to the sequence, but don't add noise to the overlap region during training
        # For training, this happens automatically because the loss is only computed on the 
        # non-overlap region. During inference, we'll handle this with the mask.
        noised_sequence = sqrt_alpha_hat * sequence_tensor + sqrt_one_minus_alpha_hat * noise

        noised_sequence = noised_sequence * (1 - self.overlap_mask) + sequence_tensor * self.overlap_mask
        
        return noised_sequence
    
    def get_sequence_timesteps(self, current_timestep: torch.Tensor):
        # Return frame-specific timesteps for current level
        return self.timestep_matrix[current_timestep.to(dtype=torch.int)]
    
    @property
    def number_of_timesteps(self) -> int:
        return self.num_timesteps
    
    @property
    def clean_frame_index(self) -> int:
        return self.overlap_frames
    
    def get_WnB_config_specs(self):
        return {
            "type": "outpaint_diffusion",
            "num_timesteps": self.num_timesteps,
            "sequence_length": self.sequence_length,
            "overlap_frames": self.overlap_frames,
            **self.noise_schedule_hyper_params,
        }