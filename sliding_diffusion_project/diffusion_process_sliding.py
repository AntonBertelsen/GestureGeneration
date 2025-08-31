import torch
from typing import Callable, Union
from diffusion_process_super import *

class SlidingDiffusion(Diffusion):

    def __init__(self, 
                 num_clean_frames: int, 
                 num_denoise_frames: int, 
                 num_noise_frames: int,
                 num_timestep_stackings: int,
                 noise_schedule: tuple[Callable[[int, int], float], dict[str, Union[str, int, float, bool]]],
                 device: str
        ):

        self.noise_schedule_hyper_params: dict[str, Union[str, int, float, bool]] = noise_schedule[1]
        self.noise_schedule: Callable[[int, int], float] = noise_schedule[0]

        self.num_timestep_stackings = num_timestep_stackings
        self.num_clean_frames = num_clean_frames
        self.num_denoise_frames = num_denoise_frames
        self.num_noise_frames = num_noise_frames

        # We support stacking timesteps to allow multiple denoising steps for every frame slide.
        # I.e. with a stacking level for 3 the timesteps will be stacked like this: [1, 2, 3, 4, 5, 6, ...] --> [1, 4, ...], [2, 5, ...], [3, 6, ...]
        frame_timesteps_list = []
        sqrt_alpha_hats_list = []
        sqrt_one_minus_alpha_hats_list = []

        for stacking_step in range(num_timestep_stackings):
            clean_frames_timesteps = [0] * num_clean_frames
            denoise_frames_timesteps = [stacking_step + frame * num_timestep_stackings for frame in range(num_denoise_frames)]
            noise_frames_timesteps = [num_denoise_frames * num_timestep_stackings] * num_noise_frames
            
            frame_timesteps = torch.tensor(clean_frames_timesteps + denoise_frames_timesteps + noise_frames_timesteps, device=device)

            # we precalculate beta values according to the coresponing stacking step.
            beta_values = torch.tensor([self.noise_schedule(frame_timestep, num_denoise_frames * num_timestep_stackings) for frame_timestep in frame_timesteps], device=device)
            
            # We replace all the values in clean frames with 0, since they are not suppoed to be noised at all.
            beta_values[:num_clean_frames] = 0.0
            
            # Alpha is 1 - beta, so here we precalculate all alpha values
            alpha_values = 1 - beta_values
            
            # We also precalculate the cumulative product of alpha values. This is what will allow us to jump to any timestep in a single step.
            alpha_hat_values = torch.cumprod(alpha_values, dim=0)

            # and we precalculate the square roots
            sqrt_alpha_hats = torch.sqrt(alpha_hat_values)            
            sqrt_one_minus_alpha_hats = torch.sqrt(1 - alpha_hat_values)

            # Normalize the fame_timesteps to the range [0, 1]
            frame_timesteps = frame_timesteps.float() / (num_denoise_frames * num_timestep_stackings - 1)

            frame_timesteps_list.append(frame_timesteps)  # Add a dimension for the stacking level
            sqrt_alpha_hats_list.append(sqrt_alpha_hats)
            sqrt_one_minus_alpha_hats_list.append(sqrt_one_minus_alpha_hats)

        self.frame_timesteps_stacked = torch.stack(frame_timesteps_list)
        self.sqrt_alpha_hats_stacked = torch.stack(sqrt_alpha_hats_list)
        self.sqrt_one_minus_alpha_hats_stacked = torch.stack(sqrt_one_minus_alpha_hats_list)


    def forward(self, sequence_tensor: torch.Tensor, timestep: torch.Tensor = 0) -> torch.Tensor:
        # We generate a tensor of gaussian noise with the same shape as the sequence_tensor
        noise = torch.randn_like(sequence_tensor, device=sequence_tensor.device)
        
        sqrt_alpha_hats = self.sqrt_alpha_hats_stacked[timestep]
        sqrt_one_minus_alpha_hats = self.sqrt_one_minus_alpha_hats_stacked[timestep]

        # Add an unsqueeze operation to create proper broadcasting dimension
        sqrt_alpha_hats = sqrt_alpha_hats.unsqueeze(-1)         # [256, 100] -> [256, 100, 1]
        sqrt_one_minus_alpha_hats = sqrt_one_minus_alpha_hats.unsqueeze(-1)  # [256, 100] -> [256, 100, 1]

        # Calculate the noised image
        noised_image = sqrt_alpha_hats * sequence_tensor + sqrt_one_minus_alpha_hats * noise
        
        return noised_image # , noise
    
    def get_sequence_timesteps(self, current_timestep: torch.Tensor):
        # Returns a tensor of timesteps for the current stacking level
        return self.frame_timesteps_stacked[current_timestep.to(dtype=torch.int)]
    
    @property
    def number_of_timesteps(self) -> int:
        return self.num_timestep_stackings
    
    @property
    def clean_frame_index(self) -> int:
        return self.num_clean_frames
    
    def get_WnB_config_specs(self):
        return {
            "type": "sliding_diffusion",
            "num_clean_frames": self.num_clean_frames,
            "num_denoise_frames": self.num_denoise_frames,
            "num_noise_frames": self.num_noise_frames,
            "num_timestep_stackings": self.num_timestep_stackings,
            **self.noise_schedule_hyper_params,
        }