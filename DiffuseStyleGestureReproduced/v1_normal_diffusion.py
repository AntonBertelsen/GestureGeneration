import torch
from typing import Callable, Union
from diffusion import *

class NormalDiffusion(Diffusion):

    def __init__(self,
                 num_timesteps: int,
                 sequence_length: int, # TODO: Can we get rid of this somehow?
                 noise_schedule: tuple[Callable[[int, int], float], dict[str, Union[str, int, float, bool]]],
                 device = None  
        ):
        self.num_timesteps = num_timesteps
        self.sequence_length = sequence_length
        self.noise_schedule_hyper_params: dict[str, Union[str, int, float, bool]] = noise_schedule[1]
        self.noise_schedule: Callable[[int, int], float] = noise_schedule[0]
        self.device = device

        # In order to train faster we want to be able to jump to any level of noise at any time.
        # To do this, we precalculate the amount of noise that would have been added at any timestep. This means adding up noise
        # From all previous timesteps. We can cheat by simply scaling beta / alpha

        # We use the provided noise schedule funtion to get the intensity (?) of the noise at the current time step.
        # This is a value between 0 and 1, and determines the amount of noise to add to sequence_tensor.
        
        # Precalculate all beta values
        timesteps = torch.arange(0, num_timesteps, device=device).float() / (num_timesteps - 1)
        self.timestep_matrix = timesteps.unsqueeze(1).expand(1, sequence_length)
        
        self.beta_values = torch.tensor([self.noise_schedule(t, self.num_timesteps) for t in range(self.num_timesteps)], device=device)
        # Alpha is 1 - beta, so here we precalculate all alpha values
        self.alpha_values = 1 - self.beta_values
        # We also precalculate the cumulative product of alpha values. This is what will allow us to jump to any timestep in a single step.
        self.alpha_hat_values = torch.cumprod(self.alpha_values, dim=0)

        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hat_values)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hat_values)

    def forward(self, sequence_tensor: torch.Tensor, timestep: torch.Tensor = 0) -> torch.Tensor:
        # We generate a tensor of gaussian noise with the same shape as the sequence_tensor
        noise = torch.randn_like(sequence_tensor, device=sequence_tensor.device)
        
        sqrt_alpha_hat = self.sqrt_alpha_hats[timestep]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[timestep]

        # Reshape for proper broadcasting (from [bs] to [bs, 1, 1])
        sqrt_alpha_hat = sqrt_alpha_hat.view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(-1, 1, 1)

        # Apply the noise to the sequence_tensor
        noised_image = sqrt_alpha_hat * sequence_tensor + sqrt_one_minus_alpha_hat * noise
        
        return noised_image # , noise
    
    def get_sequence_timesteps(self, current_timestep: torch.Tensor):
        # current_timesteps is of length [bs, 1] and contains the current timestep for each sequence in the batch.
        # We need to return a tensor of shape [bs, sequence_length] where each sequence has the same timesteps.
        # We can do this by repeating the current_timesteps for each sequence in the batch.
        return self.timestep_matrix[current_timestep.to(dtype=torch.int)]
    
    @property
    def number_of_timesteps(self) -> int:
        return self.num_timesteps
    
    @property
    def clean_frame_index(self) -> int:
        raise ValueError("Normal Diffusion does not support predict_full_duration=false. (The entire gesture starts as noise, and must be predicted in full)")
    
    def get_WnB_config_specs(self):
        return {
            "num_timesteps": self.num_timesteps,
            "sequence_length": self.sequence_length,
            **self.noise_schedule_hyper_params,
        }