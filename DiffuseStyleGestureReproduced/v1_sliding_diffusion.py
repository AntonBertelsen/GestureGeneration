import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from local_attention import transformer
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
import matplotlib.pyplot as plt
from typing import Union
from io import BytesIO
from PIL import Image
from moviepy import ImageSequenceClip
from datetime import datetime

from einops import rearrange


import math
from typing import Callable

from WnB_trackable import WnBTrackable

class Diffusion(WnBTrackable):


    def __init__(self, 
                 num_of_pre_timestep_frames: int, 
                 num_of_timestep_frames: int, 
                 num_of_post_timestep_frames: int,
                 noise_schedule: tuple[Callable[[int, int], float], dict[str, Union[str, int, float, bool]]],
                 device: str):
        
        self.device = device
        self.num_of_pre_timestep_frames = num_of_pre_timestep_frames
        self.num_of_timestep_frames = num_of_timestep_frames
        self.num_of_post_timestep_frames = num_of_post_timestep_frames

        # This is the noise schedule function that will be used to add noise at diffrent levels, given a timestamp, and a max number of timestpes.
        self.noise_schedule: Callable[[int, int], float] = noise_schedule[0]

        # This is a dictionary with the hyperparameters used for W&B tracking
        self.noise_schedule_hyper_params: dict[str, Union[str, int, float, bool]] = noise_schedule[1] 


        # In order to train faster we want to be able to jump to any level of noise at any time.
        # To do this, we precalculate the amount of noise that would have been added at any timestep. This means adding up noise
        # From all previous timesteps. We can cheat by simply scaling beta / alpha
        
        # Precalculate all beta values
        self.beta_values = torch.tensor([self.noise_schedule(t, self.num_of_timestep_frames) for t in range(self.num_of_timestep_frames)]).to(self.device)
        # Alpha is 1 - beta, so here we precalculate all alpha values
        self.alpha_values = 1 - self.beta_values
        # We also precalculate the cumulative product of alpha values. This is what will allow us to jump to any timestep in a single step.
        self.alpha_hat_values = torch.cumprod(self.alpha_values, dim=0)

        # and we precalculate the square roots so we dont have to do them in every field in the 
        # tensor since they are all identical
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hat_values)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hat_values)


        # Finaly, we precalculate the sqrt_alpha_hat and sqrt_one_minus_alpha_hat vectors for each timestep
        # These vectores are then padded with 1s and 0s in the pre-timestep and post-timestep frames, to ensure that 
        # the pre-timestep frames are not noised, and the post-timestep frames are fully noised, in the forward function.
        self.sqrt_alpha_hats = [1] * self.num_of_pre_timestep_frames + self.sqrt_alpha_hats.tolist() + [0] * self.num_of_post_timestep_frames
        self.sqrt_one_minus_alpha_hats = [0] * self.num_of_pre_timestep_frames + self.sqrt_one_minus_alpha_hats.tolist() + [1] * self.num_of_post_timestep_frames

        # Turn it back into tensors
        self.sqrt_alpha_hats = torch.tensor(self.sqrt_alpha_hats).to(self.device)
        self.sqrt_one_minus_alpha_hats = torch.tensor(self.sqrt_one_minus_alpha_hats).to(self.device)

        print("sqrt_alpha_hats", self.sqrt_alpha_hats.shape, self.sqrt_alpha_hats)
        print("sqrt_one_minus_alpha_hats", self.sqrt_one_minus_alpha_hats.shape, self.sqrt_one_minus_alpha_hats)

        # The underlying math is still
        # noised_squence_collumn = sqrt_alpha_hat * image + sqrt_one_minus_alpha_hat * noise TODO: check this
        #
        # For the pre-timestep frames, the sqrt_alpha_hat is 1, and the sqrt_one_minus_alpha_hat is 0:
        # noised_squence_collumn = 1 * squence_collumn + 0 * noise = squence_collumn = squence_collumn
        #
        # For the post-timestep frames, the sqrt_alpha_hat is 0, and the sqrt_one_minus_alpha_hat is 1:
        # noised_squence_collumn = 0 * squence_collumn + 1 * noise = noise = noise


    def forward(self, seqence_tensor: torch.Tensor) -> torch.Tensor:

        # 1 - We use the provided noise schedule funtion to get the intensity (?) of the noise at the current time step.
        #     This is a value between 0 and 1, and determine the amount of noise to add to the 'image'.
        #     Some noise schedules, like cosine and sigmoid, require a hyperparam, but this is already gien as a paramenter.
        # beta = self.noise_schedule(current_timestep, self.num_of_timestep_frames)
        # beta = self.beta_values[current_timestep]
        
        # 2 - We generate a tensor of noise with the same shape as the the 'image' tensor
        #     In torch.randn_like each elements are from a Gaussian distribution by default.
        noise = torch.randn_like(seqence_tensor)

        # 3 - We use the same device as the 'image' tensor for this noice tensor
        noise = noise.to(self.device)
        
        # 4 - Compute the noising of the culoumns in the timestep frame section.
        #     We do this by multipling with our pre-prepared sqrt_alpha_hat and sqrt_one_minus_alpha_hat vectores. 
        #     To not aplay noise to the pre-timestep frames and aply full noise to the post-timestep frames,
        #     we have allready added 1 and 0s in the apropeate places in the pre calculated vectors 
        #
        #     The underlying math is still
        #     noised_squence_collumn = sqrt_alpha_hat * image + sqrt_one_minus_alpha_hat * noise TODO: check this
        #
        #     For the pre-timestep frames, the sqrt_alpha_hat is 1, and the sqrt_one_minus_alpha_hat is 0:
        #     noised_squence_collumn = 1 * squence_collumn + 0 * noise = squence_collumn = squence_collumn
        #
        #     For the post-timestep frames, the sqrt_alpha_hat is 0, and the sqrt_one_minus_alpha_hat is 1:
        #     noised_squence_collumn = 0 * squence_collumn + 1 * noise = noise = noise

        # print("seqence_tensor", seqence_tensor.shape, seqence_tensor)
        # print("noise", noise.shape, noise)
        # print("sqrt_alpha_hats", self.sqrt_alpha_hats.shape, self.sqrt_alpha_hats)
        # print("sqrt_one_minus_alpha_hats", self.sqrt_one_minus_alpha_hats.shape, self.sqrt_one_minus_alpha_hats)

        if len(seqence_tensor.shape) == 3:
            # If the input is a 3D tensor, we need to add a batch dimension
            self.sqrt_alpha_hats = self.sqrt_alpha_hats.unsqueeze(1)
            self.sqrt_one_minus_alpha_hats = self.sqrt_one_minus_alpha_hats.unsqueeze(1)
        
        # print("AFTER seqence_tensor", seqence_tensor.shape, seqence_tensor)
        # print("AFTER noise", noise.shape, noise)
        # print("AFTER sqrt_alpha_hats", self.sqrt_alpha_hats.shape, self.sqrt_alpha_hats)
        # print("AFTER sqrt_one_minus_alpha_hats", self.sqrt_one_minus_alpha_hats.shape, self.sqrt_one_minus_alpha_hats)

        noised_image = self.sqrt_alpha_hats * seqence_tensor + self.sqrt_one_minus_alpha_hats * noise

        # print("noised_image", noised_image.shape, noised_image)
        
        return noised_image

    # noise schedules

    # There are several different noise schedules that can be used.
    # which one to use is a hyperparameter that can be tuned.
    # We detach them from the forward diffusion function, so it is easier to switch between them.

    @staticmethod
    def linear_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # This is a simple linear schedule. Noice is added as a uniform increase.
        # Pros: Simple and works well.
        # Cons: Can be suboptimal, and to0 simple

        def linear_schedule(t: int, T: int) -> float:
            return beta_min + (beta_max - beta_min) * (t / T)
        
        return (linear_schedule, {"name": "linear_schedule", "beta_min": beta_min, "beta_max": beta_max})
    
    @staticmethod
    def quadratic_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # This makes the diffusion more 'agressive' (high var) when t is high. Starts slow, then increases rapidly.
        # Pro: Preserves details early on
        # Con: High noise at later timesteps

        def quadratic_schedule(t: int, T: int) -> float:
            return beta_min + (beta_max - beta_min) * (t / T) ** 2
            # return (t / T) ** 2

        return (quadratic_schedule, {"name": "quadratic_schedule", "beta_min": beta_min, "beta_max": beta_max})
    
    @staticmethod
    def cosine_schedule(s = 0.008) -> Callable[[int, int], float]:
        # 's' is the hyperparam of the cosine - its described as a 'Small offset for stability'.

        # This is a smooth cosine-shaped increase of the variance. 
        # Pro: Improves sample quality, and is more stable than linear and quadratic
        # Con: Needs hyperparameter tuning

        def cosine_schedule(t: int, T: int):
            return math.cos((t / T + s) / (1 + s) * (math.pi / 2)) ** 2

        return (cosine_schedule, {"name": "cosine_schedule", "s": s})
    
    @staticmethod
    def exponential_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # 'beta' is the hyperparams of the exponentaial schedual.
        # Typical beta values are 0.0001 and 0.02, and they are used to control the rate of exponental decay of the variance.

        def exponential_schedule(t: int, T: int) -> float:
            return beta_min * ((beta_max / beta_min) ** (t / T))

        return (exponential_schedule, {"name": "exponential_schedule", "beta_min": beta_min, "beta_max": beta_max})
    
    @staticmethod
    def sigmoid_schedule(k = 10, beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # 'k' is the hyperparam of the sigmoid schedule.
        # It controls the steepness of the sigmoid function, than inturn controls how fast the variance changes 
        # at the start, end and middel of the diffusion.

        def sigmoid_schedule(t: int, T: int):
            return beta_min + (beta_max - beta_min) / (1 + math.exp(-k * (t / T - 0.5)))
            # return 1 / (1 + math.exp(-k * (t / T - 0.5)))

        return (sigmoid_schedule, {"name": "sigmoid_schedule", "k": k, "beta_min": beta_min, "beta_max": beta_max})

    # Implenents the abstract method from the WnBTrackable ABC class (interface)
    def get_WnB_config_specs(self):
        # Return the configuration specs needed for Weights & Biases tracking.
        # This should be a dictionary with keys as the parameter names and values as their types.
        return {
            "num_of_pre_timestep_frames": self.num_of_pre_timestep_frames,
            "num_of_timestep_frames": self.num_of_timestep_frames,
            "num_of_post_timestep_frames": self.num_of_post_timestep_frames,
            **self.noise_schedule_hyper_params,
        }