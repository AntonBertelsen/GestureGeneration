import torch
import torch.nn as nn
from local_attention import transformer
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from typing import Union
import numpy as np

from v1_sliding_diffusion import Diffusion
from debugger import Debugger, Show


class TestMotionModel(nn.Module):
    def __init__(self, 
                device,
                n_gesture_length: int,                      # Length of the sequence snippets to generate. We geneate in autoregressive manner, where we are constantly generating small chunks continously
                diffusion_noise_scheduler: Diffusion,
                number_of_styles: int,                      # Number of unique styles. In this context this is the number of speakers, since we treat each speaker as a style 
                audio_features_per_frame: int,              # Number of audio features per frame. This is a mixture of prosodic features, onsets, wavlm, etc.
                pose_features_per_frame: int, 
                number_of_attention_heads: int = 8,
                 
                debugger: Debugger = Debugger(False)):      # Number of pose features per frame. These are the rotations / translations of the bones in the character skeleton. We may not pay attention to every channel for every bone, or every bone. 
        super().__init__()

        self.device = device
        self.debugger = debugger
        self.diffusion_noise_scheduler = diffusion_noise_scheduler

        # Parameter from the diffusion model:
        self.max_timestep_stacking_level = diffusion_noise_scheduler.num_of_timestep_stackings

        self.num_of_pre_timestep_frames = diffusion_noise_scheduler.num_of_pre_timestep_frames
        self.num_of_timestep_frames = diffusion_noise_scheduler.num_of_timestep_frames
        self.num_of_post_timestep_frames = diffusion_noise_scheduler.num_of_post_timestep_frames

        self.n_gesture_length = n_gesture_length

        self.gesture_linear = nn.Linear(
            in_features=pose_features_per_frame, 
            out_features=pose_features_per_frame
        )
    

    def display_debug_info(self, display_debug_info: bool, filter_keys: Union[str, list[str]]):
        self.debugger = Debugger(on=display_debug_info, keys=filter_keys)


    def forward(self, 
                current_time_step_stacking_level: int,
                one_hot_style, 
                audio_features, 
                noisy_gesture_sequence,
                condition_mask_probabilty = 0.1):

        output_tensor = self.gesture_linear(noisy_gesture_sequence)
        return output_tensor

    # Functions for Weights & Biases tracking
    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict: dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)


    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking
