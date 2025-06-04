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
from v1_sliding_diffusion import Diffusion

import wandb

from v1_model import ContinuousMotionModel
from DiffuseStyleGestureReproduced.utils.debugger import Debugger
from v1_traing_loop import train
from v1_sliding_diffusion import Diffusion
from torch.utils.data import DataLoader
from dataset.dataset import *
import argparse




def configur_trainfunction_for_wandb_sweep(
        experiment_collection_name: str,
        current_model_name: str,):
    
    def _configured_trainfunction_for_wandb_sweep():
        run = wandb.init(
                project="v1_sliding_diffusion", 
                group=experiment_collection_name,
                name=current_model_name,
                entity="", # W&B username or team, when its empty, it will use the default team
            )
        
        config = wandb.config

        device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )


        """
        !!! IMPORTANT !!!

        IT IS ESSENTIAL TO REPLACE THE PARAMETERS FROM THE TRAINING FUNCTION WITH 
        THE PARAMETERS FROM THE SWEEP CONFIGURATION IN THE CALL TO THE TRAINING FUNCTION
        """

        train(
            run=run,
            config=config,
            debug_run=True,
            experiment_collection_name="first_tests",
            model_check_point_interval_in_epocs = 2, # how often to save the model checkpoint - set to infinity to disable
            uplaod_model_check_point = False, # should upload the model checkpoint to wandb
            model_checkpoint_dir="v1_models",
            model=ContinuousMotionModel(
                    deffsion_noise_scheduler=Diffusion(
                        device=device,
                        num_of_pre_timestep_frames=50,
                        num_of_timestep_frames=100,
                        num_of_post_timestep_frames=0,
                        noise_schedule=Diffusion.linear_schedule(0.0002, 0.005)),
                    number_of_styles = 17,
                    n_gesture_length = 150,
                    audio_features_per_frame = 804,
                    pose_features_per_frame = 345,
                    number_of_attention_heads = 8,
                    debugger = Debugger(
                        on=True, 
                        keys_for_printing_while_running=["ALL"]
                    ),
                    device=device,
            ),
            device=device,
            training_loader=DataLoader(
                    ConsolidatedRAMDataset(
                        consolidated_file= "dataset/genea2023_dataset/trn/main-agent/consolidated.npz", # or "dataset/genea2023_dataset/trn/main-agent/training_windows_100k.npz"
                        seq_length=150,
                        seed_length=8,
                        batch_size=2,
                        epoch_length=5000
                    ),
                    batch_size = 1,
                    num_workers = 0,
                    pin_memory = True
                ),
            val_loader=None, 
            num_epochs=1000,
            lr=config.learning_rate,
            variance_loss_weight = config.variance_loss_weight,
            velocity_loss_weight = 0.1,
            acceleration_loss_weight = 0.1,
            category_weighting = {'hands': 0.1, 'arms': 2.0, 'legs': 2.0, 'spine': 1.0}
        )
    return _configured_trainfunction_for_wandb_sweep

if __name__ == "__main__":

    wandb.login()

    sweep_config = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "variance_loss_weight": {"values": [0.1, 0.5, 1.0]},
        },
    }

    """
    !!! IMPORTANT !!!

    IT IS ESSENTIAL TO REPLACE THE PARAMETERS FROM THE TRAINING FUNCTION WITH 
    THE PARAMETERS FROM THE SWEEP CONFIGURATION IN THE CALL TO THE TRAINING FUNCTION

    Alwayse check the configured_trainfunction_for_wandb_sweep
    """

    the_configured_trainfunction_for_wandb_sweep = configur_trainfunction_for_wandb_sweep()
    
    sweep_id = wandb.sweep(sweep_config, project="v1_sliding_diffusion")



    wandb.agent(sweep_id, function=the_configured_trainfunction_for_wandb_sweep)

    
