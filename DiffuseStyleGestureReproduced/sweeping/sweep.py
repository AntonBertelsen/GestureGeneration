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
from variational_autoencoder import VAE
from v1_vae import train_vae


def configur_trainfunction_for_wandb_sweep(
        project_name: str,
        experiment_collection_name: str,
        current_model_name: str,):
    
    def _configured_trainfunction_for_wandb_sweep():
        run = wandb.init(
                project=project_name, 
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

        # Configger the Training function, using the wandb configs:
        training_func = sweep_traing_confics(
            device, run=run, config=config, num_epochs=10000,
            ).VAE_sweep(
                model_save_name = current_model_name,
                
                warm_up_kl_anneal_steps=20, 
                kl_anneal_steps=200, 
                kl_beta=0.0015, 
                z_dim=48, 
                lr=1e-4,
            )
        
        # Call the produced traing function:
        training_func()

    return _configured_trainfunction_for_wandb_sweep

class sweep_traing_confics:

    def __init__(self, device, num_epochs, run, config):
        self.device = device
        self.num_epochs = num_epochs
        self.run = run
        self.config = config
        
    def VAE_sweep(self, model_save_name = "vae_model", warm_up_kl_anneal_steps=20, kl_anneal_steps=200, kl_beta=0.0015, z_dim=48, lr=1e-4,):
        return train_vae(
            run=self.run,
            wandb_config=self.config,
            device=self.device,
            pose_training_loader = DataLoader(
                    GPUDataset(
                        consolidated_file= "dataset/genea2023_dataset/toy/main-agent/consolidated.npz",
                        seq_length=1,
                        seed_length=0,
                        batch_size=64,
                        epoch_length=1000,
                        device=self.device
                    ),
                    batch_size = 1,
                    num_workers = 0,
                    pin_memory = False  
                ),
            model=VAE(
                z_dim=z_dim, 
                pose_dim=345
            ),
            bone_category_weights = {
                'fingers': 0.2,
                'arms': 3.0,
                'legs': 1.0,
                'spine': 2.0,
                'head': 1.0,
                'root': 2.0
            },
            lr = lr,
            optimizer_f = optim.AdamW,
            num_epochs = self.num_epochs,
            warm_up_kl_anneal_steps = warm_up_kl_anneal_steps,
            kl_anneal_steps = kl_anneal_steps,
            free_bits = 0.0,  # Free bits threshold for KL divergence
            kl_beta=kl_beta,
            visualize_steps = 20, # Number of steps to visualize the training process
            model_save_dir = "VAE_models_checkpoints",
            model_save_name = model_save_name,
            display_progress=False
        )
    
    def main_sd_model_sweep(self, vae_model, lr):
        return train(
            run=self.run,
            wandb_config=self.config,
            experiment_collection_name="first_tests",
            upload_model_check_point = False, # should upload the model checkpoint to wandb
            model_checkpoint_dir="v1_models",
            model=ContinuousMotionModel(
                    diffusion_noise_scheduler=Diffusion(
                        device=self.device,
                        num_of_pre_timestep_frames=50,
                        num_of_timestep_frames=50,
                        num_of_post_timestep_frames=0,
                        noise_schedule = Diffusion.linear_schedule(0.00015, 0.15)),
                    number_of_styles = 17,
                    n_gesture_length = 100,
                    audio_features_per_frame = 37,
                    pose_features_per_frame = 48,
                    number_of_attention_heads = 8,
                    debugger = Debugger(
                        on=False, 
                        keys_for_printing_while_running=["ALL"]
                    ),
                    device=self.device,
            ),
            autoencoder_model=vae_model,
            device=self.device,
            training_loader=DataLoader(
                    GPUDataset(
                        consolidated_file= "dataset/genea2023_dataset/toy/main-agent/consolidated.npz",
                        seq_length=100,
                        seed_length=0,
                        batch_size=64,
                        epoch_length=30,
                        device=self.device
                    ),
                    batch_size = 1,
                    num_workers = 0,
                    pin_memory = False  
                ),
            val_loader=DataLoader(
                    GPUDataset(
                        consolidated_file= "dataset/genea2023_dataset/toy/main-agent/consolidated.npz",
                        seq_length=100,
                        seed_length=0,
                        batch_size=64,
                        epoch_length=30,
                        device=self.device
                    ),
                    batch_size = 1,
                    num_workers = 0,
                    pin_memory = False
                ),
            num_epochs=1000,
            learning_rate=lr,
            reconstruction_loss_weight = 1.0,
            variance_loss_weight = 1.0,
            velocity_loss_weight = 1.0,
            acceleration_loss_weight = 1.0,
            latent_space_loss_weight = 0.25,
            category_weighting = {
                'fingers': 0.1,
                'arms': 2.0,
                'legs': 1.0,
                'spine': 2.0,
                'head': 1.0,
                'root': 2.0
            },
            visualize_step=10
        )
    


if __name__ == "__main__":

    # Load the key from the file .wanddbkey
    with open(".wandbkey", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    project_name = "v1_sliding_diffusion"

    """
    !!! IMPORTANT !!!

    IT IS ESSENTIAL TO REPLACE THE PARAMETERS FROM THE TRAINING FUNCTION WITH 
    THE PARAMETERS FROM THE SWEEP CONFIGURATION IN THE CALL TO THE TRAINING FUNCTION

    Alwayse check the configured_trainfunction_for_wandb_sweep
    """

    sweep_config = {
        "method": "random",
        "metric": {"name": "total_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.00001, "max": 0.001},
        },
    }

    the_configured_trainfunction_for_wandb_sweep = configur_trainfunction_for_wandb_sweep(
        project_name=project_name,
        experiment_collection_name="VAE_V1_sweep_1",
        current_model_name="VAE_V1",
    )

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, function=the_configured_trainfunction_for_wandb_sweep)
