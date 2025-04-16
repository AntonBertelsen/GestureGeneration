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
from debugger import Debugger
from v1_traing_loop import train
from v1_sliding_diffusion import Diffusion
from torch.utils.data import DataLoader
from dataset.dataset import *


if __name__ == "__main__":
    wandb.login()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu")

    trained_model = train(
        debug_run=True,
        experiment_collection_name="first_tests",
        device=device,
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
        training_loader=DataLoader(
                ConsolidatedRAMDataset(
                    consolidated_file= "dataset/genea2023_dataset/trn/main-agent/consolidated.npz", # or "dataset/genea2023_dataset/trn/main-agent/training_windows_100k.npz"
                    seq_length=150,
                    seed_length=8,
                    batch_size=10,
                    epoch_length=1
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = True
            ),
        val_loader=None, 
        num_epochs=1
    )
