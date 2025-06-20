import torch
from v1_sliding_diffusion import Diffusion
from v1_model import ContinuousMotionModel
from utils.debugger import Debugger
from v1_training_loop import train
from diffusion import Diffusion
from v1_normal_diffusion import NormalDiffusion
from v1_sliding_diffusion import SlidingDiffusion
from pose_encoder.advanced_pose_encoder import AdvancedPoseEncoder
from torch.utils.data import DataLoader
from dataset.dataset import *
import wandb
import utils.utils as utils

wandb.login()

run = wandb.init(project="no_autoencoder", name=f"no_autoencoder_normal_diffusion_test_1")

device = utils.get_device()

train(
    experiment_collection_name = "first_tests",
    upload_model_check_point = False, # should upload the model checkpoint to wandb
    model_checkpoint_dir = "v1_models",
    model = ContinuousMotionModel(
        # diffusion = SlidingDiffusion(
        #     device = device,
        #     num_of_pre_timestep_frames = 50,
        #     num_of_timestep_frames = 50,
        #     num_of_post_timestep_frames = 0,
        #     noise_schedule = Diffusion.linear_schedule(0.00015, 0.15)),
        diffusion = NormalDiffusion(
            num_timesteps=100,
            sequence_length=100,
            noise_schedule=Diffusion.linear_schedule(0.00015, 0.075),
            device=device
        ),
        # pose_encoder = AdvancedPoseEncoder.load_from_checkpoint("advanced_pose_encoder_ik_pca_64", device),
        number_of_styles = 17,
        n_gesture_length = 100,
        audio_features_per_frame = 37,
        pose_features_per_frame = 345,
        condition_mask_probabilty = 0.1,
        number_of_attention_heads = 8,
        predict_full_duration = True,
        debugger = Debugger(
            on = False, 
            keys_for_printing_while_running = ["ALL"]
        ),
        device = device
    ),
    device = device,
    training_loader = DataLoader(
            GPUDataset(
                consolidated_file = "dataset/genea2023_dataset/trn/main-agent/consolidated.npz",
                seq_length = 100,
                seed_length = 0,
                batch_size = 256,
                epoch_length = 1000,
                loading_encoded_data = False,
                device = device
            ),
            batch_size = 1,
            num_workers = 0,
            pin_memory = False  
        ),
    val_loader = DataLoader(
            GPUDataset(
                consolidated_file = "dataset/genea2023_dataset/val/main-agent/consolidated.npz",
                seq_length = 100,
                seed_length = 0,
                batch_size = 64,
                epoch_length = 30,
                loading_encoded_data = False,
                device = device
            ),
            batch_size = 1,
            num_workers = 0,
            pin_memory = False
        ),
    num_epochs = 1000,
    learning_rate = 0.000025,
    reconstruction_loss_weight = 4.0,
    variance_loss_weight = 0.1,
    velocity_loss_weight = 1.0,
    acceleration_loss_weight = 1.5,
    jerk_loss_weight= 0.2,
    latent_space_loss_weight = 0.0,
    category_weighting = {
        'fingers': 0.1,
        'arms': 2.0,
        'legs': 1.0,
        'spine': 2.0,
        'head': 1.0,
        'root': 2.0
    },
    # category_weighting = {
    #     'left_arm_ik': 2.0,
    #     'right_arm_ik': 2.0,
    #     'left_leg_ik': 2.0,
    #     'right_leg_ik': 2.0,
    # },
    # frame_weighting_segments_info = [
    #     (0.1, 0.2, 25),
    #     (0.2, 0.4, 40),
    #     (0.4, 1.0, 50),
    #     (1.0, 1.0, 55),
    #     (1.0, 0.85, 80),
    #     (0.85, 0.6, 100)
    # ],
    visualize_step = 100,
    run = run,
    wandb_config = run.config
    # continue_from_checkpoint="latest"
)