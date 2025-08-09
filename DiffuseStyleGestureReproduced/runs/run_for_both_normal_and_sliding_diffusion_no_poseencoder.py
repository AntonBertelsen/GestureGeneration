import torch
from v1_sliding_diffusion import Diffusion
from v1_model import ContinuousMotionModel
from utils.debugger import Debugger
from v1_training_loop import train
from diffusion import Diffusion
from v1_sliding_diffusion import SlidingDiffusion
from v1_normal_diffusion import NormalDiffusion
from pose_encoder.advanced_pose_encoder import AdvancedPoseEncoder
from torch.utils.data import DataLoader
from dataset.dataset import *
import wandb
import utils.utils as utils
import argparse


if __name__ == "__main__":
    
    print("Starting the run for both normal and sliding diffusion without pose encoder...")

    with open(".wandbkey", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    # Initialize the parser, to get the sweep_id from command line arguments
    parser = argparse.ArgumentParser(description="W&B Sweep Agent Launcher")
    # parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to join.")
    parser.add_argument("--model_type", type=str, default="normal", choices=["normal", "sliding"], help="Type of diffusion model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--entity", type=str, default="pefu-it-university-of-copenhagen", help="W&B entity (team or username).")
    parser.add_argument("--project", type=str, default="normal_vs_sliding", help="W&B project name.")

    args = parser.parse_args()

    print(f"Running with model type: {args.model_type}, seed: {args.seed}, entity: {args.entity}, project: {args.project}")

    seed = args.seed

    run = wandb.init(project=args.project, name=f"_{args.model_type}_diffusion_seed_{seed}")

    device = utils.get_device()

    utils.set_seed(seed)


    if (args.model_type == "sliding"):
        diffusion_model = SlidingDiffusion(
            num_clean_frames = 20,
            num_denoise_frames = 50,
            num_noise_frames = 0,
            num_timestep_stackings = 1,
            noise_schedule = Diffusion.linear_schedule(0.00015, 0.15),
            device = device
        )
    elif (args.model_type == "normal"):
        diffusion_model = NormalDiffusion(
            num_timesteps = 100,
            sequence_length = 70,
            noise_schedule = Diffusion.linear_schedule(0.00015, 0.075),
            device = device
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose 'normal' or 'sliding'.")
    
    print("Starting training...")

    train(
        num_fgd_samples = 2048,
        experiment_collection_name = f"sliding_vs_normal_team_{args.model_type}",
        upload_model_check_point = False, # should upload the model checkpoint to wandb
        model_checkpoint_dir = f"v1_models_sweeped/sliding_vs_normal_models/{args.model_type}",
        model = ContinuousMotionModel(
            diffusion = diffusion_model,
            pose_encoder = None, # AdvancedPoseEncoder.load_from_checkpoint("advanced_pose_encoder_ik_pca_64", device),
            number_of_styles = 17,
            gesture_length = 70,
            seed_length = 0,
            audio_features_per_frame = 37,
            pose_features_per_frame = 345,
            condition_mask_probabilty = 0.1,
            number_of_attention_heads = 8,
            predict_full_duration = True,
            reinject_seed_style_full_t = False,
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
                    seq_length = 70,
                    seed_length = 0,
                    batch_size = 256,
                    epoch_length = 1000,
                    loading_encoded_data = False,
                    include_vel_acc_features = False,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False  
            ),
        val_loader = DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/val/main-agent/consolidated.npz",
                    seq_length = 70,
                    seed_length = 0,
                    batch_size = 64,
                    epoch_length = 30,
                    loading_encoded_data = False,
                    include_vel_acc_features = False,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False
            ),
        num_epochs = 10000,
        learning_rate = 0.00005,
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
    )