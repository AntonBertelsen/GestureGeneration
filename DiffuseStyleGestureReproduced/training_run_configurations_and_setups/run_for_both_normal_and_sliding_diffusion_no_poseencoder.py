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
    parser.add_argument("--num_clean_frames", type=int, default=None, choices=[100, 50, 20, 5,], help="Number of clean frames for sliding diffusion.")
    parser.add_argument("--num_denoise_frames", type=int, default=None, choices=[200, 100, 50, 20, 5], help="Number of denoise frames for sliding diffusion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--entity", type=str, default="pefu-it-university-of-copenhagen", help="W&B entity (team or username).")
    parser.add_argument("--project", type=str, default="normal_vs_sliding", help="W&B project name.")

    args = parser.parse_args()

    print(f"Running with model type: {args.model_type}, seed: {args.seed}, entity: {args.entity}, project: {args.project}")

    seed = args.seed

    name_parts = [f"{args.model_type}_seed_{seed}"]

    if args.num_clean_frames is not None:
        name_parts.append(f"clean_{args.num_clean_frames}")

    if args.num_denoise_frames is not None:
        name_parts.append(f"denoise_{args.num_denoise_frames}")

    name_parts.append(f"with_8_seed_frames")

    final_name = "_".join(name_parts)

    run = wandb.init(project=args.project, name=final_name)

    print(f"Run initialized with name: {final_name}")
    print(f"Run initialized with name: {final_name}")
    
    if args.num_clean_frames is None or args.num_denoise_frames is None:
        run.config.update({
            "model_type": args.model_type,
            "num_clean_frames": args.num_clean_frames,
            "num_denoise_frames": args.num_denoise_frames,
            "seed": seed,
        })

    device = utils.get_device()

    utils.set_seed(seed)

    if (args.model_type == "sliding"):
        num_clean_frames = args.num_clean_frames
        num_denoise_frames = args.num_denoise_frames
        seq_length = num_clean_frames + num_denoise_frames
        
        if num_denoise_frames == 5:
            noise_schedule = Diffusion.linear_schedule(0.00005, 0.55)
        if num_denoise_frames == 20:
            noise_schedule = Diffusion.linear_schedule(0.00015, 0.2)
        if num_denoise_frames == 50:
            noise_schedule = Diffusion.linear_schedule(0.00020, 0.1)
        if num_denoise_frames == 100:
            noise_schedule = Diffusion.linear_schedule(0.00020, 0.06)
        if num_denoise_frames == 200:
            noise_schedule = Diffusion.linear_schedule(0.00025, 0.003)

        diffusion_model = SlidingDiffusion(
            num_clean_frames = num_clean_frames,
            num_denoise_frames = num_denoise_frames,
            num_noise_frames = 0,
            num_timestep_stackings = 1,
            noise_schedule = noise_schedule,
            device = device
        )
    elif (args.model_type == "normal"):
        seq_length = 70  # Normal diffusion uses a fixed sequence length of 70

        diffusion_model = NormalDiffusion(
            num_timesteps = 100,
            sequence_length = seq_length,
            noise_schedule = Diffusion.linear_schedule(0.00015, 0.075),
            device = device
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose 'normal' or 'sliding'.")
    
    print("Starting training...")

    seed_length = 8  # Seed length for the model

    train(
        num_fgd_samples = 2048,
        experiment_collection_name = f"sliding_vs_normal_team_{args.model_type}",
        upload_model_check_point = False, # should upload the model checkpoint to wandb
        model_checkpoint_dir = f"v1_models_sweeped/sliding_vs_normal_models/{args.model_type}",
        model = ContinuousMotionModel(
            diffusion = diffusion_model,
            pose_encoder = None, # AdvancedPoseEncoder.load_from_checkpoint("advanced_pose_encoder_ik_pca_64", device),
            number_of_styles = 17,
            gesture_length = seq_length,
            seed_length = seed_length,
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
                    seq_length = seq_length,
                    seed_length = seed_length,
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
                    seq_length = seq_length,
                    seed_length = seed_length,
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
        num_epochs = 2000,
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
        visualize_step = 200,
        run = run,
        wandb_config = run.config
    )