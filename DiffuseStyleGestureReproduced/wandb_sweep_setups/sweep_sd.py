import argparse
from torch.utils.data import DataLoader
from v1_training_loop import train
import wandb
from utils.utils import get_device
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

def sd_model_sweep():
    run = wandb.init(
        entity="pefu-it-university-of-copenhagen", 
        project="hyperSweep_2_attention_batch_size_lr_sliding_diffusion"
    )

    config = run.config
    device = get_device()

    # Fixed sequence length
    seq_length = 100

    train(
        experiment_collection_name = "first_tests",
        upload_model_check_point = False,
        model_checkpoint_dir = "v1_models",
        model = ContinuousMotionModel(
            diffusion = SlidingDiffusion(
                num_clean_frames = 50,
                num_denoise_frames = 50,
                num_noise_frames = 0,
                num_timestep_stackings = 1,
                noise_schedule = Diffusion.linear_schedule(0.00015, 0.15),
                device = device
            ),
            pose_encoder = AdvancedPoseEncoder.load_from_checkpoint("advanced_pose_encoder_ik_pca_64", device),
            number_of_styles = 17,
            gesture_length = 100,
            seed_length = 0,
            audio_features_per_frame = 37,
            pose_features_per_frame = 64,
            condition_mask_probabilty = 0.1,
            number_of_attention_heads = config['number_of_attention_heads'],
            num_transformer_layers = config['num_transformer_layers'],
            predict_full_duration = False,
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
                    consolidated_file = "dataset/genea2023_dataset/trn/main-agent/advanced_encoder/consolidated.npz",
                    seq_length = 100,
                    seed_length = 0,
                    batch_size = config['batch_size'],
                    epoch_length = 1000,
                    loading_encoded_data = True,
                    include_vel_acc_features = False,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False  
            ),
        val_loader = DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/val/main-agent/advanced_encoder/consolidated.npz",
                    seq_length = 100,
                    seed_length = 0,
                    batch_size = 64,
                    epoch_length = 30,
                    loading_encoded_data = True,
                    include_vel_acc_features = False,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False
            ),
        num_epochs = 10000,
        learning_rate = config['learning_rate'],
        reconstruction_loss_weight = config['reconstruction_loss_weight'],
        variance_loss_weight = config['variance_loss_weight'],
        velocity_loss_weight = config['velocity_loss_weight'],
        acceleration_loss_weight = config['acceleration_loss_weight'],
        jerk_loss_weight = config['jerk_loss_weight'],
        latent_space_loss_weight = config['latent_space_loss_weight'],
        visualize_step = 100,
        run = run,
        wandb_config = run.config
    )


if __name__ == "__main__":
    with open(".wandbkey", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    # Initialize the parser, to get the sweep_id from command line arguments
    parser = argparse.ArgumentParser(description="W&B Sweep Agent Launcher")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to join.")
    parser.add_argument("--entity", type=str, default="pefu-it-university-of-copenhagen", help="W&B entity (team or username).")
    parser.add_argument("--project", type=str, default="hyperSweep_2_attention_batch_size_lr_sliding_diffusion", help="W&B project name.")

    args = parser.parse_args()

    if args.sweep_id is None or args.sweep_id == "":
        print("\033[91mWarning: No sweep_id provided. Please provide a sweep_id using --sweep_id.\033[00m")
    else:
        print(f"Joining sweep with ID: {args.sweep_id}")

    sweep_id = args.sweep_id
    wandb.agent(sweep_id, function=sd_model_sweep, entity=args.entity, project=args.project)





"""
FOR LATER VALUE REFS: 


train(
        experiment_collection_name = "first_tests",
        upload_model_check_point = False, # should upload the model checkpoint to wandb
        model_checkpoint_dir = "v1_models",
        model = ContinuousMotionModel(
            diffusion = SlidingDiffusion(
                num_clean_frames = 50,
                num_denoise_frames = 50,
                num_noise_frames = 0,
                num_timestep_stackings = 1,
                noise_schedule = Diffusion.linear_schedule(0.00015, 0.15),
                device = device
            ),
            # diffusion = NormalDiffusion(
            #     num_timesteps = 100,
            #     sequence_length = 100,
            #     noise_schedule = Diffusion.linear_schedule(0.00015, 0.075),
            #     device = device
            # ),
            pose_encoder = AdvancedPoseEncoder.load_from_checkpoint("advanced_pose_encoder_ik_pca_64", device),
            number_of_styles = 17,
            gesture_length = 100,
            seed_length = 0,
            audio_features_per_frame = 37,
            pose_features_per_frame = 64,
            condition_mask_probabilty = 0.1,
            number_of_attention_heads = 8,
            num_transformer_layers = 6,
            predict_full_duration = False,
            reinject_seed_style_t = False,
            debugger = Debugger(
                on = False, 
                keys_for_printing_while_running = ["ALL"]
            ),
            device = device
        ),
        device = device,
        training_loader = DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/trn/main-agent/advanced_encoder/consolidated.npz",
                    seq_length = 100,
                    seed_length = 0,
                    batch_size = 256,
                    epoch_length = 1000,
                    loading_encoded_data = True,
                    include_vel_acc_features = False,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False  
            ),
        val_loader = DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/val/main-agent/advanced_encoder/consolidated.npz",
                    seq_length = 100,
                    seed_length = 0,
                    batch_size = 64,
                    epoch_length = 30,
                    loading_encoded_data = True,
                    include_vel_acc_features = False,
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
        # category_weighting = {
        #     'fingers': 0.1,
        #     'arms': 2.0,
        #     'legs': 1.0,
        #     'spine': 2.0,
        #     'head': 1.0,
        #     'root': 2.0
        # },
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
"""