from torch.utils.data import DataLoader
from dataset.dataset import *
from pose_encoder.pose_encoder import PoseEncoder
from pose_encoder.pose_encoder_training_loop import train as train_pose_encoder
from v1_training_loop import train as train_sd_model
from v1_model import construct_model
import wandb
from utils.utils import get_device

def sd_model_sweep():
    run = wandb.init()
    config = run.config
    device = get_device()

    train_sd_model(
        experiment_collection_name = run.group,
        upload_model_check_point = False,
        model_checkpoint_dir = "v1_sweep_models",
        model = construct_model(config['model']),
        device = device,
        training_loader = DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/trn/main-agent/consolidated.npz",
                    seq_length = config['seq_length'],
                    seed_length = 0,
                    batch_size = config['batch_size'],
                    epoch_length = config['epoch_length'],
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False  
            ),
        val_loader=DataLoader(
                GPUDataset(
                    consolidated_file = "dataset/genea2023_dataset/val/main-agent/consolidated.npz",
                    seq_length = config['seq_length'],
                    seed_length = 0,
                    batch_size = config['batch_size'],
                    epoch_length = 30,
                    device = device
                ),
                batch_size = 1,
                num_workers = 0,
                pin_memory = False
            ),
        num_epochs = config['num_epochs'],
        learning_rate = config['learning_rate'],
        reconstruction_loss_weight = config['reconstruction_loss_weight'],
        variance_loss_weight = config['variance_loss_weight'],
        velocity_loss_weight = config['velocity_loss_weight'],
        acceleration_loss_weight = config['acceleration_loss_weight'],
        latent_space_loss_weight = config['latent_space_loss_weight'],
        category_weighting = {
            'fingers': config['fingers_weight'],
            'arms': config['arms_weight'],
            'legs': config['legs_weight'],
            'spine': config['spine_weight'],
            'head': config['head_weight'],
            'root': config['root_weight']
        },
        frame_weighting_segments_info = [
            (0.1, 0.2, 25),
            (0.2, 0.4, 40),
            (0.4, 1.0, 50),
            (1.0, 1.0, 55),
            (1.0, 0.85, 80),
            (0.85, 0.6, 100)
        ],
        visualize_step = 100,
        continue_from_checkpoint="latest"
    )

if __name__ == "__main__":
    # Load the key from the file .wanddbkey
    with open(".wandbkey", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    project_name = "v1_sliding_diffusion"
    
    sweep_config = {
        "method": "random",
        "metric": {"name": "total_loss", "goal": "minimize"},
        "parameters": {
            # Training parameters
            "learning_rate": {"min": 0.00001, "max": 0.001},
            "num_epochs": {"values": [500, 1000, 1500]},
            "batch_size": {"values": [128, 256, 512]},
            "seq_length": {"values": [50, 100, 150]},
            "epoch_length": {"values": [500, 1000, 2000]},
            
            # Loss weights
            "reconstruction_loss_weight": {"min": 0.5, "max": 2.0},
            "variance_loss_weight": {"min": 0.01, "max": 0.5},
            "velocity_loss_weight": {"min": 0.5, "max": 2.0},
            "acceleration_loss_weight": {"min": 1.0, "max": 4.0},
            "latent_space_loss_weight": {"min": 2.0, "max": 8.0},
            
            # Category weighting parameters
            "fingers_weight": {"min": 0.05, "max": 0.5},
            "arms_weight": {"min": 1.0, "max": 4.0},
            "legs_weight": {"min": 0.5, "max": 2.0},
            "spine_weight": {"min": 1.0, "max": 4.0},
            "head_weight": {"min": 0.5, "max": 2.0},
            "root_weight": {"min": 1.0, "max": 4.0},
            
            # Nested model parameters
            "model": {
                "parameters": {
                    # General model parameters
                    "n_gesture_length": {"values": [50, 100, 150]},
                    "audio_features_per_frame": {"value": 37},  # Fixed based on dataset
                    "pose_features_per_frame": {"value": 64},   # From pose encoder
                    "number_of_styles": {"value": 17},          # Fixed based on dataset
                    "condition_mask_probabilty": {"min": 0.05, "max": 0.2},
                    "number_of_attention_heads": {"values": [4, 8, 16]},
                    "predict_full_duration": {"value": False},
                    
                    # Diffusion parameters
                    "diffusion": {
                        "parameters": {
                            "num_of_pre_timestep_frames": {"values": [25, 50, 75]},
                            "num_of_timestep_frames": {"values": [25, 50, 75]},
                            "num_of_post_timestep_frames": {"values": [0, 25, 50]},
                            "beta_min": {"min": 0.0001, "max": 0.001},
                            "beta_max": {"min": 0.1, "max": 0.2},
                            "name": {"value": "linear_schedule"},
                            "num_of_timestep_stackings": {"values": [1, 2, 3]}
                        }
                    },
                    
                    # Pose encoder parameters (if used)
                    "pose_encoder": {
                        "parameters": {
                            "z_dim": {"values": [32, 64, 128]},
                            "pose_dim": {"value": 345}  # Fixed based on dataset
                        }
                    }
                }
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, function=sd_model_sweep)
