from torch.utils.data import DataLoader
from dataset.dataset import *
from pose_encoder.pose_encoder import PoseEncoder
from pose_encoder.pose_encoder_training_loop import train as train_pose_encoder
from v1_training_loop import train as train_sd_model
from v1_model import construct_model
import wandb
from utils.utils import get_device

        
def pose_encoder_sweep():
    
    run = wandb.init()
    config = run.config
    device = get_device()

    train_pose_encoder(
        model=PoseEncoder(
            z_dim=config['z_dim'], 
            pose_dim=config['pose_dim'],
        ),
        pose_training_loader = DataLoader(
            GPUDataset(
                consolidated_file= "dataset/genea2023_dataset/toy/main-agent/consolidated.npz",
                seq_length=1,
                seed_length=0,
                batch_size=config['batch_size'],
                epoch_length=config['epoch_length'],
                device=device
            ),
            batch_size = 1,
            num_workers = 0,
            pin_memory = False  
        ),
        device = device,
        run = run,
        learning_rate = config['learning_rate'],
        num_epochs = config['num_epochs'],
        warm_up_kl_anneal_steps = config['warm_up_kl_anneal_steps'],
        kl_anneal_steps = config['kl_anneal_steps'],
        kl_beta=config['kl_beta'],
        category_weighting = {
            'fingers': config['fingers_weight'],
            'arms': config['arms_weight'],
            'legs': config['legs_weight'],
            'spine': config['spine_weight'],
            'head': config['head_weight'],
            'root': config['root_weight']
        },
        visualize_steps = 20,
        name = run.name,
        display_progress=False
    )

if __name__ == "__main__":

    # Load the key from the file .wanddbkey
    with open(".wandbkey", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    project_name = "v1_sliding_diffusion"
    
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'z_dim': {'values': [32, 64, 128]},
            'pose_dim': {'values': [345]},
            'batch_size': {'values': [32, 64]},
            'epoch_length': {'values': [1000]},
            'learning_rate': {'values': [1e-3, 5e-4, 1e-4]},
            'num_epochs': {'values': [20, 50]},
            'warm_up_kl_anneal_steps': {'values': [100, 500]},
            'kl_anneal_steps': {'values': [1000, 2000]},
            'kl_beta': {'values': [0.01, 0.1, 1.0]},
            'fingers_weight': {'values': [1.0, 2.0]},
            'arms_weight': {'values': [1.0, 2.0]},
            'legs_weight': {'values': [1.0, 2.0]},
            'spine_weight': {'values': [1.0, 2.0]},
            'head_weight': {'values': [1.0, 2.0]},
            'root_weight': {'values': [1.0, 2.0]}
        }
    }
        
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, function=pose_encoder_sweep)
