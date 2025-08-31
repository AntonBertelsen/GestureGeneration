import wandb

# ----------------- !!MAKING!! THE CONFIG(s) FOR THE SWEEP -----------------

hyper_sweep_config = {
        "method": "random",
        "metric": {"name": "training/total_loss", "goal": "minimize"},
        "parameters": {
            # Model architecture parameters
            "num_transformer_layers": {"values": [4, 6, 8, 12]},
            "number_of_attention_heads": {"values": [4, 8, 16]},
            
            # Training parameters
            "batch_size": {"values": [32, 64, 128, 256, 512]},
            "learning_rate": {"min": 0.00001, "max": 0.0001},
            
            # Loss weights with single values
            "reconstruction_loss_weight": {"value": 5.0},
            "variance_loss_weight": {"value": 0.1},
            "velocity_loss_weight": {"value": 1.0},
            "acceleration_loss_weight": {"value": 1.5},
            "jerk_loss_weight": {"value": 0.2},
            "latent_space_loss_weight": {"value": 0.0}
        }
    }

# ----------------- !!USEING!! THE CONFIG FOR THE SWEEP -----------------


# Login
with open(".wandbkey", "r") as f:
    wandb_key = f.read().strip()

wandb.login(key=wandb_key)


# Start a new sweep with the hyperparameter configuration
# Print, then use the sweep ID in the agent function - just call e.g. sweep_sd_agent.py 
sweep_id = wandb.sweep(hyper_sweep_config, project="hyperSweep_2_attention_batch_size_lr_sliding_diffusion")
print(f"Sweep ID: {sweep_id}")