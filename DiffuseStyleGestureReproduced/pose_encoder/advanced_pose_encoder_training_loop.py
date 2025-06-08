import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

def train_advanced_encoder(
        model,
        pose_training_loader,
        device,
        run=None,
        learning_rate=3e-3,
        num_epochs=1000,
        warm_up_kl_anneal_steps=100,
        kl_anneal_steps=300,
        kl_beta=0.0001,
        visualize_steps=10,
        name="advanced_pose_encoder",
        display_progress=True
    ):
    # Optimization: Turn off anomaly detection for faster training after initial debugging
    # torch.autograd.set_detect_anomaly(True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    losses = []
    reconstruction_losses = []
    kl_losses = []
    component_losses = {}
    
    # Initialize losses for each encoder component
    for name in model.encoders.keys():
        component_losses[name] = []

    if run is not None:
        # Set up logging config
        run_config = {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warm_up_kl_anneal_steps": warm_up_kl_anneal_steps,
            "kl_anneal_steps": kl_anneal_steps,
            "kl_beta": kl_beta,
        }
        
        if hasattr(pose_training_loader.dataset, 'batch_size'):
            run_config["batch_size"] = pose_training_loader.dataset.batch_size
            
        if hasattr(model, 'hyperparameter_dict_to_WnB_tracking'):
            run_config["model_hyperparams"] = model.hyperparameter_dict_to_WnB_tracking
            
        run.config.update(run_config, allow_val_change=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_reconstruction_loss = 0
        train_kl_loss = 0
        component_epoch_losses = {name: 0 for name in model.encoders.keys()}

        # Reshuffle dataset if method exists
        if hasattr(pose_training_loader.dataset, 'reshuffle'):
            pose_training_loader.dataset.reshuffle()
        
        progress_bar = tqdm(pose_training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Extract pose data from batch (handles different dataset formats)
            if len(batch_data) >= 4:
                pose, _, _, _ = [item.squeeze(0).squeeze(1).to(device) for item in batch_data]
            else:
                pose = batch_data[0].to(device)
            
            # Ensure pose is float type
            pose = pose.float()

            # Reset gradients
            optimizer.zero_grad()
            
            # NEW APPROACH: Train only the encoder components directly
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Calculate KL intensity based on current epoch
            kl_intensity = min(1.0, max(0.0, (epoch - warm_up_kl_anneal_steps)) / kl_anneal_steps)
            kl_weight = kl_beta * kl_intensity
            
            # Track mu/logvar for visualization
            all_mu = []
            all_logvar = []
            
            # Process each encoder component separately
            for name, encoder in model.encoders.items():
                # Get component info
                comp = next(comp for comp_name, comp in model.components.items() if comp_name == name)
                indices = comp['indices']
                
                # Extract relevant part of the pose for this component
                component_input = pose[:, indices].clone()  # Clone to avoid in-place modifications
                
                # Encode
                mu, logvar = encoder.encode(component_input, return_logvar=True)
                all_mu.append(mu)
                all_logvar.append(logvar)
                
                # Sample from latent space (reparameterization trick)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Decode
                component_reconstruction = encoder.decode(z)
                
                component_recon_loss = nn.HuberLoss(reduction="mean")(component_reconstruction, component_input)
                
                # Compute KL loss for this component
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_per_dim.mean(dim=0).sum()
                
                # Add weighted loss for this component
                component_loss = component_recon_loss + kl_weight * kl_loss
                total_loss += component_loss
                
                # Track metrics
                total_recon_loss += component_recon_loss.item()
                total_kl_loss += kl_loss.item()
                component_epoch_losses[name] += kl_loss.item()
            
            # Backpropagate and update weights
            total_loss.backward()
            optimizer.step()
            
            # Update tracking
            train_loss += total_loss.item()
            train_reconstruction_loss += total_recon_loss
            train_kl_loss += total_kl_loss * kl_weight
            
            # Update the progress bar
            active_dims_count = sum((mu.var(dim=0) > 0.01).sum().item() for mu in all_mu)
            total_dims = sum(len(mu[0]) for mu in all_mu)
            
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Recon": f"{total_recon_loss:.4f}",
                "KL": f"{(total_kl_loss * kl_weight):.6f}",
                "Active": f"{active_dims_count}/{total_dims}"
            })
            
            # For visualization, we'll need to create full reconstructed pose - but only in visualization epochs
            if epoch % visualize_steps == 0 and display_progress and batch_idx == len(progress_bar) - 1:
                # This is just for visualization, not for training
                with torch.no_grad():
                    try:
                        # Get full pose reconstruction from the model
                        z = model.encode(pose)
                        x_reconstructed = model.decode(z)
                    except:
                        x_reconstructed = pose.clone()  # Fallback if model.decode fails
        
        # Store epoch losses
        dataset_size = len(pose_training_loader.dataset) if hasattr(pose_training_loader.dataset, '__len__') else len(pose_training_loader)
        losses.append(train_loss / len(pose_training_loader))
        reconstruction_losses.append(train_reconstruction_loss / len(pose_training_loader))
        kl_losses.append(train_kl_loss / len(pose_training_loader))
        
        for name in model.encoders.keys():
            component_losses[name].append(component_epoch_losses[name] / len(pose_training_loader))
        
        # Log metrics if run is provided
        if run is not None:
            run_logs = {
                "loss": losses[-1],
                "reconstruction_loss": reconstruction_losses[-1],
                "kl_loss": kl_losses[-1]
            }
            
            for name in model.encoders.keys():
                run_logs[f"kl_loss_{name}"] = component_losses[name][-1]
                
            run.log(run_logs)

        # Visualize training progress
        if epoch % visualize_steps == 0 and display_progress and not is_running_on_slurm():
            try:
                visualize_advanced_training(
                    model,
                    pose,
                    x_reconstructed if 'x_reconstructed' in locals() else pose.clone(),
                    all_mu,
                    all_logvar,
                    losses,
                    reconstruction_losses,
                    kl_losses,
                    component_losses
                )
            except Exception as e:
                print(f"Visualization error: {e}")

    # Save the trained model
    os.makedirs("pose_encoder/models", exist_ok=True)
    torch.save(model.state_dict(), f"pose_encoder/models/{name}.pth")
    print(f"Model saved to pose_encoder/models/{name}.pth")
    return model

def visualize_advanced_training(
        model,
        pose,
        x_reconstructed,
        all_mu,
        all_logvar,
        losses,
        reconstruction_losses,
        kl_losses,
        component_losses
    ):
    """Visualize training progress for advanced encoder."""
    with torch.no_grad():
        clear_output(wait=True)
        
        # Plot pose reconstructions
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 3, 1)
        plt.title("Original Pose")
        plt.imshow(pose[0].repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)
        
        plt.subplot(2, 3, 2)
        plt.title("Reconstructed Pose")
        plt.imshow(x_reconstructed[0].repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)

        difference = pose[0] - x_reconstructed[0]
        plt.subplot(2, 3, 3)
        plt.title("Difference")
        plt.imshow(difference.repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)
        
        # Plot latent variables if we have them
        if all_mu and len(all_mu) > 0:
            try:
                combined_mu = torch.cat([mu[0] for mu in all_mu], dim=0)
                combined_logvar = torch.cat([logvar[0] for logvar in all_logvar], dim=0)
                
                plt.subplot(2, 3, 4)
                plt.title("Latent Space (mu)")
                plt.imshow(combined_mu.unsqueeze(0).repeat(20,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)

                plt.subplot(2, 3, 5)
                plt.title("Latent Space (logvar)")
                plt.imshow(combined_logvar.unsqueeze(0).repeat(20,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)
            except:
                pass
                
        plt.tight_layout()
        plt.show()
        
        # Draw the training loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label="Total Loss", color="blue")
        plt.plot(reconstruction_losses, label="Reconstruction Loss", color="green")
        plt.plot(kl_losses, label="KL Loss", color="red")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Draw component-specific losses
        if component_losses:
            plt.subplot(1, 2, 2)
            for name, comp_loss in component_losses.items():
                if comp_loss:  # Check if list is not empty
                    plt.plot(comp_loss, label=name)
            plt.title("Component KL Losses")
            plt.xlabel("Epoch")
            plt.ylabel("KL Loss")
            plt.legend()
            
        plt.tight_layout()
        plt.show()

def is_running_on_slurm():
    """Check if running in SLURM environment."""
    return 'SLURM_JOB_ID' in os.environ or 'SLURM_JOB_NAME' in os.environ