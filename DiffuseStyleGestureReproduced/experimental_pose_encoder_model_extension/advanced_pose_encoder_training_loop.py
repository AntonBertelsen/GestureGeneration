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
        category_weighting={},
        visualize_steps=10,
        model_name="advanced_pose_encoder",
        display_progress=True
    ):
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
        
        # For visualization, track the last batch's data
        visualization_data = None
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Extract pose data
            if len(batch_data) >= 4:
                pose, _, _, _ = [item.squeeze(0).squeeze(1).to(device) for item in batch_data]
            else:
                pose = batch_data[0].to(device)
            
            # Ensure pose is float type
            pose = pose.float()

            # Reset gradients
            optimizer.zero_grad()
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Calculate KL intensity based on current epoch
            kl_intensity = min(1.0, max(0.0, (epoch - warm_up_kl_anneal_steps)) / kl_anneal_steps)
            kl_weight = kl_beta * kl_intensity
            
            # Track mu/logvar and component data for visualization
            all_mu = []
            all_logvar = []
            component_originals = {}
            component_reconstructions = {}
            
            # Process each encoder component separately
            for name, encoder in model.encoders.items():
                # Get component info
                comp = next(comp for comp_name, comp in model.components.items() if comp_name == name)
                indices = comp['indices']
                
                # Extract relevant part of the pose for this component
                component_input = pose[:, indices].clone()
                
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
                
                # Store component data for visualization
                component_originals[name] = component_input
                component_reconstructions[name] = component_reconstruction
                
                # Calculate reconstruction loss
                component_recon_loss = nn.HuberLoss(reduction="mean")(component_reconstruction, component_input)
                
                # Compute KL loss
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
            
            # Save visualization data from last batch
            if batch_idx == len(progress_bar) - 1:
                visualization_data = {
                    'component_originals': component_originals,
                    'component_reconstructions': component_reconstructions,
                    'all_mu': all_mu,
                    'all_logvar': all_logvar
                }
            
            # Update the progress bar
            active_dims_count = sum((mu.var(dim=0) > 0.01).sum().item() for mu in all_mu)
            total_dims = sum(len(mu[0]) for mu in all_mu)
            
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Recon": f"{total_recon_loss:.4f}",
                "KL": f"{(total_kl_loss * kl_weight):.6f}",
                "Active": f"{active_dims_count}/{total_dims}"
            })
        
        # Store epoch losses
        losses.append(train_loss / len(pose_training_loader))
        reconstruction_losses.append(train_reconstruction_loss / len(pose_training_loader))
        kl_losses.append(train_kl_loss / len(pose_training_loader))
        
        for name in model.encoders.keys():
            component_losses[name].append(component_epoch_losses[name] / len(pose_training_loader))
        
        # Log metrics
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
        if epoch % visualize_steps == 0 and display_progress and not is_running_on_slurm() and visualization_data:
            try:
                visualize_component_training(
                    model,
                    visualization_data['component_originals'],
                    visualization_data['component_reconstructions'],
                    visualization_data['all_mu'],
                    visualization_data['all_logvar'],
                    losses,
                    reconstruction_losses,
                    kl_losses,
                    component_losses
                )
            except Exception as e:
                print(f"Visualization error: {e}")

    # Save the trained model

    os.makedirs("pose_encoder/models", exist_ok=True)
    # Collect all data needed for reconstruction
    save_data = {
        'state_dict': model.state_dict(),
        'hyperparameters': model.hyperparameter_dict_to_WnB_tracking,
        'skeleton': model.skeleton
    }
    
    # Save everything in a single file
    torch.save(save_data, f"pose_encoder/models/{model_name}.pth")
    print(f"Model saved to pose_encoder/models/{model_name}.pth")
    return model

def visualize_component_training(
        model,
        component_originals,
        component_reconstructions,
        all_mu,
        all_logvar,
        losses,
        reconstruction_losses,
        kl_losses,
        component_losses
    ):
    """Visualize training progress for component encoders."""
    with torch.no_grad():
        clear_output(wait=True)
        
        # Create concatenated views of original and reconstructed parts
        component_names = list(component_originals.keys())
        
        # Get sample from first batch for visualization
        stacked_originals = torch.cat([component_originals[name][0] for name in component_names], dim=0)
        stacked_reconstructions = torch.cat([component_reconstructions[name][0] for name in component_names], dim=0)
        stacked_difference = stacked_originals - stacked_reconstructions
        
        # Plot component reconstructions
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Components")
        plt.imshow(stacked_originals.unsqueeze(0).repeat(100, 1).cpu().numpy(), 
                  cmap='viridis', aspect='auto', vmin=-3, vmax=3)
        
        plt.subplot(1, 3, 2)
        plt.title("Reconstructed Components")
        plt.imshow(stacked_reconstructions.unsqueeze(0).repeat(100, 1).cpu().numpy(), 
                  cmap='viridis', aspect='auto', vmin=-3, vmax=3)

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(stacked_difference.unsqueeze(0).repeat(100, 1).cpu().numpy(), 
                  cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add component labels
        component_widths = [len(component_originals[name][0]) for name in component_names]
        component_positions = [sum(component_widths[:i]) + component_widths[i]/2 for i in range(len(component_widths))]
        
        for ax_idx in [0, 1, 2]:
            plt.subplot(1, 3, ax_idx+1)
            for i, name in enumerate(component_names):
                plt.axvline(x=sum(component_widths[:i]), color='white', linestyle='--', alpha=0.5)
                plt.text(component_positions[i], 90, name, 
                        horizontalalignment='center', verticalalignment='center', 
                        rotation=90, color='white', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Plot latent spaces
        if all_mu and len(all_mu) > 0:
            plt.figure(figsize=(12, 4))
            
            # Plot combined latent mu for all components
            plt.subplot(1, 2, 1)
            plt.title("Latent Space (mu)")
            
            # Calculate positions for component labels
            z_dims = [mu.shape[1] for mu in all_mu]
            z_positions = [sum(z_dims[:i]) + z_dims[i]/2 for i in range(len(z_dims))]
            
            # Stack all mu values for visualization
            stacked_mu = torch.cat([mu[0] for mu in all_mu], dim=0).cpu().numpy()
            plt.bar(range(len(stacked_mu)), stacked_mu)
            plt.grid(True, alpha=0.3)
            
            # Add component dividers and labels
            for i, name in enumerate(component_names):
                plt.axvline(x=sum(z_dims[:i])-0.5, color='red', linestyle='--', alpha=0.5)
                plt.text(z_positions[i], min(stacked_mu)-0.5, name, 
                        horizontalalignment='center', color='red', fontsize=8)
            
            # KL loss per component
            plt.subplot(1, 2, 2)
            plt.title("Component KL Losses")
            for name, loss_values in component_losses.items():
                if loss_values:  # Check if list is not empty
                    plt.plot(loss_values, label=name)
            plt.xlabel("Epoch")
            plt.ylabel("KL Loss")
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Draw the training loss
        plt.figure(figsize=(12, 4))
        plt.plot(losses, label="Total Loss", color="blue")
        plt.plot(reconstruction_losses, label="Reconstruction Loss", color="green")
        plt.plot(kl_losses, label="KL Loss", color="red")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def is_running_on_slurm():
    """Check if running in SLURM environment."""
    return 'SLURM_JOB_ID' in os.environ or 'SLURM_JOB_NAME' in os.environ