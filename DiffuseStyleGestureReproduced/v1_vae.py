import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.decomposition import PCA
import torch.nn.functional as F
from dataset.dataset import *
from IPython.display import clear_output


def vae_loss_function(
        x_reconstructed, 
        x, 
        mu, 
        logvar, 
        reconstruction_loss_f,
        bone_weights, 
        KL_beta, 
        free_bits):
    
    # Use the casted target for all loss calculations
    reconstruction_loss = reconstruction_loss_f(x_reconstructed, x)

    # Applying the bone category weighting
    if bone_weights is not None:
        reconstruction_loss = reconstruction_loss * bone_weights
    # print(f"reconstruction_loss shape after weighting: {reconstruction_loss.shape}")

    # # Now we find the mean
    reconstruction_loss = reconstruction_loss.sum(dim=1).mean()
    # print(f"Loss shape after meaning: {reconstruction_loss.shape}")



    # KL Divergence between the learned distribution and the prior (Gaussian)
    # D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where mu, sigma are the mean and log-variance output by the encoder
    # and the prior is a standard normal distribution (mu=0, sigma=1)
    # The KL divergence measures how much our learned distribution differs from the prior
    # We want to minimize this difference, so the encoder outputs z's close to standard normal distribution.
    # A small KL divergence leads to better generalization.

    # Calculate the KL divergence
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    # print("reconstion loss magnitude: ", reconstruction_loss.item())
    # KL = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # shape: (batch, latent_dim)

    # Free bits threshold: allow minimum KL per dimension (in nats)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Mean over batch, sum over dimensions
    kl_divergence = kl_per_dim.mean(dim=0).sum()

    # print("KL loss magnitude: ", KL.item())
    return reconstruction_loss + KL_beta * kl_divergence, reconstruction_loss, kl_divergence, kl_per_dim

def train_vae(
        device,
        pose_training_loader,
        model,
        run = None,
        wandb_config = None,
        bone_category_weights: dict = {
            'fingers': 0.1,
            'arms': 2.0,
            'legs': 1.0,
            'spine': 2.0,
            'head': 1.0,
            'root': 2.0
        },
        lr=1e-4,
        optimizer_f = optim.Adam,
        num_epochs = 1000,
        warm_up_kl_anneal_steps = 20,
        kl_anneal_steps = 200,
        free_bits = 0.0,  # Free bits threshold for KL divergence
        kl_beta=0.0015,
        visualize_steps = 20, # Number of steps to visualize the training process
        model_save_dir = "vae_model",
        model_save_name = "vae_model",
):
    model.to(device)
    model.train()

    optimizer = optimizer_f(model.parameters(), lr=lr)

    if wandb_config is not None:
        wandb_config.update({
            # Training hyper parameters
            "batch_size": pose_training_loader.batch_size,
            "epochs": num_epochs,
            "optimizer": optimizer.__class__.__name__,

            # Data hyper params:
            "dataset_type": pose_training_loader.__class__.__name__,

            # training_loader hyper parameters
            **model.get_WnB_config_specs(),
        }, allow_val_change=True)  # <- Important: allows adding/updating keys


    skeleton_info = pose_training_loader.dataset.skeleton_info
    num_features = skeleton_info['number_of_features']
    bone_index_weighted_by_category_vector = torch.ones(num_features)

    bone_index_weighted_by_category_vector = bone_index_weighted_by_category_vector.to(device)

    # Assign weights based on the categories
    for category, weight in bone_category_weights.items():
        # Check if the category exists in the skeleton info
        if category not in skeleton_info['bone_categories']:
            print(f"Warning!!! Category '{category}' not found in skeleton info. Skipping.")
            continue
        for bone_name in skeleton_info['bone_categories'][category]:
            bone_indices = skeleton_info['bone_to_indices'][bone_name]
            # Check if bone exists in the skeleton info
            if bone_indices is None:
                print(f"Warning!!! Bone '{bone_name}' not found in skeleton info. Skipping.")
                continue
            for index in bone_indices:
                bone_index_weighted_by_category_vector[index] = weight

    # Normalize the weights to sum to 1. This is to prevent the loss from being too large or too small compared to the KL divergence term
    bone_index_weighted_by_category_vector /= bone_index_weighted_by_category_vector.sum()

    # For logging purposes, we predefine:
    losses = []
    reconstruction_losses = []
    kl_losses = []
    kl_per_dim_last_epoch = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_reconstruction_loss = 0
        train_kl_loss = 0

        pose_training_loader.dataset.reshuffle()

        for batch_idx, batch_data in enumerate(pose_training_loader):

            pose, _, _, _ = [
                item.squeeze(0).squeeze(1).to(device) for item in batch_data
            ]
            
            # pose to float # TODO: Probably dont do this
            pose = pose.float()

            optimizer.zero_grad()
            # Forward pass
            x_reconstructed, mu, logvar, z = model(pose)
            
            kl_weight = kl_beta * torch.sigmoid(
                torch.tensor(min(1.0, max(0.0, (epoch - warm_up_kl_anneal_steps) / kl_anneal_steps)))
            )


            # Compute loss
            loss, reconstruction_loss, kl_loss, kl_per_dim = vae_loss_function(x_reconstructed, pose, mu, logvar, bone_index_weighted_by_category_vector, beta=kl_weight, free_bits=free_bits)
            # loss, reconstruction_loss, kl_loss, total_correlation, kl_per_dim = tc_vae_loss(x_reconstructed, pose, mu, logvar, z, beta=0.001, beta_tc=5.0, beta_kl=0.1, beta_mi=0.5, bone_weights=bone_index_weighted_by_category_vector)
            kl_per_dim_last_epoch = kl_per_dim
            
            if run is not None: 
                step = i + epoch * len(pose_training_loader)
                run.log({
                        "total_loss": loss, 
                        "kl_loss": kl_loss,
                        "reconstruction_loss": reconstruction_loss,
                    }, 
                    step=step)

            # Compute the gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            train_loss += loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_kl_loss += kl_loss.item() * kl_beta
        
        losses.append(train_loss / len(pose_training_loader.dataset))
        reconstruction_losses.append(train_reconstruction_loss / len(pose_training_loader.dataset))
        kl_losses.append(train_kl_loss / len(pose_training_loader.dataset))
        
        if epoch % visualize_steps == 0:

            with torch.no_grad():
                clear_output(wait=True)
                plt.figure(figsize=(10, 5))
                plt.subplot(2, 3, 1)
                plt.title("Original")
                plt.text(10, 45, f"mean: {pose[0].mean().item():.2f}, \nstd: {pose[0].std().item():.2f}, \nmin: {pose[0].min().item():.2f}, \nmax: {pose[0].max().item():.2f}", fontsize=9, ha='left', va='center')
                plt.imshow(pose[0].repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)
                
                plt.subplot(2, 3, 2)
                plt.title("Reconstructed")
                plt.text(10, 45, f"mean: {x_reconstructed[0].mean().item():.2f}, \nstd: {x_reconstructed[0].std().item():.2f}, \nmin: {x_reconstructed[0].min().item():.2f}, \nmax: {x_reconstructed[0].max().item():.2f}", fontsize=9, ha='left', va='center')
                plt.imshow(x_reconstructed[0].repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)

                difference = pose[0] - x_reconstructed[0]
                plt.subplot(2, 3, 3)
                plt.title("difference")
                plt.text(10, 45, f"mean: {difference.mean().item():.2f}, \nstd: {difference.std().item():.2f}, \nmin: {difference.min().item():.2f}, \nmax: {difference.max().item():.2f}", fontsize=9, ha='left', va='center')
                plt.imshow((difference).repeat(200,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)

                plt.subplot(2, 3, 4)
                plt.title("z_mu")
                plt.text(1, 5, f"mean: {mu.mean().item():.2f}, \nstd: {mu.std().item():.2f}, \nmin: {mu.min().item():.2f}, \nmax: {mu.max().item():.2f}", fontsize=9, ha='left', va='center')
                plt.imshow(mu[0].repeat(20,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)

                plt.subplot(2, 3, 5)
                plt.title("z_logvar")
                plt.text(1, 5, f"mean: {logvar.mean().item():.2f}, \nstd: {logvar.std().item():.2f}, \nmin: {logvar.min().item():.2f}, \nmax: {logvar.max().item():.2f}", fontsize=9, ha='left', va='center')
                plt.imshow(logvar[0].repeat(20,1).cpu().detach().numpy(), cmap='viridis', vmin=-3, vmax=3)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 5))
                mu_var = mu.var(dim=0).cpu().numpy()
                plt.subplot(1, 2, 1)
                plt.title("Latent Dimension Usage (Variance per Dimension)")
                plt.bar(range(len(mu_var)), mu_var)

                kl_per_dim_last_epoch = kl_per_dim_last_epoch[0].cpu().numpy()
                plt.subplot(1, 2, 2)
                plt.bar(range(len(kl_per_dim_last_epoch)), kl_per_dim_last_epoch)
                plt.xlabel("Latent Dimension")
                plt.ylabel("KL Divergence (nats)")
                plt.title("KL per Dimension")
                plt.axhline(y=free_bits, color='r', linestyle='--', label='Free Bits Threshold')
                plt.legend()
                plt.show()

                latent_samples = model.reparameterize(mu, logvar)

                pca = PCA(n_components=2)
                latent_2d = pca.fit_transform(latent_samples.cpu().numpy())
                plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
                plt.title('Latent Space PCA Projection')
                plt.show()

                # After obtaining latent variables (mu or z)
                plt.hist(mu.detach().cpu().numpy(), bins=100, alpha=0.7)
                plt.title('Distribution of Latent Variables')
                plt.show()

            def normalize_list(data_list):
                epsilon = 1e-8  # Small constant to prevent division by zero
                min_val = min(data_list)
                max_val = max(data_list)
                return [(x - min_val) / (max_val - min_val + epsilon) for x in data_list]

            # normalized_losses = normalize_list(losses)
            # normalized_reconstruction_losses = normalize_list(reconstruction_losses)
            # normalized_kl_losses = normalize_list(kl_losses)

            # Draw the training loss
            plt.plot(losses, label=f"Total Loss (min={min(losses):.2f}, max={max(losses):.2f})", color="blue")
            plt.plot(reconstruction_losses, label=f"Reconstruction Loss (min={min(reconstruction_losses):.2f}, max={max(reconstruction_losses):.2f})", color="green")
            plt.plot(kl_losses, label=f"KL Loss (min={min(kl_losses):.2f}, max={max(kl_losses):.2f})", color="red")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            active_dims = (kl_per_dim_last_epoch > 0.01).sum()
            print(f"Active dimensions: {active_dims}/{model.z_dim}")

            # As an example, print the value of the 50th pixel for both the original and reconstructed images
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss / len(pose_training_loader.dataset):.4f}")
            print(f"Reconstruction Loss: {train_reconstruction_loss / len(pose_training_loader.dataset):.4f}")
            print(f"KL Loss: {train_kl_loss / len(pose_training_loader.dataset):.4f}")
            print(f"Used KL weight: {kl_weight:.4f}")

            recon = reconstruction_loss.item()
            ratio = kl_weight / recon
            print(f"KL Weight: {kl_weight:.4f}, Reconstruction Loss: {recon:.4f}, Ratio: {ratio:.4f}")

            
    # Saving the trained model
    model_save_path = f"{model_save_dir}/{model_save_name}.pth"

    torch.save(model.state_dict(), model_save_path)