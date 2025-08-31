import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pose_encoder.pose_encoder import PoseEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output

def train(
        model: PoseEncoder,
        pose_training_loader: DataLoader,
        device: torch.device,
        run = None,
        learning_rate: float = 3e-3,
        num_epochs: int = 1000,
        warm_up_kl_anneal_steps: int = 100,
        kl_anneal_steps: int = 300,
        kl_beta: float = 0.0001,
        category_weighting: dict[str, float] = {},
        visualize_steps: int = 10,
        name: str = "pose_encoder.pth",
        display_progress: bool = True
    ):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    bone_weighting_vector = pose_training_loader.dataset.skeleton.construct_bone_weighting_vector(category_weighting)

    losses = []
    reconstruction_losses = []
    kl_losses = []
    kl_per_dim_last_epoch = None

    if run is not None:
        run.config.update({
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warm_up_kl_anneal_steps": warm_up_kl_anneal_steps,
            "kl_anneal_steps": kl_anneal_steps,
            "kl_beta": kl_beta,
            "category_weighting": category_weighting,
            "batch_size": pose_training_loader.dataset.batch_size,
            "dataset_type": pose_training_loader.dataset.__class__.__name__,
            "pose_encoder_model": model.get_WnB_config_specs(),
        }, allow_val_change=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_reconstruction_loss = 0
        train_kl_loss = 0

        pose_training_loader.dataset.reshuffle()

        # During the epoch, all the data items are iterated over.        
        progress_bar = tqdm(pose_training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        
        for batch_idx, batch_data in enumerate(progress_bar):

            pose, _, _, _ = [
                item.squeeze(0).squeeze(1).to(device) for item in batch_data
            ]
            
            # pose to float # TODO: Probably dont do this
            pose = pose.float()

            optimizer.zero_grad()
            # Forward pass
            x_reconstructed, mu, logvar, _ = model(pose)
            
            kl_intensity = min(1.0, max(0.0, (epoch - warm_up_kl_anneal_steps)) / kl_anneal_steps)
            kl_weight = kl_beta * kl_intensity

            # Compute loss
            loss, reconstruction_loss, kl_loss, kl_per_dim = vae_loss(x_reconstructed, pose, mu, logvar, bone_weighting_vector, beta=kl_weight)
            kl_per_dim_last_epoch = kl_per_dim
            loss.backward()
            
            # Update weights
            optimizer.step()
            train_loss += loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_kl_loss += kl_loss.item() * kl_beta

            # Update the progress bar with the current loss
            progress_bar.set_postfix({f"\033[91mTrain": f"{loss.item():.4f}\033[0m", 
                                      f"\033[92mRecon": f"{reconstruction_loss.item():.4f}\033[0m",
                                      f"\033[94mKL": f"{kl_loss.item():.1f}, Weighted: {(kl_loss.item() * kl_weight):.9f}, weight: {kl_weight:.9f}, (anneal: {kl_intensity:.9f})\033[0m",
                                      f"\033[95mRatio": f"{((reconstruction_loss.item() / (kl_loss.item() * kl_weight) if kl_weight > 0 else 1.0)):.1f}\033[0m",
                                      f"\033[96mActive dims": f"{(kl_per_dim[0] > 0.01).sum()}/{model.z_dim}\033[0m",
                                      })
        
        losses.append(train_loss / len(pose_training_loader.dataset))
        reconstruction_losses.append(train_reconstruction_loss / len(pose_training_loader.dataset))
        kl_losses.append(train_kl_loss / len(pose_training_loader.dataset))
        
        if run is not None:
            run.log({
                "loss": losses[-1],
                "reconstruction_loss": reconstruction_losses[-1],
                "kl_loss": kl_losses[-1]
            })

        if epoch % visualize_steps == 0 and display_progress and not is_running_on_slurm():
            visualize_training(
                model,
                pose,
                x_reconstructed,
                mu,
                logvar,
                kl_per_dim_last_epoch,
                losses,
                reconstruction_losses,
                kl_losses
            )

    # Saving the trained model
    torch.save(model.state_dict(), "pose_encoder/models/" + name + ".pth")

def visualize_training(
        model: PoseEncoder,
        pose: torch.Tensor,
        x_reconstructed: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_per_dim_last_epoch: torch.Tensor,
        losses: list[float],
        reconstruction_losses: list[float],
        kl_losses: list[float],

    ):
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

    # Draw the training loss
    plt.plot(losses, label=f"Total Loss (min={min(losses):.2f}, max={max(losses):.2f})", color="blue")
    plt.plot(reconstruction_losses, label=f"Reconstruction Loss (min={min(reconstruction_losses):.2f}, max={max(reconstruction_losses):.2f})", color="green")
    plt.plot(kl_losses, label=f"KL Loss (min={min(kl_losses):.2f}, max={max(kl_losses):.2f})", color="red")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Loss function: VAE Loss = Reconstruction Loss + KL Divergence
def vae_loss(x_reconstructed, x, mu, logvar, bone_weights=None, beta=0.5):

    # Use the casted target for all loss calculations
    reconstruction_loss = nn.HuberLoss(reduction="none")(x_reconstructed, x)

    # Applying the bone category weighting
    if bone_weights is not None:
        reconstruction_loss = reconstruction_loss * bone_weights

    # # Now we find the mean
    reconstruction_loss = reconstruction_loss.sum(dim=1).mean()

    # KL Divergence between the learned distribution and the prior (Gaussian)
    # D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where mu, sigma are the mean and log-variance output by the encoder
    # and the prior is a standard normal distribution (mu=0, sigma=1)
    # The KL divergence measures how much our learned distribution differs from the prior
    # We want to minimize this difference, so the encoder outputs z's close to standard normal distribution.
    # A small KL divergence leads to better generalization.

    # Calculate the KL divergence
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # shape: (batch, latent_dim)

    # Mean over batch, sum over dimensions
    kl_divergence = kl_per_dim.mean(dim=0).sum()

    # print("KL loss magnitude: ", KL.item())
    return reconstruction_loss + beta * kl_divergence, reconstruction_loss, kl_divergence, kl_per_dim

# We use to to not draw the plots when running on SLURM
def is_running_on_slurm():
    return 'SLURM_JOB_ID' in os.environ or 'SLURM_JOB_NAME' in os.environ