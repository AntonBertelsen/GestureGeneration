import torch
import torch.nn as nn

from utils.WnB_trackable import WnBTrackable

# We use a Variational Autoencoder (beta-VAE) architecture to encode and decode pose data.
# This is highly effective for learning a compact representation of the pose data, which can then be used in the diffusion model.
# We find that it improves the quality of the generated poses significantly compared to using raw pose data directly.
# Another advantage is that the entire latent space becomes continous, and we can essentially sample any point in the latent space and 
# get a reasonable pose. In a sense it becomes impossible to generate unreasonable poses, since the VAE will always map the latent space to a reasonable pose.
class PoseEncoder(nn.Module, WnBTrackable):
    def __init__(self, z_dim=32, pose_dim=345, device=None, checkpoint_path=None):
        super(PoseEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.z_dim = z_dim

        # Encoder network
        self.fc1 = nn.Linear(pose_dim, 128)
        self.fc2_mu = nn.Linear(128, z_dim)  # Mean of latent space
        self.fc2_logvar = nn.Linear(128, z_dim)  # Log variance of latent space

        # Decoder network
        self.fc3 = nn.Linear(z_dim, 128)
        self.fc4 = nn.Linear(128, pose_dim)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load("pose_encoder/models/" + checkpoint_path, map_location=device))
            print(f"VAE model loaded from {checkpoint_path}")

        self.hyperparameter_dict_to_WnB_tracking = {
            "z_dim": z_dim,
            "pose_dim": pose_dim,
            "checkpoint_path": checkpoint_path
        }

        if device is not None:
            self.to(device)

    def encode(self, x, return_logvar=False):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        if not return_logvar:
            return mu
        else:
            logvar = self.fc2_logvar(h1)
            return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x, return_logvar=True)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z
    
    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking