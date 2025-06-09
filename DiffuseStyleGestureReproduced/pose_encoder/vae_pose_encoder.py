import torch
import torch.nn as nn
from pose_encoder.pose_encoder import PoseEncoder

# We use a Variational Autoencoder (beta-VAE) architecture to encode and decode pose data.
# This is highly effective for learning a compact representation of the pose data, which can then be used in the diffusion model.
# We find that it improves the quality of the generated poses significantly compared to using raw pose data directly.
# Another advantage is that the entire latent space becomes continous, and we can essentially sample any point in the latent space and 
# get a reasonable pose. In a sense it becomes impossible to generate unreasonable poses, since the VAE will always map the latent space to a reasonable pose.
class VAEPoseEncoder(PoseEncoder):
    def __init__(self, z_dim=64, pose_dim=345, device=None, checkpoint_path=None):
        super(VAEPoseEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.z_dim = z_dim

        self.hyperparameter_dict_to_WnB_tracking = {
            "z_dim": z_dim,
            "pose_dim": pose_dim,
            "checkpoint_path": checkpoint_path,
            "activation": "GELU",
            "encoder_layers": [pose_dim, 512, 256, 128],
            "decoder_layers": [z_dim, 128, 256, 512, pose_dim],
            "normalization": "None" # "BatchNorm1d"
        }

        activation = nn.GELU

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, 512),
            nn.LayerNorm(512),
            activation(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            activation(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            activation()
        )

        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LayerNorm(128),
            activation(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            activation(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            activation(),
            nn.Linear(512, pose_dim)
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load("pose_encoder/models/" + checkpoint_path, map_location=device))
            print(f"VAE model loaded from {checkpoint_path}")

        if device is not None:
            self.to(device)

    def encode(self, x, return_logvar=False):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        if return_logvar:
            logvar = self.fc_logvar(h)
            return mu, logvar
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x, return_logvar=True)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z

    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking

    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)