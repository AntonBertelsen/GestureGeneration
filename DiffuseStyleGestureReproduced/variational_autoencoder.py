# Define the VAE model
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=48, pose_dim=345):
        super(VAE, self).__init__()
        self.pose_dim = pose_dim
        self.z_dim = z_dim

        self.hyperparameter_dict_to_WnB_tracking = {
            "pose_dim": pose_dim,
            "z_dim": z_dim,
        }

        # Encoder network
        self.fc1 = nn.Linear(pose_dim, 128)
        # self.dropout = nn.Dropout(0.2)  # Dropout layer
        self.fc2_mu = nn.Linear(128, z_dim)  # Mean of latent space
        self.fc2_logvar = nn.Linear(128, z_dim)  # Log variance of latent space

        self.add_hyperparameters_to_WnB_tracking({"num_of_encoder_layers" : 2})

        # Decoder network
        self.fc3 = nn.Linear(z_dim, 128)
        self.fc4 = nn.Linear(128, pose_dim)

        self.add_hyperparameters_to_WnB_tracking({"num_of_decoder_layers" : 2})


    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        # h1 = self.dropout(h1)  # Apply dropout
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        # h3 = self.dropout(h3)  # Apply dropout
        # x_reconstructed = torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z
    
    # Functions for Weights & Biases tracking
    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict: dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)
    
    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking