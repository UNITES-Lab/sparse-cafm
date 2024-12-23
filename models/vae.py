# with <3 to...
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/lvae.py
# https://mbernste.github.io/posts/vae/

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim=50):
        super(VAE, self).__init__()

        # Define autoencoding layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1
        )  # 32x32x32
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 64x16x16
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 128x8x8
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # 256x4x4

        # Define autoencoding layers
        self.enc_fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.enc_fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

        # Decoder: Fully connected layer to expand latent vector
        self.dec_fc = nn.Linear(z_dim, 256 * 4 * 4)
        
        self.flatten = nn.Flatten()

        # Decoder: Transposed convolutional layers
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 128x8x8
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 64x16x16
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 32x32x32
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)  # 3x64x64

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = self.flatten(x)
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decoder(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 4, 4)  # Reshape to (batch_size, 256, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.tanh(self.dec_conv4(x))  # Sigmoid for output between 0 and 1
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar
    
    @staticmethod
    def get(weights=None):
        return VAE()


# Define the loss function
def vae_loss_function(output, x, mu, logvar):
    # reconstruction loss
    recon_loss = F.mse_loss(output, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + 0.002 * kl_loss) * .001
