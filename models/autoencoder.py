import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (channels, 64, 64)
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),  # (16, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),        # (32, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),        # (64, 8, 8)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),       # (128, 4, 4)
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (32, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # (16, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1),  # (channels, 64, 64)
            nn.Sigmoid()  # To ensure the output is between 0 and 1
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    @staticmethod
    def get(weights=None):
        return Autoencoder()
    
if __name__ == "__main__":
    pass
