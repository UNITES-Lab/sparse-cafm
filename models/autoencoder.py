import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIDE_SIZE = 64


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self):
        super(Trim, self).__init__()

    def forward(self, x):
        # Assuming x is in the shape (batch_size, channels, height, width)
        return x[:, :, :IMG_SIDE_SIZE, :IMG_SIDE_SIZE]  # Cropping to 28x28 dimensions

# https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/#
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3 * 64 * 64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3 * 64 * 64),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x: torch.Tensor):
        # (B, C, H, W) -> (B, C * H * W)
        x = x.view(-1, 3 * 64 * 64)
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        decoded = decoded.view(-1, 3, 64, 64)
        return decoded
    
    @staticmethod
    def get(weights=None):
        return AutoEncoder()


if __name__ == "__main__":
    ae = AutoEncoder()
    x = torch.rand((1, 3, 64, 64))
    breakpoint()
