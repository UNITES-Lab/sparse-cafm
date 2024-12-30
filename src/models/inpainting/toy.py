import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, mask):
        return torch.rand((3, 256, 256))