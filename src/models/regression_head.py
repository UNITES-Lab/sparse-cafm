import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Custom classification head used for predicting the final output value z.
    """

    def __init__(self, in_channels):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))
