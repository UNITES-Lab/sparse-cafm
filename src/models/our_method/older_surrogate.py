import torch
import torch.nn as nn

class OlderSurrogate(nn.Module):
    """
    Predict OLDER: [0, inf) from ground-truth current-maps y.
    """
    def __init__(self, ): super(OlderSurrogate, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: pass