import torch
import torch.nn as nn
import torchvision.models as models


class DoGE(nn.Module):
    """
    A Domain Guided Encoder.
    """

    def __init__(self):
        super(DoGE, self).__init__()
        self.encoder = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.encoder(x)

if __name__ == "__main__": pass