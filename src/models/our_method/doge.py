import torch
import torch.nn as nn
import torchvision.models as models


class DoGE(nn.Module):
    """
    A Domain Guided Encoder; returns multi-level features.
    """
    def __init__(self):
        super(DoGE, self).__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        self.features = vgg.features  # This is a Sequential of all conv layers

        self.selected_layers = []
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.ReLU):
                self.selected_layers.append(idx)
    
    def forward(self, x: torch.Tensor) -> list:

        feature_maps = []
        
        # [B, H, W] -> [B, 1, H, W]
        x = x.unsqueeze(1)
        # [B, 1, H, W] -> [B, 3, H, W]
        x = x.repeat(1, 3, 1, 1)

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.selected_layers:
                feature_maps.append(x)

        return feature_maps


if __name__ == "__main__": pass