import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet
import torch.nn.functional as F

from torchvision.models import VisionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

NUM_HEADS = 1


class MultiHeadOLDERSurrogate(nn.Module):
    """
    Predict Celano-Lab characterizations of current-map samples.
    
    - [H, W] -> surrogate -> [NUM_FEATS]
    - [H, W] -> swinir -> surrogate -> [NUM_FEATS]
    """

    def __init__(self, num_heads: int = NUM_HEADS):
        
        super(MultiHeadOLDERSurrogate, self).__init__()
        self.num_heads = num_heads
        
        # ---- VGG-19 Feature Extractor ----
        self.backbone = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        
        # remove the final layer
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])

        # ---- classification heads ----
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, 1),
            ) for _ in range(num_heads)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---
        :param y: current-map [H, W]

        Returns
        ---
        older_predicted_value: [NUM_FEATS]
        """
        
        # [B, H, W] -> [B, 224, 224]
        if isinstance(self.backbone, torchvision.models.vision_transformer.VisionTransformer):
            x = torchvision.transforms.Resize((224, 224))(x)
        # [B, H, W] -> [B, 1, H, W]
        x = x.unsqueeze(1)
        # [B, 1, H, W] -> [B, 3, H, W]
        x = x.repeat(1, 3, 1, 1)
        # [B, 3, H, W] -> [B, 1000]
        x = self.backbone(x)
        # predict each characteristic
        preds = [head(x) for head in self.heads]
        out = torch.cat(preds, dim=-1)
        # [B, NUM_HEADS]
        return out
    
    @staticmethod
    def get(weights=None): return MultiHeadOLDERSurrogate()


# https://pytorch-enhance.readthedocs.io/en/latest/_modules/torch_enhance/losses/vgg.html
class OLDERPerceptualLoss(nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, surrogate: MultiHeadOLDERSurrogate, conv_index: str = '22'):
        super(OLDERPerceptualLoss, self).__init__()
        self.vgg = surrogate.backbone
        vgg_features = self.vgg.features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        self.vgg.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        def _forward(x):
            #x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
        loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss

if __name__ == "__main__": pass