import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet

from torchvision.models import VisionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

NUM_HEADS = 9


class MultiHeadOLDERSurrogate(nn.Module):
    """
    Predict Celano-Lab characterizations of samples.
    
    - [H, W] -> surrogate -> [9]
    - [H, W] -> swinir -> surrogate -> [9]
    
    TODO:
    - Perform a rigorous study of feature importance;
    - Determine which features are most positively correlated with L1, negatively, spuriously
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
                nn.Linear(4096, 1),
            ) for _ in range(num_heads)
        ])
        # ------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---
        :param y: current-map [H, W]

        Returns
        ---
        older_predicted_value: [1]
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

if __name__ == "__main__": pass