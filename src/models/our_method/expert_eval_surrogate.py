import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet
import torch.nn.functional as F

from torchvision.models import VisionTransformer


EXPERT_FEATURES = [
    "average_surface_current",
    "coverage_percentage",
    "total_area_extended_shapes",
    "total_len_detected_curves",
    "total_area_circular_shapes",
    "total_defect_area",
    "num_extended_shapes",
    "num_circular_shapes",
    "num_curved_lines",
]


class AvgSurfaceCurrentSurrogate(nn.Module):

    def __init__(self):
        super(AvgSurfaceCurrentSurrogate, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the average surface current of a sample.
        """

        return (torch.mean(x, dim=(1,2,3))).unsqueeze(1)
    

class ExpertSurrogate(nn.Module):
    """
    Predict Celano-Lab characterizations of current-map samples.
    
    - [H, W] -> surrogate -> [NUM_FEATS]
    - [H, W] -> swinir -> surrogate -> [NUM_FEATS]
    """

    def __init__(self, features: list = EXPERT_FEATURES):
        """
        Expert C-AFM current features:
        """
        
        super(ExpertSurrogate, self).__init__()

        self.features = features
        self.num_heads = len(features)
        
        # specialized module that predicts `avg_surface_current`
        self.avg_surface_current_head = AvgSurfaceCurrentSurrogate()
        
        # ---- VGG-19 Feature Extractor ----
        # self.backbone = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        # # remove the final layer
        # self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        
        # ---- ViT-B-16 Feature Extractor ----
        self.backbone = models.vit_b_16()

        # ---- classification heads ----
        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, 1),
            ) for _ in range(self.num_heads)]
        )

        # NOTE: use a specialized module for avg_surface_current
        # if "average_surface_current" in features:
        #     self.heads[0] = self.avg_surface_current_head

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
        
        skip = x.clone()

        # [B, 3, H, W] -> [B, 1000]
        x = self.backbone(x)

        if "average_surface_current" in self.features:
            preds = [None]
            if len(self.heads) > 1:
                preds = [None] + [head(x) for head in self.heads[1: ]]
            preds[0] = self.avg_surface_current_head(skip)
        else:
            preds = [head(x) for head in self.heads]

        out = torch.cat(preds, dim=-1)
        
        # [B, NUM_HEADS]
        return out
    
    @staticmethod
    def get(weights=None): return ExpertSurrogate()


if __name__ == "__main__": 
    pass