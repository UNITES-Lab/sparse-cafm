import torch
import torch.nn as nn
from util.loss import DiceLoss, FocalLoss
from torchvision.models import resnet152, swin_b, efficientnet_v2_l, vit_l_16
from torchvision.models import (
    ResNet152_Weights,
    Swin_B_Weights,
    EfficientNet_V2_L_Weights,
    ViT_L_16_Weights,
)

LOSS_FUNCTIONS = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "Dice": DiceLoss,
    "Focal": FocalLoss,
    "Huber": nn.HuberLoss,
}

MODELS = {
    "resnet152": {
        "fn": resnet152,
        "weights": ResNet152_Weights.IMAGENET1K_V2,
    },
    "swin_b": {
        "fn": swin_b,
        "weights": Swin_B_Weights.IMAGENET1K_V1,
    },
    "efficientnet_v2_l": {
        "fn": efficientnet_v2_l,
        "weights": EfficientNet_V2_L_Weights.IMAGENET1K_V1,
    },
    "vit_l_16": {
        "fn": vit_l_16,
        "weights": ViT_L_16_Weights.IMAGENET1K_V1,
    },       
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
