import os
import yaml
import torch
import torch.nn as nn

from src.util.loss import DiceLoss, FocalLoss, VAELoss
from torchvision.models import resnet152, swin_b, efficientnet_v2_l, vit_l_16
from torchvision.models import (
    ResNet152_Weights,
    Swin_B_Weights,
    EfficientNet_V2_L_Weights,
    ViT_L_16_Weights,
)
from src.models.simple_z_predictor import SimpleZRegressionVisionTransformer
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.unet.unet import UNet, ThickUNet
from src.models.unetr.unetr import UNETR
from _SwinIR.models.network_swinir import SwinIR


def parse_config(fp: str) -> dict:
    r"""
    Args
        :param fp: path to config file
    Returns
        :return: dict
    """
    assert os.path.isfile(fp), f"Error: config file @ {fp} does not exist"
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


LOSS_FUNCTIONS = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "Dice": DiceLoss,
    "Focal": FocalLoss,
    "Huber": nn.HuberLoss,
    "VAE": VAELoss,
}

MODELS = {
    "simple_z_reg_vit": {
        "fn": SimpleZRegressionVisionTransformer.get,
        "weights": None,
    },
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
    "ae": {"fn": Autoencoder.get, "weights": None},
    "vae": {"fn": VAE.get, "weights": None},
    "unet": {"fn": UNet.get, "weights": None},
    "thick_unet": {"fn": ThickUNet.get, "weights": None},
    "unetr": {"fn": UNETR.get, "weights": None},
    "swinir": {"fn": SwinIR.get, "weights": None},
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
