import os
import yaml
import torch
import torch.nn as nn

from src.util.loss import DiceLoss, FocalLoss, VAELoss, ImageInpaintingL1Loss
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
from src.models.classic_recon import (
    LinearInterpolationInpainter,
    BicubicInterpolationInpainter,
    AMPInpainter,
    NearestNeighborsInpainter,
)
from src.models.our_method.swin_cafm import SwinCAFM
from src.models.prev_methods.sstem import SSTEM
from src.models.prev_methods.gpstruct import GPSTRUCT
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
    "InpaintingL1": ImageInpaintingL1Loss,
}

MODELS = {
    "ours": {
        "fn": SwinCAFM.get,
        "weights": None,
    },
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
    "linear_interpolation": {"fn": LinearInterpolationInpainter.get, "weights": None},
    "bicubic_interpolation": {"fn": BicubicInterpolationInpainter.get, "weights": None},
    "amp_interpolation": {"fn": AMPInpainter.get, "weights": None},
    "nn_interpolation": {"fn": NearestNeighborsInpainter.get, "weights": None},
    "sstem_interpolation": {"fn": SSTEM.get, "weights": None},
    "gpstruct_interpolation": {"fn": GPSTRUCT.get, "weights": None},
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

import os
import yaml


def parse_config(config_fp: str) -> dict:
    """
    Parse the YAML configuration file and return a dictionary.
    """
    with open(config_fp, "r") as f:
        return yaml.safe_load(f)


class TrainConfig:
    def __init__(self, config_fp: str):

        # Ensure the configuration file exists
        if not os.path.isfile(config_fp):
            raise FileNotFoundError(f"Config file not found: {config_fp}")

        # Load the configuration file into a dictionary
        config_dict = parse_config(config_fp)

        # --- Global settings ---
        global_cfg = config_dict.get("global", {})
        self.device = global_cfg.get("device", 0)
        self.mode = global_cfg.get("mode", "train")
        self.formulation = global_cfg.get("formulation", None)

        # --- Model settings ---
        model_cfg = config_dict.get("model", {})
        self.model_name = model_cfg.get("name", "")
        self.pretrained = model_cfg.get("pretrained", False)
        self.weights = model_cfg.get("weights", None)
        self.model_config_file = model_cfg.get("config", None)

        # --- Training settings ---
        training_cfg = config_dict.get("training", {})
        self.train_batch_size = training_cfg.get("batch_size", 1)
        self.steps_per_epoch = training_cfg.get("steps_per_epoch", 1024)
        self.epochs = training_cfg.get("epochs", 2000)
        self.train_loss = training_cfg.get("loss", None)
        self.learning_rate = training_cfg.get("lr", 1e-4)
        self.optimizer = training_cfg.get("optimizer", "Adam")

        # --- Validation settings ---
        validation_cfg = config_dict.get("validation", {})
        self.val_batch_size = validation_cfg.get("batch_size", 1)
        self.val_steps_per_epoch = validation_cfg.get("steps_per_epoch", 256)
        self.val_loss = validation_cfg.get("loss", None)

        # --- Dataset settings ---
        dataset_cfg = config_dict.get("dataset", {})
        self.dataset_name = dataset_cfg.get("name", "")
        self.image_size = dataset_cfg.get("image_size", None)
        self.crop_size = dataset_cfg.get("crop_size", None)
        self.num_workers = dataset_cfg.get("num_workers", 0)
        self.masking_ratio = dataset_cfg.get("masking_ratio", 1)

        # --- Logging settings ---
        logging_cfg = config_dict.get("logging", {})
        self.log_root = logging_cfg.get("root", "")
        self.exp_name = logging_cfg.get("exp_name", "")
        self.result_columns = logging_cfg.get("result_columns", [])
        self.save_weights = logging_cfg.get("save_weights", True)
        self.save_only_best_weights = logging_cfg.get("save_only_best_weights", True)
        self.enable_tensorboard = logging_cfg.get("enable_tensorboard", False)
        self.log_figures = logging_cfg.get("log_figures", False)
        self.log_interval = logging_cfg.get("log_interval", 1)

    def __repr__(self):
        """
        Provide a string representation of the configuration for debugging.
        """
        return (
            f"TrainConfig(device={self.device}, mode='{self.mode}', formulation='{self.formulation}', "
            f"model_name='{self.model_name}', pretrained={self.pretrained}, weights='{self.weights}', "
            f"model_config_file='{self.model_config_file}', train_batch_size={self.train_batch_size}, "
            f"steps_per_epoch={self.steps_per_epoch}, epochs={self.epochs}, train_loss='{self.train_loss}', "
            f"learning_rate={self.learning_rate}, optimizer='{self.optimizer}', val_batch_size={self.val_batch_size}, "
            f"val_steps_per_epoch={self.val_steps_per_epoch}, val_loss='{self.val_loss}', "
            f"dataset_name='{self.dataset_name}', image_size={self.image_size}, crop_size={self.crop_size}, "
            f"num_workers={self.num_workers}, masking_ratio={self.masking_ratio}, "
            f"log_root='{self.log_root}', exp_name='{self.exp_name}', result_columns={self.result_columns}, "
            f"save_weights={self.save_weights}, save_only_best_weights={self.save_only_best_weights}, "
            f"enable_tensorboard={self.enable_tensorboard}, log_figures={self.log_figures}, "
            f"log_interval={self.log_interval})"
        )


class ModelConfig:
    pass
