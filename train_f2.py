import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.sapphire import SapphireDataset
from util.logger import ExperimentLogger
from util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS, parse_config

CONFIG_FP = (
    "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/config.yaml"
)
Z_MULT = 1

"""
Formulation 2/4.

Models
    1. conditional diffusion model:     X -> y_hat
    2. simple regression model:     y_hat -> z_hat
    
Training diffusion model will be a different procedure from eval.

"""


def train():
    config = parse_config(CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=CONFIG_FP,
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])


def eval():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
