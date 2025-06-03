"""
Training script for `sparse-bto` models.
"""

import torch
import yaml
import wandb
import importlib
import argparse
import warnings

from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from src.util.logger import Logger
from src.util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import BTOSRDataset, BTO_MANY_RES

warnings.simplefilter("always")


def _init_module_from_target(mod_config: dict) -> Any:
    """
    Init a module from a module config dict,
       expect keywords `target` and `args`.
    """
    mod_path, cls_name = mod_config["target"].rsplit(".", 1)
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    args = mod_config.get("args", {})
    return cls(**args)


def _get_dataloader(config: dict, split: str): pass


def train(config: dict):

    logger = _init_module_from_target(config["logger"])

    # some cleaver run initiatization
    if bool(config['wandb']['use_wandb']) == True:
        _init_module_from_target(config['wandb']['login'])
        _init_module_from_target(config['wandb']['init'])

    # init datasets/dataloaders
    train_dataset = _init_module_from_target(config['train_args']['dataset'])
    val_dataset   = _init_module_from_target(config['val_args']['dataset'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = int(config['train_args']['batch_size']),
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size = int(config['val_args']['batch_size']),
        shuffle=False,
    )
    
    # init loss
    train_loss = _init_module_from_target(config['train_args']['loss'])
    val_loss = _init_module_from_target(config['val_args']['loss'])

    model = config['model']
    breakpoint()

    # init optim
    optimizer  = _init_module_from_target(config['train_args']['optimizer'])

def main(config: dict) -> None:
    train(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Exeriment run .yaml config.", default="")
    args = parser.parse_args()
    
    assert str(args.config).endswith(".yaml"), f"Error: run config must be a `.yaml` file."
    assert Path(str(args.config)).is_file(), f"Error: config is not a valid file."
    config_path = Path(str(args.config))

    try:
        with open(str(args.config), "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: exception opening config: {e}")
        raise Exception()

    main(config)
