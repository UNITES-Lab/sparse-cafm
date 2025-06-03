"""
Training script for `sparse-bto` models.
"""

import torch
import yaml
import wandb
import importlib
import argparse
import warnings

from tqdm import tqdm
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from src.util.logger import Logger
from src.util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import BTOSRDataset, BTO_MANY_RES

warnings.simplefilter("ignore")


def _init_module_from_target(mod_config: dict, *, additional_args: dict={}) -> Any:
    """
    Init a module from a module config dict,
       expect keywords `target` and `args`.
    """
    mod_path, cls_name = mod_config["target"].rsplit(".", 1)
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    args: dict = mod_config.get("args", {})
    args.update(additional_args)
    return cls(**args)


def train(config: dict) -> None:

    logger = _init_module_from_target(config["logger"])

    # some cleaver run initiatization
    if bool(config['wandb']['use_wandb']) == True:
        _init_module_from_target(config['wandb']['login'])
        _init_module_from_target(config['wandb']['init'])

    # init datasets/dataloaders
    train_dataset    = _init_module_from_target(config['train_args']['dataset'])
    val_dataset      = _init_module_from_target(config['val_args']['dataset'])
    train_dataloader = DataLoader(train_dataset, batch_size = int(config['train_args']['batch_size']), shuffle=False)
    val_dataloader   = DataLoader(val_dataset, batch_size = int(config['val_args']['batch_size']), shuffle=False)
    
    # init loss
    train_loss = _init_module_from_target(config['train_args']['loss'])
    val_loss   = _init_module_from_target(config['val_args']['loss'])

    model: torch.nn.Module = _init_module_from_target(config['model'])
    model.float().cuda()

    # init optim
    optimizer  = _init_module_from_target(config['train_args']['optimizer'], additional_args={"params": model.parameters()})

    # main training loop
    for epoch in range(int(config['train_args']['num_epochs'])):
        
        # train
        model.train()

        for step, item in tqdm(enumerate(train_dataloader), desc=f"🚀 Training Epoch: {epoch + 1}/{int(config['train_args']['num_epochs'])}", total=int(config['train_args']['dataset']['args']['steps_per_epoch'])):

            X        = item["X"].float().cuda()
            X_sparse = item["X_sparse"].float().cuda()

        # validate
        model.eval()

        with torch.no_grad():

            for step, item in tqdm(enumerate(val_dataloader), desc=f"🚀 Validation Epoch: {epoch + 1}/{int(config['train_args']['num_epochs'])}", total=int(config['val_args']['dataset']['args']['steps_per_epoch'])):

                X        = item["X"].float().cuda()
                X_sparse = item["X_sparse"].float().cuda()

        quit()


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
