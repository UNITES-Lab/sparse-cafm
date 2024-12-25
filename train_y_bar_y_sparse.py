import os
import yaml
import sys
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.sapphire import SapphireDataset, Formulation as F
from util.logger import ExperimentLogger
from util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS, parse_config
from models.regression_head import RegressionHead
from models.unet.unet import ThickUNet

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")
Z_MULT = 1

"""
Formulation 1/4.

Models
    1. simple regression model:     X -> z_hat

Training diffusion model will be a different procedure from eval.
"""

def train():

    config = parse_config(TRAIN_CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])

    # dynamically load model
    model_fn = MODELS[config["model"]["name"]]["fn"]
    model_weights = MODELS[config["model"]["name"]]["weights"]
    model: torch.nn.Module = model_fn(weights=model_weights)

    # create train/val datasets and dataloaders
    img_size = int(config["dataset"]["image_size"])
    train_dataset = SapphireDataset(
        split="train",
        formulation=F.P_Y_BAR_X,
        steps_per_epoch=config["training"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    val_dataset = SapphireDataset(
        split="val",
        formulation=F.P_Y_BAR_X,
        steps_per_epoch=config["validation"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    # define loss function and optimizer
    train_loss = LOSS_FUNCTIONS[config["training"]["loss"]]()
    val_loss = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["training"]["optimizer"]](
        model.parameters(), lr=float(config["training"]["lr"])
    )

    best_loss = sys.maxsize
    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]
    model.cuda(device)
    model.float()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):
            # feature: X
            X: torch.Tensor = batch["X"]
            X = X.cuda()
            # feature: y_sparse
            y_sparse: torch.Tensor = batch["y_sparse"]
            y_sparse = y_sparse.cuda(device)
            # target: y
            y: torch.Tensor = batch["y"]
            y = y.cuda(device)
            # zero gradients
            optimizer.zero_grad()
            
            # forward
            # # P(y | y_sparse)
            # outputs = model(y_sparse)
            # # P(y | X)
            # outputs = model(X)
            # # P(y | X, y_sparse)
            # assert isinstance(model, ThickUNet)
            # outputs = model.wide_forward(X, y_sparse)
            # P(y | X, y_sparse*c)
            C = 0.00001
            assert isinstance(model, ThickUNet)
            outputs = model.wide_forward(X, y_sparse*C)
            
            loss = train_loss(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y_sparse.size(0)
            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + i,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": None,
                }
            )

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps = 0

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):
                # feature: X
                X: torch.Tensor = batch["X"]
                X = X.cuda()
                # feature: y_sparse
                y_sparse: torch.Tensor = batch["y_sparse"]
                y_sparse = y_sparse.cuda(device)
                # target: y
                y: torch.Tensor = batch["y"]
                y = y.cuda(device)
                
                # forward
                # # P(y | y_sparse)
                # outputs = model(y_sparse)
                # # P(y | X)
                # outputs = model(X)
                # # P(y | X, y_sparse)
                # assert isinstance(model, ThickUNet)
                # outputs = model.wide_forward(X, y_sparse)
                # P(y | X, y_sparse*c)
                C = 0.00001
                assert isinstance(model, ThickUNet)
                outputs = model.wide_forward(X, y_sparse*C)
                
                loss = val_loss(outputs, y)
                val_running_loss += loss.item() * y_sparse.size(0)
                logger.log(
                    **{
                        "global_train_step": None,
                        "global_val_step": len(val_dataloader) * (epoch) + i,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": loss.item(),
                    }
                )
                num_val_steps += 1

            # optionally log best/epoch model weights
            avg_val_loss = val_running_loss / num_val_steps
            if bool(config["logging"]["save_weights"]):
                if bool(config["logging"]["save_only_best_weights"]):
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logger.save_weights(model, "best")
                else:
                    logger.save_weights(model, f"epoch_{epoch}")


def main():
    train()


if __name__ == "__main__":
    main()
