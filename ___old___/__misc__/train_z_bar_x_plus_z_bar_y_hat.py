import os
import yaml
import sys
import torch
import torch.nn as nn

from models.simple_z_predictor import SimpleZRegressionVisionTransformer
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.sapphire import SapphireDatasetFixedGridSampling, Formulation
from util.logger import ExperimentLogger
from util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS, parse_config
from models.regression_head import RegressionHead


TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")
Z_MULT = 1


CONFIG_FP = (
    "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/config.yaml"
)
Z_MULT = 1

"""
Formulation 4/4.

Models
    1. conditional diffusion model:                  X -> y_hat
    2. simple regression model:                 X + M1 -> z_hat_1
    3. simple regression model:                 X + M2 -> z_hat_2
    4. ensemble:               (z_hat_1 + z_hat_2) / 2 -> z_hat
    
Training diffusion model will be a different procedure from eval.
"""


@torch.no_grad()
def eval():
    """
    Evaluate a ViT regression model (data) -> (z_pred) for 1000 steps on held out data.
    - Report MAE and avg MAE.
    """

    config = parse_config(EVAL_CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=EVAL_CONFIG_FP,
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )

    # add metrics to log
    logger.add_result_columns(config["logging"]["result_columns"])
    img_size = int(config["dataset"]["image_size"])
    model: torch.nn.Module = torch.load(config["model"]["weights_path"])
    device = config["global"]["device"]
    model.cuda(device)
    model.eval()

    val_dataset = SapphireDatasetFixedGridSampling(
        "val",
        Formulation.P_Z_BAR_X,
        steps_per_epoch=config["validation"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
    )
    val_loss = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    loss_total = 0.0
    loss_mvg_avg = 0.0

    for i, batch in enumerate(tqdm(val_dataloader, desc=f"Evaluating Model")):

        # move inputs -> device
        X = batch["X"].to(device)
        z = batch["z"].to(device)

        # forward pass
        outputs = model(X)

        # ensure the target has the correct shape
        if z.dim() == 1:
            z = z.unsqueeze(1)

        # compute loss
        loss = val_loss(outputs, z)

        # accumulate validation loss
        loss_total += loss.item() * X.size(0)
        loss_mvg_avg = loss_total / (i + 1)
        logger.log(
            **{
                "L1 Loss": loss_total,
                "Avg. L1 Loss": loss_mvg_avg,
            }
        )


def train():

    config = parse_config(TRAIN_CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])

    # load model
    m1: torch.nn.Module = SimpleZRegressionVisionTransformer()
    m2: torch.nn.Module = SimpleZRegressionVisionTransformer()
    
    # create train/val datasets and dataloaders
    img_size = int(config["dataset"]["image_size"])
    train_dataset = SapphireDatasetFixedGridSampling(
        split="train",
        formulation=Formulation.P_Z_BAR_Y,
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
    val_dataset = SapphireDatasetFixedGridSampling(
        split="val",
        formulation=Formulation.P_Z_BAR_Y,
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
    tl1 = LOSS_FUNCTIONS[config["training"]["loss"]]()
    tl2 = LOSS_FUNCTIONS[config["training"]["loss"]]()
    
    vl1 = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    vl2 = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    
    o1: torch.optim.Optimizer = OPTIMIZERS[config["training"]["optimizer"]](
        m1.parameters(), lr=float(config["training"]["lr"])
    )
    o2: torch.optim.Optimizer = OPTIMIZERS[config["training"]["optimizer"]](
        m2.parameters(), lr=float(config["training"]["lr"])
    )

    best_loss = sys.maxsize
    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]
    
    m1.cuda(device)
    m2.cuda(device)

    for epoch in range(num_epochs):

        m1.train()
        m2.train()
        
        running_loss = 0.0

        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):
            # feature: topo-map X
            X: torch.Tensor = batch["X"]
            X = X.cuda(device)
            # feature: predicted current map y_hat
            y_hat: torch.Tensor = batch["y_hat"]
            y_hat = y_hat.cuda(device)
            # target: scalar-value z
            z: torch.Tensor = batch["z"]
            z = z.cuda(device)
            # zero gradients
            o1.zero_grad()
            o2.zero_grad()
            # forwards
            out_1 = m1(X)
            out_2 = m2(y_hat)
            # TODO: do we need this?
            if z.dim() == 1:
                z = z.unsqueeze(1)
            loss1 = tl1(out_1, z)
            loss2 = tl2(out_2, z)
            # backprop
            loss1.backward()
            loss2.backward()
            # grad descent
            o1.step()
            o2.step()
            # ensemble loss
            increment = ((loss1.item() * X.size(0)) + (loss2.item() * y_hat.size(0))) / 2
            running_loss += increment
            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + i,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": (loss1.item() + loss2.item()) / 2,
                    "val_loss": None,
                    "z": z.mean().item(),
                }
            )

        # validation
        m1.eval()
        m2.eval()
        val_running_loss = 0.0
        num_val_steps = 0

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):
                # feature: topo-map X
                X: torch.Tensor = batch["X"]
                X = X.cuda(device)
                # feature: predicted current map y_hat
                y_hat: torch.Tensor = batch["y_hat"]
                y_hat = y_hat.cuda(device)
                # target: scalar-value z
                z: torch.Tensor = batch["z"]
                z = z.cuda(device)
                # forwards
                out_1 = m1(X)
                out_2 = m2(y_hat)
                # TODO: do we need this?
                if z.dim() == 1:
                    z = z.unsqueeze(1)
                loss1 = vl1(out_1, z)
                loss2 = vl2(out_2, z)
                # calc ensemble loss
                increment = ((loss1.item() * X.size(0)) + (loss2.item() * y_hat.size(0))) / 2
                val_running_loss += increment
                logger.log(
                    **{
                        "global_train_step": None,
                        "global_val_step": len(val_dataloader) * (epoch) + i,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": (loss1.item() + loss2.item()) / 2,
                        "z": z.mean().item(),
                    }
                )
                num_val_steps += 1

            # optionally log best/epoch model weights
            avg_val_loss = val_running_loss / num_val_steps
            if bool(config["logging"]["save_weights"]):
                if bool(config["logging"]["save_only_best_weights"]):
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logger.save_weights(m1, "best_m1")
                        logger.save_weights(m2, "best_m2")
                else:
                    # logger.save_weights(model, f"epoch_{epoch}")
                    pass


def main():
    train()


if __name__ == "__main__":
    main()
