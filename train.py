import os
import yaml
import sys
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.datasets.sapphire import SapphireDataset, Formulation as F
from src.util.logger import ExperimentLogger
from src.util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS, parse_config
from src.models.regression_head import RegressionHead
from src.models.unet.unet import ThickUNet
from src.util.loss import ImageInpaintingL1Loss

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")
Z_MULT = 1


@torch.no_grad()
def eval(config: dict) -> None:

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

    if model_weights != None:
        model: torch.nn.Module = model_fn(weights=model_weights)
    elif config["model"]["name"] == "hiera":
        # HACK
        model: torch.nn.Module = model_fn
        model.freeze()
    else:
        model: torch.nn.Module = model_fn()

    # create train/val datasets and dataloaders
    img_size = int(config["dataset"]["image_size"])
    val_dataset = SapphireDataset(
        split="val",
        formulation=F.P_Y_BAR_X,
        steps_per_epoch=config["validation"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    print(val_dataset.current_maps_std, val_dataset.current_maps_mean)
    return 

    # define loss function and optimizer
    val_loss = LOSS_FUNCTIONS[config["validation"]["loss"]]()

    best_loss = sys.maxsize
    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]

    model.cuda(device)
    model.float()

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        model = torch.load(config["model"]["weights"])

    # validation
    model.eval()
    val_running_loss = 0.0
    num_val_steps = 0
    epoch = 0

    for i, batch in enumerate(
        tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
    ):
        # target: y
        y: torch.Tensor = batch["y"].cuda(device)
        # mask
        y_mask: torch.Tensor = batch["y_mask"].cuda(device)
        y_sparse = (y * y_mask).float()
        # forward : p(y | y_sparse)
        outputs: torch.Tensor = model(y_sparse)
        # loss = val_loss(outputs, y)
        # NOTE: inpainting loss
        loss = val_loss(predicted_image=outputs, target_image=y, mask=y_mask)
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
        logger.save_sample(f"{i}.npy", y.squeeze(0), "ground-truth-current-maps")
        logger.save_sample(f"{i}.npy", outputs.squeeze(0), "predicted-current-maps")
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


def train(config: dict) -> None:

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
    if model_weights != None:
        model: torch.nn.Module = model_fn(weights=model_weights)
    elif config["model"]["name"] == "hiera":
        # HACK
        model: torch.nn.Module = model_fn
        model.freeze()
    else:
        model: torch.nn.Module = model_fn()

    # create train/val datasets and dataloaders
    img_size = int(config["dataset"]["image_size"])
    train_dataset = SapphireDataset(
        split="train",
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        steps_per_epoch=config["training"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    val_dataset = SapphireDataset(
        split="val",
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        steps_per_epoch=config["validation"]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
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

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        model.load_state_dict(torch.load(config["model"]["weights"]))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):
            # feature: X
            X: torch.Tensor = batch["X"].cuda(device)
            # target: y
            y: torch.Tensor = batch["y"].cuda(device)
            # mask
            y_mask: torch.Tensor = batch["y_mask"].cuda(device)
            y_sparse = (y * y_mask).float()
            # forward
            # zero gradients
            optimizer.zero_grad()
            # P(y | y_sparse)
            outputs = model(y_sparse)

            # loss = train_loss(outputs, y)
            # NOTE: inpainting loss
            loss = train_loss(predicted_image=outputs, target_image=y, mask=y_mask)

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
                X: torch.Tensor = batch["X"].cuda(device)
                # target: y
                y: torch.Tensor = batch["y"].cuda(device)
                # mask
                y_mask: torch.Tensor = batch["y_mask"].cuda(device)
                y_sparse = (y * y_mask).float()
                # forward : p(y | y_sparse)
                outputs = model(y_sparse)

                # loss = val_loss(outputs, y)
                # NOTE: inpainting loss
                loss = val_loss(predicted_image=outputs, target_image=y, mask=y_mask)

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
    config = parse_config(TRAIN_CONFIG_FP)
    if config["global"]["mode"] == "eval":
        eval(config)
    elif config["global"]["mode"] == "train":
        train(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
