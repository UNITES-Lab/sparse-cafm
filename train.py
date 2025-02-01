from ast import parse
import os
import sys
from typing import Optional
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.logger import ExperimentLogger
from src.util.loss import ImageInpaintingL1Loss
from src.util.config import (
    TrainConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    MODELS,
    parse_config,
)

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")


def setup_logger(config: TrainConfig) -> ExperimentLogger:
    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config.log_root,
        exp_name=config.exp_name,
        log_interval=config.log_interval,
    )
    logger.add_result_columns(config.result_columns)
    return logger


def create_model(config: TrainConfig) -> nn.Module:
    model_fn = MODELS[config.model_name]["fn"]
    model_weights = MODELS[config.model_name]["weights"]
    if model_weights:
        model = model_fn(weights=model_weights)
    elif config.model_name == "hiera":
        model = model_fn
        model.freeze()
    else:
        model = model_fn()
    assert isinstance(model, nn.Module)
    return model.cuda(config.device).float()


def create_dataloader(config: TrainConfig, split: str) -> DataLoader:
    split_str = "training" if split == "train" else "validation"
    img_size = int(config.image_size)
    dataset = MOS2SEFDataset(
        split=split,
        side_length=int(config.crop_size),
        formulation=F.get_formulation_from_str(config.formulation),
        steps_per_epoch=(
            config.steps_per_epoch if split == "train" else config.val_steps_per_epoch
        ),
        device=config.device,
        original_image_size=(img_size, img_size),
        masking_ratio=int(config.masking_ratio),
    )
    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def train(config: TrainConfig, model_config: Optional[ModelConfig] = None) -> None:

    logger = setup_logger(config)
    model = create_model(config)
    train_dataloader = create_dataloader(config, "train")
    val_dataloader = create_dataloader(config, "val")

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config.train_loss]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config.val_loss]()
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config.optimizer](
        model.parameters(), lr=float(config.learning_rate)
    )

    best_loss = sys.maxsize
    num_epochs = config.epochs
    device = config.device

    # create model using model config obj
    # NOTE: only supported for SwinCAFM atm
    if config.model_config_file != None:
        assert isinstance(model, SwinCAFM), f"Only SwinCAFM supports init from config."
        model = SwinCAFM.init_from_config(model_config.to_dict())

    # load weights from checkpoint
    if config.weights != None:
        # load weights only:
        # model.load_state_dict(torch.load(config["model"]["weights"]), strict=False)
        # load enitre model object:
        model = torch.load(config.weights).float().cuda()

    model.cuda(device)
    model.float()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            # feature: X
            # X: torch.Tensor = batch["X"].cuda(device)

            # target: y
            y: torch.Tensor = batch["y"].cuda(device)

            # mask
            y_mask: torch.Tensor = batch["y_mask"].cuda(device)
            y_sparse = (y * y_mask).float()

            # zero gradients
            optimizer.zero_grad()

            # forward
            # P(y | y_sparse)
            outputs = model(y_sparse)

            final_pred = ImageInpaintingL1Loss.get_final_prediction(
                predicted_image=outputs, target_image=y, mask=y_mask
            )

            # # NOTE: standard loss (e.g., L1)
            # loss = train_loss(outputs, y)

            # NOTE: inpainting loss
            loss: torch.Tensor = train_loss(
                predicted_image=outputs, target_image=y, mask=y_mask
            )

            # HACK: manually scale up loss
            loss = loss

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
            # log a triplet (original, masked, predicted) every 100 steps
            if i % 100 == 0:
                triplet_name = f"train_epoch_{epoch}_step_{i}.png"
                final_pred = ImageInpaintingL1Loss.get_final_prediction(
                    predicted_image=outputs, target_image=y, mask=y_mask
                )
                logger.log_original_masked_predicted_sample_triplet(
                    y, y_sparse, final_pred, triplet_name
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
                # X: torch.Tensor = batch["X"].cuda(device)

                # target: y
                y: torch.Tensor = batch["y"].cuda(device)

                # mask
                y_mask: torch.Tensor = batch["y_mask"].cuda(device)
                y_sparse = (y * y_mask).float()

                # forward : p(y | y_sparse)
                outputs = model(y_sparse)

                # # NOTE: standard loss (e.g., L1)
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

                # log a triplet (original, masked, predicted) every 100 steps
                if i % 100 == 0:
                    triplet_name = f"val_epoch_{epoch}_step_{i}.png"
                    final_pred = ImageInpaintingL1Loss.get_final_prediction(
                        predicted_image=outputs, target_image=y, mask=y_mask
                    )
                    logger.log_original_masked_predicted_sample_triplet(
                        y, y_sparse, final_pred, triplet_name
                    )

            # optionally log best/epoch model weights
            avg_val_loss = val_running_loss / num_val_steps

            if bool(config.save_weights):
                if bool(config.save_only_best_weights):
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logger.save_weights(model, "best")
                    else:
                        logger.save_weights(model, f"latest_{epoch}")
                else:
                    logger.save_weights(model, f"epoch_{epoch}")


def main():
    config = TrainConfig(TRAIN_CONFIG_FP)
    model_config: Optional[ModelConfig] = None
    if config.model_config_file != None:
        model_config_abs_path = os.path.join(
            Path(TRAIN_CONFIG_FP).parent.__str__(), config.model_config_file
        )
        assert os.path.isfile(
            model_config_abs_path
        ), f"Bad path to model config: {model_config_abs_path}"
        model_config = ModelConfig(model_config_abs_path)
    train(config, model_config)


if __name__ == "__main__":
    main()
