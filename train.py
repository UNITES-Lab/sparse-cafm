import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
from torchmetrics import LPIPS, PSNR, SSIM

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import MOS2SRDataset, MOS2_SILICON_DIR, MOS2_SAPPHIRE_DIR, MOS2_SEF_SRC_DIR
from src.util.logger import ExperimentLogger
from src.util.config import (
    TrainConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    MODELS,
)

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")


def setup_logger(
    train_config: TrainConfig, model_config: Optional[ModelConfig]
) -> ExperimentLogger:
    logger = ExperimentLogger(
        train_config_dict=train_config.to_dict(),
        model_config_dict=model_config.to_dict() if model_config != None else None,
        root=train_config.log_root,
        exp_name=train_config.exp_name,
        log_interval=train_config.log_interval,
    )
    logger.add_result_columns(train_config.result_columns)
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
    return model


def create_dataloader(args, config: TrainConfig, split: str) -> DataLoader:
    dataset = MOS2SRDataset(
        src_dir=MOS2_SEF_SRC_DIR,
        split=split,
        steps_per_epoch=(
            int(config.steps_per_epoch * config.train_batch_size)
            if split == "train"
            else config.val_steps_per_epoch
        ),
        upsample_factor=int(args.upsample_factor)
    )
    return DataLoader(
        dataset,
        batch_size=(
            config.train_batch_size if split == "train" else config.val_batch_size
        ),
        shuffle=False,
        num_workers=config.num_workers,
    )


def train(args, config: TrainConfig, model_config: Optional[ModelConfig] = None,) -> None:

    breakpoint()
    
    logger = setup_logger(config, model_config)
    model = create_model(config)

    train_dataloader = create_dataloader(args, config, "train")
    val_dataloader = create_dataloader(args, config, "val")

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config.train_loss]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config.val_loss]()

    # use to save model checkpoints
    best_val_loss = sys.maxsize

    num_epochs = config.epochs
    device = config.device

    # create model using model config obj
    # NOTE: only supported for SwinCAFM atm
    if config.model_config_file != None:
        assert isinstance(model, SwinCAFM), f"Only SwinCAFM supports init from config."
        model = SwinCAFM.init_from_config(model_config.to_dict())

    optimizer: torch.optim.Optimizer = OPTIMIZERS[config.optimizer](
        model.parameters(), lr=float(config.learning_rate), weight_decay=1e-3,
    )

    model.cuda(device)
    model.float()

    # ---------- training loop ----------
    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            # current-map: y; [128, 128]
            y: torch.Tensor = batch["y"].cuda(device)

            # current-map: y_sparse; [64, 64]
            y_sparse: torch.Tensor = batch["y_sparse"].cuda(device)

            # zero gradients
            optimizer.zero_grad()

            # ---- forward: p(y | y_sparse) ----
            y_hat: torch.Tensor = model(y_sparse)

            # --- L1 ----
            loss: torch.Tensor = train_loss(y_hat, y)

            loss.backward()
            optimizer.step()

            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + step,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": None,
                }
            )

            # log figures every 100 steps
            if step % 100 != 0:
                continue
            triplet_name = f"train_epoch_{epoch}_step_{step}.png"
            logger.log_colorized_tensors(
                (y, "Target (y)"),
                (y_sparse, "Model Input (y_sparse)"),
                (y_hat, "Model Prediction"),
                file_name=triplet_name,
            )

        # validation
        model.eval()
        val_running_loss = 0.0
        num_val_steps = 1
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):
                # current-map: y; [128, 128]
                y: torch.Tensor = batch["y"].cuda(device)

                # current-map: y_sparse; [64, 64]
                y_sparse: torch.Tensor = batch["y_sparse"].cuda(device)

                # ---- forward: p(y | y_sparse) ----
                y_hat: torch.Tensor = model(y_sparse)

                # --- L1 ----
                loss = val_loss(y_hat, y)

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

                # log figures every 100 steps
                if i % 100 != 0:
                    continue
                triplet_name = f"val_epoch_{epoch}_step_{i}.png"
                logger.log_colorized_tensors(
                    (y, "Target (y)"),
                    (y_sparse, "Model Input (y_sparse)"),
                    (y_hat, "Model Prediction(y_hat)"),
                    file_name=triplet_name,
                )

            # optional: log best/recent model weights
            avg_val_loss = val_running_loss / num_val_steps

            if not bool(config.save_weights):
                continue
            if bool(config.save_only_best_weights):
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logger.save_weights(model, "best")
                else:
                    # NOTE: we overwrite previous "latest" weights
                    logger.save_weights(model, f"latest")
            else:
                logger.save_weights(model, f"epoch_{epoch}")


def main(args: argparse.Namespace) -> None:

    # load training config
    config = TrainConfig(TRAIN_CONFIG_FP)
    model_config: Optional[ModelConfig] = None

    # optional: parse model config
    if config.model_config_file != None:
        model_config_abs_path = os.path.join(
            Path(TRAIN_CONFIG_FP).parent.__str__(), config.model_config_file
        )
        assert os.path.isfile(
            model_config_abs_path
        ), f"Bad path to model config: {model_config_abs_path}"
        model_config = ModelConfig(model_config_abs_path)

    # -------------------- training config args --------------------
    config.exp_name = args.exp_name
    config.log_root = args.root
    # config.learning_rate = str(args.learning_rate)
    # config.train_batch_size = int(args.batch_size)
    # -------------------- model config args --------------------
    if model_config != None:
        # transformer block depths; e.g., [6, 6, 6, 6, 6, 6]
        model_config.depths = [args.depths] * args.num_blocks
        # num heads per block; e.g., [6, 6, 6, 6, 6, 6]
        model_config.num_heads = [args.num_heads] * args.num_blocks
        # size of sifted-attention window
        model_config.window_size = args.window_size
        model_config.drop_path_rate = args.drop_path_rate
        model_config.norm_layer = args.norm_layer

    # train
    train(args, config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------- training config args --------------------
    parser.add_argument("-e","--exp_name",type=str,help="Experiment directory name.",default="my-experiment",)
    parser.add_argument("-r","--root",type=str,help="Root directory to save experiment in.",default="__exps__/",)
    # -------------------- model config args --------------------
    parser.add_argument("-dps", "--depths", type=int, help="Depths of RSTB blocks", default=6)
    parser.add_argument("-nbs", "--num_blocks", type=int, help="Number of RSTB blocks", default=6)
    parser.add_argument("-nhs","--num_heads",type=int,help="Number of heads per RSTB block",default=6,)
    parser.add_argument("-wsz","--window_size",type=int,help="Size of shifted attention window",default=8,)
    parser.add_argument("-dpr", "--drop_path_rate", type=float, help="", default=0.1)
    parser.add_argument("-nlr", "--norm_layer", type=str, help="", default="torch.nn.LayerNorm")
    # -------------------- ablation args --------------------
    parser.add_argument("-lr", "--learning_rate", type=float, help="", default=1e-5)
    parser.add_argument("-bs", "--batch_size", type=int, help="", default=1)
    parser.add_argument("-sr", "--upsample_factor", type=int, help="", default=2)
    args = parser.parse_args()
    main(args)
