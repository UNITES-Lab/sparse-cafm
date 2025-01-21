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
from torchmetrics.functional.image.ssim import ssim

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")
Z_MULT = 1


def setup_logger(config: dict) -> ExperimentLogger:
    logger = ExperimentLogger(
        config_fp=EVAL_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])
    return logger


def create_model(config: dict) -> nn.Module:
    model_fn = MODELS[config["model"]["name"]]["fn"]
    model_weights = MODELS[config["model"]["name"]]["weights"]
    if model_weights:
        model = model_fn(weights=model_weights)
    elif config["model"]["name"] == "hiera":
        model = model_fn
        model.freeze()
    else:
        model = model_fn()
    assert isinstance(model, nn.Module)
    return model.cuda(config["global"]["device"]).float()


def create_dataloader(config: dict, split: str) -> DataLoader:
    split_str = "training" if split == "train" else "validation"
    img_size = int(config["dataset"]["image_size"])
    dataset = SapphireDataset(
        split=split,
        side_length=int(config["dataset"]["crop_size"]),
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        steps_per_epoch=config[split_str]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
    )
    return DataLoader(
        dataset,
        batch_size=config[split_str]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )


@torch.no_grad()
def eval(config: dict) -> None:

    logger = setup_logger(config)
    model = create_model(config)
    val_dataloader = create_dataloader(config, "val")

    device = config["global"]["device"]

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        model = torch.load(config["model"]["weights"])

    assert isinstance(model, torch.nn.Module)

    # validation loop
    model.eval()

    for step, batch in enumerate(
        tqdm(val_dataloader, desc=f"Evaluating... ")
    ):
        # target: y
        y: torch.Tensor = batch["y"].cuda(device)
        # mask
        y_mask: torch.Tensor = batch["y_mask"].cuda(device)
        y_sparse = (y * y_mask).float()
        # forward : p(y|y_sparse)
        y_hat: torch.Tensor = model(y_sparse)
        # 1. MAE
        mae = (y_hat - y).abs().mean()
        # 2. MSE
        mse = (y_hat - y).pow(2).mean()
        # 3. PSNR
        #   psnr = 10 * log10( peak_val^2 / mse )
        #        = 20 * log10(peak_val) - 10 * log10(mse)
        psnr = 20 * torch.log10(torch.tensor(2.)) - 10 * torch.log10(mse)
        # 4. SSIM
        # Example using torchmetrics:
        ssim_val = ssim(
            y_hat.clamp(-1, 1).float(),  # clamp just to be safe
            y.clamp(-1, 1).float(),
            data_range=2.
        )
        logger.log(
            **{
                "step": step,
                "mae": mae.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
                "ssim": ssim_val.item(),
            }
        )


def main():
    config = parse_config(EVAL_CONFIG_FP)
    if config["global"]["mode"] == "eval":
        eval(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
