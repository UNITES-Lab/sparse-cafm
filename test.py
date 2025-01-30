import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional.image.ssim import ssim
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from src.util.logger import ExperimentLogger
from src.util.config import MODELS, parse_config
from src.util.loss import ImageInpaintingL1Loss
from src.util.metrics import OLDER, PSNR, MSE, MAE

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


def create_dataset(config: dict, split: str) -> MOS2SEFDataset:
    split_str = "training" if split == "train" else "validation"
    img_size = int(config["dataset"]["image_size"])
    dataset = MOS2SEFDataset(
        split=split,
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        side_length=int(config["dataset"]["crop_size"]),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
        steps_per_epoch=config[split_str]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
    )
    return dataset


@torch.no_grad()
def eval(config: dict) -> None:

    logger = setup_logger(config)
    model = create_model(config)
    val_dataset = create_dataset(config, "val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    dataset: MOS2SEFDataset = val_dataloader.dataset
    device = config["global"]["device"]

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        model = torch.load(config["model"]["weights"])
    assert isinstance(model, torch.nn.Module)

    # validation loop
    model.eval()

    for step, batch in enumerate(tqdm(val_dataloader, desc=f"Evaluating...:")):

        # target: y
        y: torch.Tensor = batch["y"].cuda(device)

        # mask
        y_mask: torch.Tensor = batch["y_mask"].cuda(device)
        y_sparse = (y * y_mask).float()

        # forward : p(y|y_sparse)
        # y_hat: torch.Tensor = model(y_sparse)

        # forward : p(y|y_sparse)
        # y_hat: torch.Tensor = model(y_sparse, y_mask)

        # NOTE: GPSTRUCT
        # forward : p(y|y_sparse)
        y_hat: torch.Tensor = model(y_sparse, y, y_mask)

        # log final predicted image
        triplet_name = f"eval_step_{step}.png"
        final_pred = ImageInpaintingL1Loss.get_final_prediction(
            predicted_image=y_hat, target_image=y, mask=y_mask
        )

        logger.log_original_masked_predicted_sample_triplet(
            y, y_sparse, final_pred, triplet_name
        )
        
        # 1. MAE
        mae = MAE(final_pred, y)
        # 2. MSE
        mse = MSE(final_pred, y)
        # 3. PSNR; assume data in range [0, 1]
        psnr = PSNR(final_pred, y, dataset.normalized_data_range)

        # (B, H, W) -> (B, 1, H, W)
        final_pred_img_like = final_pred.clone()
        final_pred_img_like = final_pred_img_like.unsqueeze(1)
        # (B, 1, H, W) -> (B, 3, H, W)
        final_pred_img_like = final_pred_img_like.repeat(1, 3, 1, 1)

        # (B, H, W) -> (B, 1, H, W)
        y_img_like = y.clone()
        y_img_like = y_img_like.unsqueeze(1)
        # (B, 1, H, W) -> (B, 3, H, W)
        y_img_like = y_img_like.repeat(1, 3, 1, 1)

        # 4. SSIM
        ssim_val = ssim(
            final_pred_img_like.clamp(0, 1).float(),
            y_img_like.clamp(0, 1).float(),
            data_range=1.0,
        )

        mean, std = val_dataset.current_maps_mean, val_dataset.current_maps_std

        # 5a. characterize(y)
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (y * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)
        y_char = celano_lab_characterization(data, val_dataset.img_size_um)

        # 5b. characterize(y_sparse)
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (final_pred * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)
        y_sparse_char = celano_lab_characterization(data, val_dataset.img_size_um)

        logger.log(
            **{
                "step": step,
                "mae": mae.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
                "ssim": ssim_val.item(),
                "older": OLDER(y_char, y_sparse_char),
                "celano_script_y": y_char,
                "celano_script_y_sparse": y_sparse_char,
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
