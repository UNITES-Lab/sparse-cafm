import torch
import os
import cv2
import yaml
import tqdm
import numpy as np

from typing import Optional
from src.datasets.mos2_sef import (
    MOS2SEFDataset,
    Formulation,
)
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.torch_helpers import convert_to_img_like, grayscale_to_2d
from src.util.logger import ExperimentLogger
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from torch.utils.data import DataLoader
from torchmetrics.functional.image.ssim import ssim
from cldm.cldm import ControlLDM
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


TRAIN_CONFIG_FP = (
    "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/configs/train.yaml"
)
MODEL_PICKLE_FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/__weights__/sd_21_controlnet.pkl"
SD_CHECKPOINT = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/models/control_sd21_ini.ckpt"
FT_CHECKPOINT_FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/6. p(y | y_sparse)/6a. train-runs/2025-01-28_15-30-33_control_net_128x128/control_net_128x128_last.ckpt"


def save_results_to_fp(
    results: dict,
    split: str,
    index: int,
    logger: ExperimentLogger,
    dataset: MOS2SEFDataset,
):

    images: dict = results
    pred: torch.Tensor = images["samples_cfg_scale_9.00"].squeeze(0).detach().cpu()
    vae_og_recon: torch.Tensor = images["reconstruction"].squeeze(0).detach().cpu()
    control = images["control"].squeeze(0).detach().cpu()

    # images.keys(): ['reconstruction', 'control', 'conditioning', 'samples_cfg_scale_9.00']
    # "control": [0, 1]
    # --- original target image
    # "reconstruction": [-1, 1]
    # --- actual target image reconstructed from VAE
    # "conditioning": silly torch.ones() block
    # "samples_cfg_scale_9.00": [-1, 1]
    # --- model predicition

    # [C, H, W] -> [H, W, C]
    pred = pred.permute(1, 2, 0)
    vae_og_recon = vae_og_recon.permute(1, 2, 0)
    control: torch.Tensor = control.permute(1, 2, 0)

    # [H, W, C] -> [H, W]
    y = grayscale_to_2d(vae_og_recon)
    y_sparse = grayscale_to_2d(control)
    y_hat = grayscale_to_2d(pred)

    # [H, W] -> [B, H, W]
    y = y.unsqueeze(0)
    y_sparse = y_sparse.unsqueeze(0)
    y_hat = y_hat.unsqueeze(0)

    # 1. MAE
    mae = (y_hat - y).abs().mean()
    # 2. MSE
    mse = (y_hat - y).pow(2).mean()
    # 3. PSNR
    psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)

    # (B, H, W) -> (B, 1, H, W)
    final_pred_img_like = y_hat.clone()
    final_pred_img_like = final_pred_img_like.unsqueeze(1)
    # (B, 1, H, W) -> (B, 3, H, W)
    final_pred_img_like = final_pred_img_like.repeat(1, 3, 1, 1)

    # (B, H, W) -> (B, 1, H, W)
    y_img_like = y.clone()
    y_img_like = y_img_like.unsqueeze(1)
    # (B, 1, H, W) -> (B, 3, H, W)
    y_img_like = y_img_like.repeat(1, 3, 1, 1)

    # 4. SSIM
    # TODO: clamp range is incorrect
    ssim_val = ssim(
        final_pred_img_like.clamp(0, 1).float(),  # clamp just to be safe
        y_img_like.clamp(0, 1).float(),
        data_range=1.0,
    )

    # 5a. characterize(y)
    mean, std = dataset.current_maps_mean, dataset.current_maps_std
    data = (y - mean) / std
    y_char = celano_lab_characterization(data, dataset.img_size_um)

    # 5b. characterize(y_sparse)
    data = (y_hat - mean) / std
    y_sparse_char = celano_lab_characterization(data, dataset.img_size_um)

    # log metrics
    logger.log(
        **{
            "step": index,
            "train_l1": mae.item() if split == "train" else None,
            "val_l1": mae.item() if split == "val" else None,
            "train_mse": mse.item() if split == "train" else None,
            "val_mse": mse.item() if split == "val" else None,
            "train_psnr": psnr.item() if split == "train" else None,
            "val_psnr": psnr.item() if split == "val" else None,
            "train_ssim": ssim_val.item() if split == "train" else None,
            "val_ssim": ssim_val.item() if split == "val" else None,
            "val_celano_script_y": y_char if split == "val" else None,
            "val_celano_script_y_sparse": y_sparse_char if split == "val" else None,
        }
    )
    logger.log_original_masked_predicted_sample_triplet(
        y, y_sparse, y_hat, f"{index}.png"
    )


def parse_config(fp: str) -> dict:
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Sample y_hat predictions from Saphire MoS2 dataset.
    Save all resulting samples as a local file.
    """

    config = parse_config(TRAIN_CONFIG_FP)

    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )

    sd_locked = True
    only_mid_control = False

    model = create_model("./models/cldm_v21.yaml").cpu()
    model.load_state_dict(load_state_dict(FT_CHECKPOINT_FP, location="cpu"))

    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.cuda()
    model.eval()

    split_str = "validation"
    img_size = int(config["dataset"]["image_size"])
    val_dataset = MOS2SEFDataset(
        split="val",
        side_length=int(config["dataset"]["crop_size"]),
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        steps_per_epoch=config[split_str]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
    )
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)

    # perform inference
    for i, batch in tqdm.tqdm(enumerate(val_dataloader)):
        # skip existing samples
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.cuda()
        log: dict = model.log_images(batch, sample=True)
        save_results_to_fp(log, "val", i, logger, val_dataset)


if __name__ == "__main__":
    main()
