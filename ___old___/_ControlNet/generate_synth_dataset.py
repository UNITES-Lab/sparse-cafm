import os
import torch
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from pathlib import Path
from src.datasets.mos2_sef import (
    MOS2SEFDataset,
)
from src.datasets.mos2_sr import UnifiedMOS2SRDataset
from src.util.torch_helpers import grayscale_to_2d
from src.util.logger import ExperimentLogger
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from torch.utils.data import DataLoader
from torchmetrics.functional.image.ssim import ssim
from cldm.model import create_model, load_state_dict
from src.util.metrics import OLDER

NUM_TRAIN_SAMPLES = 10000
NUM_VAL_SAMPLES = 500

OUT_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/synth-datasets/topology"
FT_CHECKPOINT_FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__controlnet_runs__/2025-02-26_13-08-20_controlnet-unconditional/controlnet-unconditional_last.ckpt"


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

    # # [H, W, C] -> [H, W]
    # y = grayscale_to_2d(vae_og_recon)
    # y_sparse = grayscale_to_2d(control)
    y_hat = grayscale_to_2d(pred)

    # # NOTE: this step is only needed for ControlNet outputs
    # # [-1, 1] -> [0, 1]
    # y = (y + 1) / 2
    # y_sparse = (y_sparse + 1) / 2
    y_hat = (y_hat + 1) / 2

    # [H, W] -> [B, H, W]
    y_hat = y_hat.unsqueeze(0)
    
    # y = y.unsqueeze(0)
    # y_sparse = y_sparse.unsqueeze(0)

    # # (B, H, W) -> (B, 1, H, W)
    # final_pred_img_like = y_hat.clone()
    # final_pred_img_like = final_pred_img_like.unsqueeze(1)
    # # (B, 1, H, W) -> (B, 3, H, W)
    # final_pred_img_like = final_pred_img_like.repeat(1, 3, 1, 1)

    # # (B, H, W) -> (B, 1, H, W)
    # y_img_like = y.clone()
    # y_img_like = y_img_like.unsqueeze(1)
    # # (B, 1, H, W) -> (B, 3, H, W)
    # y_img_like = y_img_like.repeat(1, 3, 1, 1)

    # mean, std = dataset.current_maps_mean, dataset.current_maps_std

    # # 5b. characterize(y_hat)
    # # z: [0, 1] -> {std_normal}
    # z = norm.ppf(y_hat)

    # # x' = mu + (sigma * z)
    # x_prime = mean + (std * z)
    # x_prime = x_prime.squeeze()

    subdir = "train"
    if index > NUM_TRAIN_SAMPLES: subdir = "val"
    cm_out_fp  = os.path.join(OUT_DIR, subdir, "topo-maps", f"{index:06d}.npy") 
    img_out_fp = os.path.join(OUT_DIR, subdir, "images", f"{index:06d}.png")

    np.save(cm_out_fp, pred)

    # -> [0, 1]
    y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())
    plt.imsave(img_out_fp, y_hat.numpy().squeeze(), cmap='viridis')


def parse_config(fp: str) -> dict:
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def main():
    """
    Sample y_hat predictions from Saphire MoS2 dataset.
    Save all resulting samples as a local file.
    """

    config = None
    logger = None

    sd_locked = True
    only_mid_control = False

    model = create_model("./models/cldm_v21.yaml").cpu()
    model.load_state_dict(load_state_dict(FT_CHECKPOINT_FP, location="cpu"))
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.cuda()
    model.eval()

    # perform inference
    val_dataset = UnifiedMOS2SRDataset(
        split="train",
        upsample_factor=8,
        steps_per_epoch=NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES,
        original_image_size=(384, 384),
    )
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)

    # perform inference
    for i, batch in tqdm.tqdm(enumerate(val_dataloader)):

        index = i
        subdir = "train"
        if index > NUM_TRAIN_SAMPLES: subdir = "val"

        cm_out_fp  = os.path.join(OUT_DIR, subdir, "topo-maps", f"{index:06d}.npy") 
        img_out_fp = os.path.join(OUT_DIR, subdir, "images", f"{index:06d}.png")
        
        if os.path.isfile(cm_out_fp): continue
        if os.path.isfile(img_out_fp): continue

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.cuda()

        log: dict = model.log_images(batch, sample=True)
        
        save_results_to_fp(log, subdir, i, logger, val_dataset)

if __name__ == "__main__":
    main()
