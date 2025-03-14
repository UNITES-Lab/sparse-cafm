import math
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from typing import List, Dict, Optional
from cldm.metrics import calc_psnr
from cv2 import log
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from torchmetrics.functional.image.ssim import ssim
from src.util.logger import ExperimentLogger
from src.util.torch_helpers import convert_to_img_like, grayscale_to_2d
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.celano_lab_scripts import process_image as celano_lab_characterization


class ScuffedLogger:
    """
    Singleton logger.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ScuffedLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_out_path: Optional[str] = None) -> None:
        if hasattr(self, "initialized") and self.initialized:
            return

        if log_out_path is not None:
            self.csv_out_path = log_out_path
        else:
            self.csv_out_path = "log.csv"

        self.HEADERS = [
            "Step",
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Val MSE",
            "Val PSNR",
        ]

        self.steps: List[int] = []
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_mse: List[float] = []
        self.val_psnr: List[float] = []

        self.curr_step = 0
        self.curr_epoch = 0
        self.curr_train_loss = 0.0
        self.curr_val_loss = 0.0
        self.curr_mse = 0.0
        self.curr_psnr = 0.0
        self.initialized = True

    def set_log_path(self, fp):
        self.csv_out_path = fp

    @classmethod
    def get_instance(cls):
        """
        Returns the singleton instance of ScuffedLogger.
        If the instance doesn't exist, it creates one with the given log_out_path.
        Subsequent calls ignore the log_out_path parameter.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update_reconstruction_error(self, mse: float, psnr: float):
        self.curr_mse = mse
        self.curr_psnr = psnr
        self.curr_epoch += 1
        self.refresh_csv()

    def update_losess(self, loss, loss_dict: Dict):
        self.curr_step += 1
        if "val/loss" in loss_dict.keys():
            self.curr_val_loss = loss
        else:
            self.curr_train_loss = loss
        self.refresh_csv()

    def refresh_csv(self):
        self.steps.append(self.curr_step)
        self.epochs.append(self.curr_epoch)
        self.train_losses.append(self.curr_train_loss)
        self.val_losses.append(self.curr_val_loss)
        self.val_mse.append(self.curr_mse)
        self.val_psnr.append(self.curr_psnr)

        # make the output dir if needed
        out_dir = os.path.dirname(self.csv_out_path)
        os.makedirs(out_dir, exist_ok=True)

        # write to CSV
        with open(self.csv_out_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerows(
                zip(
                    self.steps,
                    self.epochs,
                    self.train_losses,
                    self.val_losses,
                    self.val_mse,
                    self.val_psnr,
                )
            )

    def set_csv_path(self, path: str):
        self.csv_out_path = path


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.global_step = 0
        self.logger: Optional[ExperimentLogger] = None
        self.dataset: Optional[MOS2SEFDataset] = None

    def register_logger(self, logger: ExperimentLogger) -> None:
        self.logger = logger

    def register_dataset(self, dataset: MOS2SEFDataset) -> None:
        self.dataset = dataset

    @rank_zero_only
    def log_local(
        self,
        save_dir: str,
        split: str,
        images,
        global_step: int,
        current_epoch,
        batch_idx,
    ):
        # # TODO: make this code great again
        # # idk what exactly what is happening here; don't really care either
        # root = os.path.join(save_dir, "image_log", split)
        # for k in images:
        #     grid = torchvision.utils.make_grid(images[k], nrow=4)
        #     if self.rescale:
        #         grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        #     grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        #     grid = grid.numpy()
        #     grid = (grid * 255).astype(np.uint8)
        #     filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
        #         k, global_step, current_epoch, batch_idx
        #     )
        #     path = os.path.join(root, filename)
        #     os.makedirs(os.path.split(path)[0], exist_ok=True)
        #     Image.fromarray(grid).save(path)

        pred: torch.Tensor = images["samples_cfg_scale_9.00"].squeeze(0).detach().cpu()
        vae_og_recon: torch.Tensor = images["reconstruction"].squeeze(0).detach().cpu()
        control = images["control"].squeeze(0).detach().cpu()

        # images.keys(): ['reconstruction', 'control', 'conditioning', 'samples_cfg_scale_9.00']
        # "control": [1, 1]
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

        # # 5a. characterize(y)
        # mean, std = self.dataset.current_maps_mean, self.dataset.current_maps_std
        # data = (y - mean) / std
        # y_char = celano_lab_characterization(data, self.dataset.img_size_um)

        # # 5b. characterize(y_sparse)
        # data = (y_hat - mean) / std
        # y_sparse_char = celano_lab_characterization(data, self.dataset.img_size_um)
        
        # 5a. characterize(y)
        y_char = None; y_sparse_char = None

        # log metrics
        self.logger.log(
            **{
                "step": self.global_step,
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
        self.logger.log_original_masked_predicted_sample_triplet(
            y, y_sparse, y_hat, f"e_{current_epoch}_batch_{batch_idx}.png"
        )

        self.global_step += 1

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)  # batch_idx % self.batch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            # logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            # for k in images:
            #     N = min(images[k].shape[0], self.max_images)
            #     images[k] = images[k][:N]
            #     if isinstance(images[k], torch.Tensor):
            #         images[k] = images[k].detach().cpu()
            #         if self.clamp:
            #             images[k] = torch.clamp(images[k], -1.0, 1.0)

            # # TODO: FIX ME.
            # img_out_dir = os.path.join(os.path.dirname(""), "samples")
            # os.makedirs(img_out_dir, exist_ok=True)

            # self.log_local(pl_module.logger.save_dir, split, images,
            #                pl_module.global_step, pl_module.current_epoch, batch_idx)

            self.log_local(
                "",
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Save weights only at the end of the epoch
        self.logger.save_weights(trainer, "last")

        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.logger.save_weights(trainer, "best")
