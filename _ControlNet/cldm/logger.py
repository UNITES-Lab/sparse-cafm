import math
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from typing import List, Dict, Optional
from cldm.metrics import calc_psnr
from cv2 import log
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from src.util.logger import ExperimentLogger


# TODO: create a custom callback
# 1. MSE train loss @ each step
# 2. MSE (i.e., pixel-loss) validation loss @ each epoch
# 3. PSNR @ each epoch


# TODO: implement
class Logger:
    def __init__(self) -> None:
        pass


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

    def register_logger(self, logger: ExperimentLogger) -> None:
        self.logger = logger

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

        # TODO: make this code great again
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

        # (B, C, H, W ) -> (3, 512, 512)
        # conditioning: just an array of 1s?
        breakpoint()
        pred: torch.Tensor = images["samples_cfg_scale_9.00"].squeeze(0).detach().cpu()
        target: torch.Tensor = images["reconstruction"].squeeze(0).detach().cpu()

        # control: [-1, 1]
            # target img (y)
        # pred: [-1, 1]
        # target: [-1, 1]
        l1 = F.l1_loss(pred, target).item()
        mse = F.mse_loss(pred, target).item()
        psnr = calc_psnr(pred, target)
        
        assert isinstance(self.logger, ExperimentLogger)
        self.logger.log(**{
            'step': self.global_step,
            'train_l1': l1 if split == 'train' else None,
            'train_psnr': psnr if split == 'train' else None,
            'val_l1': l1 if split == 'val' else None,
            'val_psnr': psnr if split == 'val' else None,
        })
        self.global_step += 1

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)  # batch_idx % self.batch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            # TODO: FIX ME.
            img_out_dir = os.path.join(
                os.path.dirname(""), "samples"
            )
            os.makedirs(img_out_dir, exist_ok=True)

            # self.log_local(pl_module.logger.save_dir, split, images,
            #                pl_module.global_step, pl_module.current_epoch, batch_idx)

            self.log_local(
                img_out_dir,
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
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")
