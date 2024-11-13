import os
import csv
from cv2 import log
import numpy as np
import torch
import torchvision
from typing import List
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

# TODO: create a custom callback
# 1. MSE train loss @ each step
# 2. MSE (i.e., pixel-loss) validation loss @ each epoch
# 3. PSNR @ each epoch

class ScuffedLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create and remember the instance
            cls._instance = super(ScuffedLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_out_path: str) -> None:
        if hasattr(self, 'initialized') and self.initialized:
            # If already initialized, do not reinitialize
            return

        self.csv_out_path = log_out_path
        self.HEADERS = ['Step', 'Epoch', 'Train Loss', 'Val Loss', 'Val PSNR']

        self.steps: List[int] = []
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_mse: List[float] = []
        self.val_psnr: List[float] = []

        self.curr_step = 0
        self.curr_epoch = 0
        self.curr_train_loss = 0.0
        self.curr_mse = 0.0
        self.curr_psnr = 0.0

        self.initialized = True  # Flag to prevent reinitialization

    @classmethod
    def get_instance(cls, log_out_path: str):
        """
        Returns the singleton instance of ScuffedLogger.
        If the instance doesn't exist, it creates one with the given log_out_path.
        Subsequent calls ignore the log_out_path parameter.
        """
        if cls._instance is None:
            cls._instance = cls(log_out_path)
        return cls._instance

    def update_val_stats(self, mse: float, psnr: float, epoch: int):
        self.curr_mse = mse
        self.curr_psnr = psnr
        self.curr_epoch = epoch
        self.refresh_csv()

    def update_train_loss(self, loss: float):
        self.curr_step += 1
        self.curr_train_loss = loss
        self.refresh_csv()

    def refresh_csv(self):
        self.steps.append(self.curr_step)
        self.epochs.append(self.curr_epoch)
        self.train_losses.append(self.curr_train_loss)
        self.val_mse.append(self.curr_mse)
        self.val_psnr.append(self.curr_psnr)

        # Write to CSV
        with open(self.csv_out_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerows(zip(self.steps, self.epochs, self.train_losses, self.val_mse, self.val_psnr))
    
class LossLogger(Callback):
    def __init__(self, log_dir='logs', log_file='metrics_log.csv'):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = log_file
        self.train_losses = []
        self.val_losses = []
        self.val_psnr = []

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        self.filepath = os.path.join(self.log_dir, self.log_file)

        # Initialize CSV file with headers
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Epoch', 'Train Loss', 'Val Loss', 'Val PSNR'])

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        # TODO: Assume 'loss' is returned by the training step
        train_loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        global_step = trainer.global_step
        epoch = trainer.current_epoch

        self.train_losses.append(train_loss)

        # Log to CSV
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, epoch, train_loss, '', ''])  # Val metrics left blank

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # Assume 'val_loss' and 'val_output' are logged in validation step
        # You might need to adjust based on how your validation step logs metrics

        # Retrieve the logged validation loss
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None and isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        self.val_losses.append(val_loss)

        # Calculate PSNR
        
        # TODO:
        # Assuming your validation step logs 'reconstructions' and 'targets'
        reconstructions = trainer.callback_metrics.get('reconstructions')  # Tensor: [batch, C, H, W]
        targets = trainer.callback_metrics.get('targets')  # Tensor: [batch, C, H, W]

        if reconstructions is not None and targets is not None:
            psnr = self.calculate_psnr(reconstructions, targets)
            self.val_psnr.append(psnr)
        else:
            psnr = None

        global_step = trainer.global_step
        epoch = trainer.current_epoch

        # Log to CSV
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, epoch, '', val_loss, psnr])

    def calculate_psnr(self, reconstructions, targets):
        # Ensure tensors are in the same device
        reconstructions = reconstructions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        mse = np.mean((reconstructions - targets) ** 2, axis=(1, 2, 3))
        # To avoid division by zero
        mse = np.maximum(mse, 1e-10)
        psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)  # Assuming pixel values are in [0,1]
        return np.mean(psnr)

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
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

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
