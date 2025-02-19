import cv2
import concurrent
import torch
import random
import pprint
import numpy as np
import albumentations as A
import torchvision.transforms as T
import torchvision.transforms.functional as F

from glob import glob
from tqdm import tqdm
from typing import Tuple, Dict
from torch.utils.data import Dataset
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation
from src.util.celano_lab_scripts import process_image
from src.util.metrics import OLDER, PSNR, MSE, MAE, SSIM

NUM_CHAR_FEATURES = 4
CROPPED_IMAGE_SIDE_LENGTH = 128
ORIGINAL_IMAGE_SIZE = (512, 512)


def scale_image(image: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scales the input image by `scale` (using bilinear interpolation)
        and then crops a central patch of the original size.
        
        Args:
            image (torch.Tensor): A 2D tensor of shape (H, W).
            scale (float): Scaling factor.
            
        Returns:
            torch.Tensor: Processed image of shape (H, W).
        """

        H, W = image.shape
        new_H, new_W = int(H * scale), int(W * scale)
        image_scaled = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0).float(),
                                    size=(new_H, new_W),
                                    mode='bilinear',
                                    align_corners=False)
        image_scaled = image_scaled.squeeze(0).squeeze(0)
        start_H = (new_H - H) // 2
        start_W = (new_W - W) // 2
        cropped_image = image_scaled[start_H:start_H+H, start_W:start_W+W]
        return cropped_image


def augment(y: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of random augmentations on the image tensor `y` and processes it.
    Returns the dictionary from process_image.
    """
    
    y_aug = y.clone()
    if random.random() < 0.5:
        y_aug = torch.flip(y_aug, dims=[1])
    if random.random() < 0.5:
        y_aug = torch.flip(y_aug, dims=[0])
    if y.shape[0] == y.shape[1] and random.random() < 0.5:
        y_aug = torch.rot90(y_aug, k=1, dims=(0, 1))
    if random.random() < 0.5:
        factor = random.uniform(0.99, 1.01)
        y_aug = y_aug * factor
    if random.random() < 0.5:
        noise_std = 0.05 * (y_aug.max() - y_aug.min())
        noise = torch.randn_like(y_aug) * noise_std
        y_aug = y_aug + noise
    if random.random() < 0.5:
        scale_factor = 1 + (random.random() * 0.03)
        y_aug = scale_image(y_aug, scale_factor)
    
    return y_aug


class MOS2SefOLDERContrastiveDataset(Dataset):
    """
    Dataset class used to train a DoGE module.
    """
    
    S_SIM_SSIM = 0.95
    S_SIM_OLDER = 0.20

    S_DIFF_SSIM = 0.30
    S_DIFF_OLDER = 0.70

    def __init__(
        self,
        split: str = "train",
        formulation: Formulation = Formulation.P_Y_BAR_Y_SPARSE,
        side_length: int = CROPPED_IMAGE_SIDE_LENGTH,
        masking_ratio: int = 0,
        steps_per_epoch: int = 100,
        device: int = 0,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
        normalize_on_init: bool = False,
    ):
        self.split = split
        self.formulation = formulation
        self.side_length = side_length
        self.masking_ratio = masking_ratio
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.original_image_size = original_image_size
        
        self.dataset = MOS2SEFDataset(
            split=split,
            formulation=formulation,
            side_length=side_length,
            masking_ratio=masking_ratio,
            steps_per_epoch=steps_per_epoch,
            device=device,
            original_image_size=original_image_size,
        )
        
        # dictionary of {"mean": float, "std": float} values
        self.normalization_dict: Dict[str, Dict] = {}
        
        # optional: run a short benchmark to determine normalization mean/std
        if normalize_on_init: self.normalize()

    def get_similar_sample(self, y: torch.Tensor) -> torch.Tensor:
        """
        Where y ~ std_norm.
        """ 

        # -> original distribution
        y_unnorm = (y - self.dataset.current_maps_mean) / self.dataset.current_maps_std
        
        y_img_like = y.clone()
        # (H, W) -> (1, H, W)
        y_img_like = y_img_like.unsqueeze(0)
        # (H, W) -> (3, H, W)
        y_img_like = y_img_like.repeat(3, 1, 1)
        # (3, H, W) -> (1, 3, H, W)
        y_img_like = y_img_like.unsqueeze(0)

        y_aug = None

        while True:

            # randomly sample an augmented y
            y_aug = augment(y)

            # -> original distribution
            y_aug_unnorm = (y_aug - self.dataset.current_maps_mean) / self.dataset.current_maps_std

            y_aug_img_like = y_aug.clone()
            # (H, W) -> (1, H, W)
            y_aug_img_like = y_aug_img_like.unsqueeze(0)
            # (1, H, W) -> (3, H, W)
            y_aug_img_like = y_aug_img_like.repeat(3, 1, 1)
            # (3, H, W) -> (1, 3, H, W)
            y_aug_img_like = y_aug_img_like.unsqueeze(0)

            ssim = SSIM(y_img_like, y_aug_img_like)
            
            # HACK: hard-coded iamge size
            older = OLDER(process_image(y_unnorm, 2.0), process_image(y_aug_unnorm, 2.0))

            l1 = torch.nn.functional.l1_loss(y, y_aug)

            if ssim > self.S_SIM_SSIM and older < self.S_SIM_OLDER and l1 > 0.0:
                break

        return y_aug


    def get_contrastive_sample(self, y: torch.Tensor, index: int) -> torch.Tensor: 
        """
        Where y ~ std_norm.
        """ 

        # -> original distribution
        y_unnorm = (y - self.dataset.current_maps_mean) / self.dataset.current_maps_std
        
        y_img_like = y.clone()
        # (H, W) -> (1, H, W)
        y_img_like = y_img_like.unsqueeze(0)
        # (H, W) -> (3, H, W)
        y_img_like = y_img_like.repeat(3, 1, 1)
        # (3, H, W) -> (1, 3, H, W)
        y_img_like = y_img_like.unsqueeze(0)

        y_aug = None

        while True:

            # randomly sample an augmented y
            y_aug = self.dataset[index + 1]["y"]

            # -> original distribution
            y_aug_unnorm = (y_aug - self.dataset.current_maps_mean) / self.dataset.current_maps_std

            y_aug_img_like = y_aug.clone()
            # (H, W) -> (1, H, W)
            y_aug_img_like = y_aug_img_like.unsqueeze(0)
            # (1, H, W) -> (3, H, W)
            y_aug_img_like = y_aug_img_like.repeat(3, 1, 1)
            # (3, H, W) -> (1, 3, H, W)
            y_aug_img_like = y_aug_img_like.unsqueeze(0)

            ssim = SSIM(y_img_like, y_aug_img_like)
            
            # HACK: hard-coded iamge size
            older = OLDER(process_image(y_unnorm, 2.0), process_image(y_aug_unnorm, 2.0))

            l1 = torch.nn.functional.l1_loss(y, y_aug)

            if ssim < self.S_DIFF_SSIM and older > self.S_DIFF_OLDER:
                break

        return y_aug

    def __len__(self) -> int: return len(self.dataset)

    def normalize(self) -> None:
        """
        Run a short test proceedure to calculate the mean and std of train/val samples;
        set global values for mean/std so that all samples are normalized roughly to the std normal.
        We make the apriori assumption that train/val samples belong to roughly the same distribution.
        """
        
        NUM_BENCHMARK_STEPS = 1000
        train_dataset = MOS2SEFDataset(
            split="train",
            formulation=self.formulation,
            side_length=self.side_length,
            masking_ratio=self.masking_ratio,
            steps_per_epoch=NUM_BENCHMARK_STEPS,
            device=self.device,
            original_image_size=self.original_image_size,
        )
        val_dataset = MOS2SEFDataset(
            split="val",
            formulation=self.formulation,
            side_length=self.side_length,
            masking_ratio=self.masking_ratio,
            steps_per_epoch=NUM_BENCHMARK_STEPS,
            device=self.device,
            original_image_size=self.original_image_size,
        )
        
        samples = {}
        
        # samples from train/val datasets
        for idx in tqdm(range(NUM_BENCHMARK_STEPS), total=NUM_BENCHMARK_STEPS, desc="Calculating global mean/stds.."):
            
            train_item = train_dataset.__getitem__(idx)
            val_item = val_dataset.__getitem__(idx)
            train_y = train_item["y"]; val_y = val_item["y"]
            
            # characterize train/val current-maps
            train_char = process_image(train_y, self.dataset.img_size_um)
            val_char = process_image(val_y, self.dataset.img_size_um)
            
            for k, v in train_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)
            
            for k, v in val_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)

        for k, v in samples.items():
            self.normalization_dict[k] = {
                "mean": np.mean(v),
                "std": np.std(v),
            }

    def __getitem__(self, index: int) -> Dict:
        """
        Provide a current-map y and a "target" Tensor.
        Returns
        ---
        {
            "y": torch.Tensor: [H, W]
            "target": torch.Tensor: [9]
                - All nine Celano-Lab characterisitics normalized to standard normal.
        }
        """
        
        batch: dict  = self.dataset[index]
        
        # [H, W]
        y: torch.Tensor = batch["y"]
        y_sim = self.get_similar_sample(y)
        y_con = self.get_contrastive_sample(y, index)

        breakpoint()
        
        item = {}
        item['y'] = y
        item['y_sim'] = y_sim
        item['y_con'] = y_con

        return item
    
if __name__ == "__main__":
    dataset = MOS2SefOLDERContrastiveDataset()
    item = dataset[0]