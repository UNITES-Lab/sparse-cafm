import pprint
import torch
import cv2
import random
import albumentations as A
import torch.nn.functional as F

from typing import Tuple, Dict
from torch.utils.data import Dataset
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation
from src.util.celano_lab_scripts import process_image

CROPPED_IMAGE_SIDE_LENGTH = 128
ORIGINAL_IMAGE_SIZE = (512, 512)

# NOTE: these values are calculated by sampling 10k times from
# train a val sets of MOS2SEFDataset. We directly feed y into characterization script.
CHARACTERISTIC_NORMALIZATION_DICT = {
    "coverage_percentage": 
        {
            "mean": 44.20514771,
            "std": 18.16990309,
        },
    "total_len_detected_curves": 
        {
            "mean": 18.40427719,
            "std": 4.57835860,
        },
    "total_area_circular_shapes":
        {
            "mean": 0.91592283,
            "std": 0.14692457,
        },
    "total_area_extended_shapes": 
        {
            "mean": 0.03225828, 
            "std": 0.05968977,
        },
    "total_defect_area": 
        {
            "mean": 0.0057872382,
            "std": 0.0008519035,
        },
    "num_circular_shapes": 
        {
            "mean": 75.2265000000, 
            "std": 14.1692059675,
        },
    "num_extended_shapes": 
        {
            "mean": 0.3658000000, 
            "std": 0.6335537546,
        },
    "num_curved_lines": 
        {
            "mean": 1.6506000000, 
            "std": 0.8053071712,
        },
    "average_surface_current": 
        {
            "mean": 495686994.9221611619, 
            "std": 62729659.3002319783
        },
}

class MOS2SefOLDERSurrogate(Dataset):
    """
    Dataset class used to train an OLDER-surrogate model.
    """
    
    def __init__(
        self,
        split: str = "train",
        formulation: Formulation = Formulation.P_Y_BAR_Y_SPARSE,
        side_length: int = CROPPED_IMAGE_SIDE_LENGTH,
        masking_ratio: int = 0,
        steps_per_epoch: int = 100,
        device: int = 0,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        self.dataset = MOS2SEFDataset(
            split=split,
            formulation=formulation,
            side_length=side_length,
            masking_ratio=masking_ratio,
            steps_per_epoch=steps_per_epoch,
            device=device,
            original_image_size=original_image_size,
        )
    
    def scale_image(self, image, scale):
        """
        Scales the input image by `scale` (using bilinear interpolation)
        and then crops a central patch of the original size.
        
        Args:
            image (torch.Tensor): A 2D tensor of shape (H, W).
            scale (float): Scaling factor.
            
        Returns:
            torch.Tensor: Processed image of shape (H, W).
        """
        # Original dimensions
        H, W = image.shape
        # New dimensions after scaling
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Scale the image: add batch and channel dims for interpolation.
        image_scaled = F.interpolate(image.unsqueeze(0).unsqueeze(0).float(),
                                    size=(new_H, new_W),
                                    mode='bilinear',
                                    align_corners=False)
        # Remove batch and channel dims
        image_scaled = image_scaled.squeeze(0).squeeze(0)
        
        # Crop the central region of size (H, W)
        start_H = (new_H - H) // 2
        start_W = (new_W - W) // 2
        cropped_image = image_scaled[start_H:start_H+H, start_W:start_W+W]
        
        return cropped_image

    def __len__(self) -> int:
        return len(self.dataset)

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
        
        y_char = process_image(y, self.dataset.img_size_um)
        
        # ---- bootstrap y_char 10x ----
        NUM_BOOTSTRAPS = 10

        for i in range(NUM_BOOTSTRAPS - 1):
            
            y_aug = y.clone()
            if random.random() < 0.5:
                y_aug = torch.flip(y_aug, dims=[1])
            if random.random() < 0.5:
                y_aug= torch.flip(y_aug, dims=[0])
            if y.shape[0] == y.shape[1] and random.random() < 0.5:
                y_aug = torch.rot90(y_aug, k=1, dims=(0, 1))
            if random.random() < 0.5:
                factor = random.uniform(0.9, 1.1)
                y_aug = y_aug * factor
            if random.random() < 0.5:
                noise_std = 0.05 * (y_aug.max() - y_aug.min())
                noise = torch.randn_like(y_aug) * noise_std
                y_aug = y_aug + noise
            if random.random() < 0.5:
                # scale randomly 1x-1.3x
                y_aug = self.scale_image(y_aug, 1 + (random.random() * 0.3))

            y_char_bootstrapped = process_image(y_aug, self.dataset.img_size_um)
            
            for k, v in y_char.items():
                y_char[k] = (y_char[k] + y_char_bootstrapped[k])
        
        for k, v in y_char.items():
            y_char[k] = (y_char[k] + y_char_bootstrapped[k]) / NUM_BOOTSTRAPS
        
        # ---- normalize all vals -> std normal ----
        for k in y_char:
            val = y_char[k]
            mean = CHARACTERISTIC_NORMALIZATION_DICT[k]['mean']
            std = CHARACTERISTIC_NORMALIZATION_DICT[k]['std']
            y_char[k] = (val - mean) / std
        
        # # HACK: we move three high-variance features
        # y_char.pop("num_curved_lines")
        # y_char.pop("num_extended_shapes")
        # y_char.pop("total_area_extended_shapes")
        
        target_arr = []
        
        # TODO: this may change the order of keys/features
        keys_sorted = sorted(list(y_char.keys()))
        for k in keys_sorted:
            target_arr.append(y_char[k])
        
        target = torch.Tensor(target_arr).float()
        
        item = {}
        item['y'] = y
        item['y_char'] = y_char
        item['target'] = target
        
        return item

if __name__ == "__main__":
    ds = MOS2SefOLDERSurrogate()
    _ = ds[0]