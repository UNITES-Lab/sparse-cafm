import torch
import cv2
import albumentations as A

from typing import Tuple, Dict
from torch.utils.data import Dataset
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation
from src.util.celano_lab_scripts import process_image

CROPPED_IMAGE_SIDE_LENGTH = 128
ORIGINAL_IMAGE_SIZE = (512, 512)

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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        
        batch: dict  = self.dataset[index]
        
        # [B, H, W]
        y: torch.Tensor = batch["y"]
        y_char = process_image(y, self.dataset.img_size_um)
        
        # ---- normalize all vals -> std normal ----
        for k in y_char:
            val = y_char[k]
            mean = CHARACTERISTIC_NORMALIZATION_DICT[k]['mean']
            std = CHARACTERISTIC_NORMALIZATION_DICT[k]['std']
            y_char[k] = (val - mean) / std
            
        target_arr = []
        keys_sorted = sorted(list(y_char.keys()))
        for k in keys_sorted:
            target_arr.append(y_char[k])
        target = torch.Tensor(target_arr).float()
        
        item = y_char.copy()
        item['y'] = y
        item['target'] = target
        return item

if __name__ == "__main__":
    ds = MOS2SefOLDERSurrogate()
    _ = ds[0]