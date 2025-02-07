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
    "coverage_percentage": None,
    "total_len_detected_curves": None,
    "total_area_circular_shapes": None,
    "total_area_extended_shapes": None,
    "total_defect_area": None,
    "num_circular_shapes": None,
    "num_extended_shapes": None,
    "num_curved_lines": None,
    "average_surface_current": None,
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
        y_char = process_image(y)

        