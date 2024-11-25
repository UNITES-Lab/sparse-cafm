import torch
import cv2
import random
import albumentations as A
import numpy as np

from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
from glob import glob

ORIGINAL_IMAGE_SIZE = (256, 256)
SRC_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/raw-data/11-19-24/2. MoS2 on Sapphire"
EXT = ".tff"


class SapphireDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        steps_per_epoch: int = 100,
        side_length: int = 64,
        device: int = 0,
        original_image_size: tuple = ORIGINAL_IMAGE_SIZE,
    ):
        super(SapphireDataset, self).__init__()
        self.side_length: int = side_length
        self.steps_per_epoch: int = steps_per_epoch
        self.split: str = split
        self.device: int = device
        self.original_image_size: Tuple[int, int] = original_image_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()

        # (B, C, H, W)
        self.current_maps: Optional[List[np.ndarray]] = None
        self.topo_maps = Optional[List[np.ndarray]] = None
        self._load_imgs()

    def _load_imgs(self) -> None:
        all_current_paths = glob(f"{SRC_DIR}/*/*Current*{EXT}")
        all_topo_paths = glob(f"{SRC_DIR}/*/*Current*{EXT}")

        # HACK: hard-code a 4/1 split for train/val sets
        if self.split == "train":
            all_current_paths = all_current_paths[:-1]
            all_topo_paths = all_topo_paths[:-1]
        elif self.split == "val":
            all_current_paths = [all_current_paths[-1]]
            all_topo_paths = [all_topo_paths[-1]]
        else:
            raise Exception(f"Invalid split: {self.split}")

        # load all images
        all_current_imgs = [
            cv2.imread(path, cv2.IMREAD_COLOR) for path in all_current_paths
        ]
        all_topo_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in all_topo_paths]

        # convert all imgs to tensors
        self.current_maps = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for img in all_current_imgs
        ]
        self.topo_maps = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for img in all_topo_imgs
        ]

    def _create_augmentation_pipeline(self):
        # NOTE: we always take a random crop
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
                A.Resize(
                    width=self.original_image_size[0],
                    height=self.original_image_size[1],
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> Dict:
        """
        Get the next randomly sampled item from the dataset.

        :param index:
        :returns:
            {
                'X': torch.Tensor, # topo-map w/ shape [C, H, W]
                'y': torch.Tensor, # target current-map w/ shape [C, H, W]
                'z': torch.Tensor, # scalar-valued target denoting 'current-under-threshold'
            }
        """

        # HACK: hard-coded train/val splits
        # choose a random sample idx
        sample_idx = 0
        if self.split == "train":
            sample_idx = random.randint(0, len(self.current_maps) - 1)
        elif self.split == "val":
            sample_idx = len(self.current_maps) - 1
        else:
            raise Exception(f"Invalid split: {self.split}")
        
        X = self.topo_maps[sample_idx]
        y = self.current_maps[sample_idx]
        
        # apply augmentations
        # convert -> tensor
        augmented = self.augmentation_pipeline(image=X, mask=y)
        X = torch.tensor(augmented['image']).permute(2, 0, 1).float().to(self.device)
        y = torch.tensor(augmented['mask']).permute(2, 0, 1).float().to(self.device)
        return
