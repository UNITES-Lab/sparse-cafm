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
EXT = "tiff"


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
        self.current_maps: tuple = None
        self.topo_maps: tuple = None
        self._load_imgs()

        # threshold representing bottom 10th percentile of current values
        self.epsilon: Optional[float] = None
        self._calculate_epsilon()

    def _calculate_epsilon(self):
        all_current_readings = []
        # for all files in sample_fps
        # open data with shape 512, 512
        # append all values to all_current_readings
        for fp in self._raw_current_fps:
            data = np.load(fp)
            all_current_readings.append(data)
        # reshape into 1d vector
        all_current_readings = np.array(all_current_readings).reshape(-1)
        # calculate the bottom 10th percentile
        self.epsilon = np.percentile(all_current_readings, 10)

    def _load_imgs(self) -> None:
        self._raw_current_fps = glob(f"{SRC_DIR}/*/*Current*.npy")
        self._raw_current_maps = [np.load(fp) for fp in self._raw_current_fps]

        _current_fps = glob(f"{SRC_DIR}/*/*Current*{EXT}")
        _topo_fps = glob(f"{SRC_DIR}/*/*Topo*{EXT}")
        
        # load all images
        all_current_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in _current_fps]
        all_topo_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in _topo_fps]
        self.current_maps = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_current_imgs
        ]
        # convert all topo maps to img tensors
        self.topo_maps = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_topo_imgs]

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
            ```
            {
                'X': torch.Tensor, # topo-map w/ shape [C, H, W]
                'X_og': torch.Tensor, # topo-map w/ shape [H, W, C]
                'y': torch.Tensor, # target current-map w/ shape [C, H, W]
                'z': torch.Tensor, # scalar-valued target denoting 'current-under-threshold'
            }
            ```
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

        X: torch.Tensor = self.topo_maps[sample_idx]
        X_og = X.copy()
        # raw (H, W) current map
        y: torch.Tensor = self.current_maps[sample_idx]
        y_og: torch.Tensot = y.copy()
        y_raw: np.ndarray = self._raw_current_maps[sample_idx]

        # z: #  pixels < self.epsilon divided by total # pixels
        z = (y_raw.flatten() < self.epsilon).sum() / (
            y_raw.shape[0] * y_raw.shape[1]
        )
        z = torch.tensor(z).float()
        # z should always be in range: [0, 1]
        assert z >= 0.0 and z <= 1.0

        # apply augmentations
        # convert -> tensor
        augmented = self.augmentation_pipeline(image=X, mask=y)
        X = torch.tensor(augmented["image"]).permute(2, 0, 1).float()
        y = torch.tensor(augmented["mask"]).permute(2, 0, 1).float()
        return {
            "X": X.float(),
            "X_og": X_og,
            "y": y.float(),
            "y_og": y_og,
            "z": z,
            "epsilon": self.epsilon,
        }
