import torch
import cv2
import os
import random
import albumentations as A
import numpy as np

from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List, Union
from glob import glob

# hacky; should most likely be removed
Z_MULT = 1
ORIGINAL_IMAGE_SIZE = (256, 256)
CROPPED_IMG_SIDE_LENGTH = 64
SRC_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/raw-data/11-19-24/2. MoS2 on Sapphire"
EXT = "tiff"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


class SapphireDataset(Dataset):
    """
    Dataset class for MoS2 samples collected on a Sapphire substrate.

    :Definitions:
    - X: topography map (height, width, depth)
    - y: current map (height, width, current)
    - z: scalar, current-under-threshold
    - epsilon: bottom 10th percentile threshold of current values
    """

    def __init__(
        self,
        split: str = "train",
        steps_per_epoch: int = 100,
        side_length: int = CROPPED_IMG_SIDE_LENGTH,
        device: int = 0,
        z_mult: Union[int, float] = Z_MULT,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        :param split: "train" or "val"
        :param steps_per_epoch: data is sampled using random augmentations, therefore the # sample per epoch is arbitrary
        :param side_length: length of the side of the square crops taken from the original, full-sized image
        :param device: number of CUDA device, not currently used
        :param z_mult: scalar norm value for z; z = z * z_mult
        :param original_image_size: size of the original images in the dataset: e.g., (256, 256)
        """

        super(SapphireDataset, self).__init__()
        self.side_length: int = side_length
        self.steps_per_epoch: int = steps_per_epoch
        self.split: str = split
        self.device: int = device
        self.z_mult: int = z_mult
        self.original_image_size: Tuple[int, int] = original_image_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()

        # (B, C, H, W)
        self.current_maps: tuple = None
        self.topo_maps: tuple = None

        # load all data from src files
        self._load_imgs()

        # calculate the value of ε: the bottom 10th percentile of raw current-map readings
        self.epsilon: Optional[float] = None
        self._calculate_epsilon()

    def _calculate_epsilon(self):
        """
        Calculate epsilon: the bottom 10th percentile of raw current-map readings.
        """
        # currently, we calculate epsilon globally (i.e., using both training and validation data)
        # persumably this is not an issue, as we expect to use a global epsilon value once ground-truth data is provided

        # all current readings (measured in nA) append as a 1D array
        all_current_readings = []
        for fp in self._raw_current_fps:
            data = np.load(fp)
            all_current_readings.append(data)
        # reshape into 1d vector
        all_current_readings = np.array(all_current_readings).reshape(-1)
        assert (
            all_current_readings.ndim == 1
        ), f"Error: could not calculate global epsilon. Expected a 1D array, got {all_current_readings.ndim}"
        # calculate the bottom 10th percentile of epsilon readings
        self.epsilon = np.percentile(all_current_readings, 10)

    def _load_imgs(self) -> None:
        """
        Load current-map + topo-images into memory from data source dir.
        """

        data_path_regex = f"{SRC_DIR}/*/*Current*.npy"
        self._raw_current_fps = glob(data_path_regex)
        assert (
            len(self._raw_current_fps) > 0
        ), f"Error: could not load images using pattern: {data_path_regex}"

        # load in current maps with shape...?
        # TODO: can we be ABSOLUTELY sure that current maps are correctly paired with topo maps
        self._raw_current_maps = [np.load(fp) for fp in self._raw_current_fps]

        _current_fps = glob(f"{SRC_DIR}/*/*Current*{EXT}")
        _topo_fps = glob(f"{SRC_DIR}/*/*Topo*{EXT}")

        # HACK: use hard-coded substring to match up current, topo pairs
        _current_fps_basenames = [
            os.path.basename(fp).split("Current")[0] for fp in _current_fps
        ]
        _topo_fps_basenames = [
            os.path.basename(fp).split("Topo")[0] for fp in _topo_fps
        ]
        assert (
            _current_fps_basenames == _topo_fps_basenames
        ), f"Error: misalignment of current maps and topo maps during dataloading"

        # load in current + topography images with `.tiff` extentions from memory
        all_current_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in _current_fps]
        all_topo_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in _topo_fps]
        assert len(all_current_imgs) > 0, f"Error: no current-map images detected."
        assert len(all_topo_imgs) > 0, f"Error: no topography-map images detected."

        # is this step necessiary?
        # convert all topo + current maps to np arrays
        self.current_maps = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_current_imgs
        ]
        self.topo_maps = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_topo_imgs]

    def _create_augmentation_pipeline(self):
        # TODO: add config support for augmenatition pipeline initialization
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
                A.Resize(
                    width=self.original_image_size[0],
                    height=self.original_image_size[1],
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            additional_targets={
                "y": "mask",
                "X_og": "mask",
                "y_og": "mask",
            },
        )

    def __len__(self):
        """
        len(self) == self.steps_per_epoch
        """
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> Dict:
        """
        Get the next randomly sampled item from the dataset.

        :param index: currently unused, necessiary for batch data-loading
        :returns:
            ```
                {

                    'X': torch.Tensor, # topo-map w/ shape [C, H, W]
                    'X_og': torch.Tensor, # topo-map w/ shape [H, W, C]
                    'y': torch.Tensor, # target current-map w/ shape [C, H, W]
                    'z': torch.Tensor, # scalar-valued target denoting 'current-under-threshold'
                }
        """

        # HACK: hard-coded train/val splits
        # choose a random sample idx
        if self.split == TRAIN_SPLIT:
            # randint is inclusive: [a, b]
            # select a random sample from self.data[:-1]
            sample_idx = random.randint(0, len(self.current_maps) - 2)
        elif self.split == VAL_SPLIT:
            # select the final data sample: self.data[-1]
            sample_idx = len(self.current_maps) - 1
        else:
            raise Exception(f"Invalid split: {self.split}")

        # get topography map with normalized depth dim
        X: np.ndarray = self.topo_maps[sample_idx]
        # copy of original X for figure logging
        X_og = X.copy()

        y: np.ndarray = self.current_maps[sample_idx]
        # copy of original y for figure logging
        y_og = y.copy()
        # raw (H, W) current map; unnormalized (very small) current values
        y_raw: np.ndarray = self._raw_current_maps[sample_idx]

        # augment samples
        # X recieves pixel-value normalization, all other data do not
        augmented = self.augmentation_pipeline(
            image=X, mask=y_raw, y=y, X_og=X_og, y_og=y_og
        )

        # convert all data -> tensor
        X = torch.tensor(augmented["image"]).permute(2, 0, 1).float()
        y = torch.tensor(augmented["y"]).permute(2, 0, 1).float()
        X_og = torch.tensor(augmented["X_og"])
        y_og = torch.tensor(augmented["y_og"])
        # stays as a np.ndarray
        y_raw = augmented["mask"]

        # MARK: calculate values of z
        # z: sum(pixels < self.epsilon) / total num pixels
        z = (y_raw.flatten() < self.epsilon).sum() / (y_raw.shape[0] * y_raw.shape[1])

        # TODO: what is the ideal way to normalize z?
        z = torch.tensor(z).float() * self.z_mult

        # z should always be in range: [0, 1.0 * Z_MULT]
        assert z >= 0.0 and z <= (1.0 * self.z_mult)

        return {
            "X": X,
            "y": y,
            "z": z,
            "X_og": X_og,
            "y_og": y_og,
            "epsilon": self.epsilon,
        }
