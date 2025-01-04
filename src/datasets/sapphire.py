import torch
import cv2
import os
import random
import albumentations as A
import numpy as np

from enum import Enum
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List, Union
from glob import glob

# hacky; should most likely be removed
Z_MULT = 1
ORIGINAL_IMAGE_SIZE = (224, 224)
CURRENT_PERCENTILE_CUTOFF = 10
CROPPED_IMG_SIDE_LENGTH = 64
SRC_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/raw-data/11-19-24/2. MoS2 on Sapphire"
EXT = "tiff"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


class Formulation(Enum):
    P_Z_BAR_X = 0
    P_Y_BAR_X = 1
    P_Y_BAR_X_CN = 2


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
        formulation: Formulation = Formulation.P_Y_BAR_X,
        side_length: int = CROPPED_IMG_SIDE_LENGTH,
        masking_ratio: int = 0,
        steps_per_epoch: int = 100,
        device: int = 0,
        z_mult: Union[int, float] = Z_MULT,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        :param split: "train" or "val"
        :param steps_per_epoch: data is sampled using random augmentations, therefore the # sample per epoch is arbitrary
        :param side_length: length of the side of the square crops taken from the original, full-sized image
        :param device: number of CUDA device, not currently used
        :param masking_ratio: 1-in-{masking_ratio} pixels masked
        :param z_mult: scalar norm value for z; z = z * z_mult
        :param original_image_size: size of the original images in the dataset: e.g., (256, 256)
        """

        super(SapphireDataset, self).__init__()
        self.side_length: int = side_length
        self.steps_per_epoch: int = steps_per_epoch
        self.split: str = split
        self.formulation: Formulation = formulation
        self.device: int = device
        self.z_mult: int = z_mult
        self.masking_ratio: int = masking_ratio
        self.original_image_size: Tuple[int, int] = original_image_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()

        # (B, C, H, W)
        self.current_maps: tuple = None
        self.topo_maps: tuple = None

        # paths to un-normalized, high-precision current maps
        self._raw_current_fps: Optional[List[str]] = None
        self._raw_topo_fps: Optional[List[str]] = None

        # for normalizing X, y, respectively later
        self.current_maps_mean = 0.0
        self.current_maps_std = 0.0
        self.topo_maps_mean = 0.0
        self.topo_maps_std = 0.0

        # load all data from src files
        self._load_imgs()

        # remove L -> R gradients
        # self._remove_gradients()

        # find the mean/std of current and topo maps
        self._calculate_mean_std()

        # calculate the value of ε: the bottom 10th percentile of raw current-map readings
        self.epsilon: Optional[float] = None
        self._calculate_epsilon()

    def _save_unnormalized_img(self, img: np.ndarray, to: str):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # 1. normalize array to [0, 255]
        img_min = np.min(img)
        img_max = np.max(img)
        img = (img - img_min) / (img_max - img_min)
        img = img * 255
        # 2. convert to uint8
        img = img.astype(np.uint8)
        img = img.transpose(1, 2, 0)
        # 3. save
        cv2.imwrite(to, img)

    def _calculate_epsilon(self):
        """
        Calculate epsilon: the bottom 10th percentile of raw current-map readings.
        """
        # currently, we calculate epsilon globally (i.e., using both training and validation data)
        # persumably this is not an issue; we will use a global epsilon value once ground-truth data is provided
        # all current readings (measured in nA) append as a 1D array
        all_current_readings = []
        for data in self.current_maps:
            all_current_readings.append(data)
        # reshape into 1d vector
        all_current_readings = np.array(all_current_readings).reshape(-1)
        assert (
            all_current_readings.ndim == 1
        ), f"Error: could not calculate global epsilon. Expected a 1D array, got {all_current_readings.ndim}"
        # calculate the bottom 10th percentile of epsilon readings
        self.epsilon = np.percentile(all_current_readings, CURRENT_PERCENTILE_CUTOFF)

    def _load_imgs(self) -> None:
        """
        Load current-map + topo-map data from source dir.
        """

        current_map_regex = f"{SRC_DIR}/*/*Current*.npy"
        topo_map_regex = f"{SRC_DIR}/*/*Topo*.npy"

        self._raw_current_fps = glob(current_map_regex)
        self._raw_topo_fps = glob(topo_map_regex)

        assert (
            len(self._raw_current_fps) > 0
        ), f"Error: could not load images using regex: {current_map_regex}"
        assert (
            len(self._raw_topo_fps) > 0
        ), f"Error: could not load images using regex: {current_map_regex}"

        # (H, W)
        self.current_maps: List[np.ndarray] = [
            np.load(fp) for fp in self._raw_current_fps
        ]
        self.topo_maps: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_fps]

        # validate current, topo map paris are aligned
        _current_fps_basenames = [
            os.path.basename(fp).split("Current")[0] for fp in self._raw_current_fps
        ]
        _topo_fps_basenames = [
            os.path.basename(fp).split("Topo")[0] for fp in self._raw_topo_fps
        ]
        assert (
            _current_fps_basenames == _topo_fps_basenames
        ), f"Error: misalignment of current maps and topo maps during dataloading"

        # convert maps to type -> float64
        self.current_maps = [cm.astype(np.float64) for cm in self.current_maps]
        self.topo_maps = [tm.astype(np.float64) for tm in self.topo_maps]

        # (H, W) -> (H, W, C)
        self.current_maps = [np.stack([cm] * 3, axis=-1) for cm in self.current_maps]
        self.topo_maps = [np.stack([tm] * 3, axis=-1) for tm in self.topo_maps]

        # HACK: only use samples: [0, 1, 2, 3]
        self.current_maps = self.current_maps[0:4]
        self.topo_maps = self.topo_maps[0:4]

    def __remove_gradient(self, current_map: np.ndarray) -> np.ndarray:
        corrected_map = np.copy(current_map)
        C, H, W = current_map.shape
        # column indices from 0..W-1
        x = np.arange(W)
        for c in range(C):
            # 1. compute column-wise mean for channel c
            # shape: (W,)
            column_means = np.mean(current_map[c, :, :], axis=0)
            # 2. fit a line (degree=1 polynomial) to these means
            # polyfit returns [slope, intercept] for a degree=1 polynomial
            slope, intercept = np.polyfit(x, column_means, deg=1)
            # evaluate the fitted line at each column index
            # shape: (W,)
            best_fit_line = slope * x + intercept
            # 3. subtract the fitted line from each pixel in the column
            # for column w, best_fit_line[w] is the "gradient" we want to remove
            for w in range(W):
                corrected_map[c, :, w] -= best_fit_line[w]
        return corrected_map

    def _remove_gradients(self):
        """
        Remove column-wise gradients from each map in self.current_maps by:
        1. Computing column-wise mean of each channel.
        2. Fitting a best-fit line to these means.
        3. Subtracting that line (per column) from the original values.
        """
        for i, current_map in enumerate(self.current_maps):
            self.current_maps[i] = self.__remove_gradient(current_map)

    def _calculate_mean_std(self) -> None:
        """
        Calculate the mean and std of topo/curr maps.
        Saves results as internal vars.
        """
        c_stack = np.concatenate(
            [np.expand_dims(arr, axis=0) for arr in self.current_maps], axis=0
        )
        t_stack = np.concatenate(
            [np.expand_dims(arr, axis=0) for arr in self.topo_maps], axis=0
        )
        self.current_maps_mean = np.mean(c_stack, axis=(0, 1, 2))
        self.current_maps_std = np.std(c_stack, axis=(0, 1, 2))
        self.topo_maps_mean = np.mean(t_stack, axis=(0, 1, 2))
        self.topo_maps_std = np.std(t_stack, axis=(0, 1, 2))

    def _create_augmentation_pipeline(self, resize_to_og_height=True):
        # HACK: optionaly resize image to original height after taking random crop.
        # We do not resize images when training a ControlNet, hence the need for the conditional.
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
                A.Resize(
                    width=self.side_length,
                    height=self.side_length,
                    interpolation=cv2.INTER_AREA,
                ),
            ],
            additional_targets={
                "y": "mask",
                "X_og": "mask",
                "y_og": "mask",
                "y_unnormed": "mask",
                "y_mask": "mask",
            },
        )

    def __len__(self) -> int:
        """
        len(self) == self.steps_per_epoch
        """
        return self.steps_per_epoch

    def get_item_p_z_bar_x(self, index: int) -> Dict:
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

    def get_item_p_y_bar_x_cn(self, index: int) -> Dict:
        """
        Get items for ControlNet (image -> image translation)
        Predict a current map y_hat from given topology map X
        - Do NOT resize images after taking a random crop.
        """
        # TODO: is the text-encoder loaded?; how do we encode text?
        items = self.get_item_p_y_bar_x(index)
        y: torch.Tensor = items["y"]
        # modify conditional input value range: [-1, 1] -> [0, 1]
        y_sparse: torch.Tensor = (items["y_sparse"] + 1) / 2
        y = y.permute(2, 1, 0)
        y_sparse = y_sparse.permute(2, 1, 0)
        return dict(jpg=y, txt="", hint=y_sparse)

    def get_item_p_y_bar_x(self, index: int) -> Dict:
        """
        Get items for image -> image translation.
        Predict a current map y_hat from given topology map X.

        - Do NOT resize images after taking a random crop.
        """

        p_y_bar_x_augmentation_pipeline = self._create_augmentation_pipeline(
            resize_to_og_height=False
        )

        # NOTE: we should only consider samples: [0, 1, 2, 3];
        # 4th sample is collected under slightly different conditions
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

        # get un-normed topography map
        X: np.ndarray = self.topo_maps[sample_idx]
        # copy of original X for figure loging
        X_og = X.copy()

        # get mask based on masking ratio
        # mask w/ shape [H, W, C]
        y_mask = np.ones(tuple(X.shape))
        y_mask[:, :: self.masking_ratio + 1, :] = 0
        
        # # HACK ---------------
        # y_mask[:, 0::10, :] = 0
        # y_mask[:, 1::10, :] = 0
        # y_mask[:, 2::10, :] = 0
        # y_mask[:, 3::10, :] = 0
        # y_mask[:, 4::10, :] = 0
        # y_mask[:, 5::10, :] = 0
        # y_mask[:, 6::10, :] = 0
        # y_mask[:, 7::10, :] = 0
        # y_mask[:, 8::10, :] = 0
        # # ---------------------

        # get un-normed current map
        y: np.ndarray = self.current_maps[sample_idx]
        # copy of original y for figure logging
        y_og = y.copy()
        y_unnormed = y.copy()

        # augment samples
        # X recieves pixel-value normalization, all other data do not
        # do not resize after random crop; 64x64 -> 64x64
        augmented = p_y_bar_x_augmentation_pipeline(
            image=X, y=y, X_og=X_og, y_og=y_og, y_unnormed=y_unnormed, y_mask=y_mask
        )

        # convert all data -> tensor
        X: np.ndarray = augmented["image"]
        X = torch.tensor(X).permute(2, 0, 1).float()
        y: np.ndarray = augmented["y"]
        y = torch.tensor(y).permute(2, 0, 1).float()
        X_og = torch.tensor(augmented["X_og"]).float()
        y_og = torch.tensor(augmented["y_og"]).float()
        y_mask: torch.Tensor = torch.Tensor(augmented["y_mask"]).permute(2, 0, 1).bool()
        y_unnormed: np.ndarray = augmented["y_unnormed"]

        # normalize X, y using standard normal
        X = (X - self.topo_maps_mean[:, None, None]) / self.topo_maps_std[:, None, None]
        y = (y - self.current_maps_mean[:, None, None]) / self.current_maps_std[
            :, None, None
        ]

        # MARK: calculate values of z
        # z: sum(pixels < self.epsilon) / total num pixels
        z = (y_unnormed.flatten() < self.epsilon).sum() / (
            y_unnormed.shape[0] * y_unnormed.shape[1] * y_unnormed.shape[2]
        )

        # OPTIONAL: scale z by z_mult
        z = torch.tensor(z).float() * self.z_mult

        # z should always be in range: [0, 1.0 * Z_MULT]
        assert z >= 0.0 and z <= (1.0 * self.z_mult)

        return {
            "X": X,
            "y": y,
            "z": z,
            "y_mask": y_mask,
            "X_og": X_og,
            "y_og": y_og,
            "epsilon": self.epsilon,
        }

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

        fn_map = {
            Formulation.P_Z_BAR_X: self.get_item_p_z_bar_x,
            Formulation.P_Y_BAR_X: self.get_item_p_y_bar_x,
            Formulation.P_Y_BAR_X_CN: self.get_item_p_y_bar_x_cn,
        }
        if self.formulation not in fn_map:
            raise Exception(
                f"Error: invalid problem problem formulation: {self.formulation}"
            )
        f = fn_map[self.formulation]
        return f(index)
