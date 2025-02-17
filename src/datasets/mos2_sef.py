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


SRC_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/raw-data/1-23-25"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
ORIGINAL_IMAGE_SIZE = (224, 224)
CROPPED_IMG_SIDE_LENGTH = 128
IMG_SIZE_UM = 2.0
NORMALIZED_DATA_RANGE = (0.0, 1.0)


class Formulation(Enum):
    P_Z_BAR_X = 0
    P_Y_BAR_X = 1
    P_Y_BAR_X_CN = 2
    P_Y_BAR_Y_SPARSE_CN = 3
    P_Y_BAR_Y_SPARSE_BENCHMARK = 4
    P_Y_BAR_Y_SPARSE = 5
    P_Y_BAR_Y_SPARSE_BENCHMARK_CN = 6
    P_OLDER_BAR_Y_Y_AUG = 7

    @staticmethod
    def get_formulation_from_str(formulation_str: str) -> Enum:
        formulation_str = formulation_str.lower()
        if formulation_str == "p(z|x)":
            return Formulation.P_Z_BAR_X
        elif formulation_str == "p(y|x)":
            return Formulation.P_Y_BAR_X
        elif formulation_str == "p(y|x_cn)":
            return Formulation.P_Y_BAR_X_CN
        elif formulation_str == "p(y|y_sparse)":
            return Formulation.P_Y_BAR_Y_SPARSE
        elif formulation_str == "p(y|y_sparse_benchmark)":
            return Formulation.P_Y_BAR_Y_SPARSE_BENCHMARK
        elif formulation_str == "p(y|y_sparse_cn)":
            return Formulation.P_Y_BAR_Y_SPARSE_CN
        elif formulation_str == "p(y|y_sparse_benchmark_cn)":
            return Formulation.P_Y_BAR_Y_SPARSE_BENCHMARK_CN
        elif formulation_str == "p(older|y,y_aug)":
            return Formulation.P_OLDER_BAR_Y_Y_AUG
        else:
            raise KeyError


class MOS2SEFDataset(Dataset):
    """
    Dataset class for MoS2 samples collected on a Sapphire substrate.

    :Definitions:
    - X: topography map             | (H, W)
    - y: current map                | (H, W)
    - y_sparse: masked current map  | (H, W)
    """

    def __init__(
        self,
        split: str = "train",
        formulation: Formulation = Formulation.P_Y_BAR_X,
        side_length: int = CROPPED_IMG_SIDE_LENGTH,
        masking_ratio: int = 0,
        steps_per_epoch: int = 100,
        device: int = 0,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        :param split: "train" or "val"
        :param steps_per_epoch: data is sampled using random augmentations, therefore the # sample per epoch is arbitrary
        :param side_length: length of the side of the square crops taken from the original, full-sized image
        :param device: number of CUDA device, not currently used
        :param masking_ratio: 1-in-{masking_ratio} pixels masked
        :param original_image_size: size of the original images in the dataset: e.g., (256, 256)
        """

        super(MOS2SEFDataset, self).__init__()
        self.side_length: int = side_length
        self.steps_per_epoch: int = steps_per_epoch
        self.split: str = split
        self.formulation: Formulation = formulation
        self.device: int = device
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

        # use these vals to normalize all data -> [0, 1]
        self.current_maps_max = 0.0
        self.current_maps_min = 0.0
        self.topo_maps_mean = 0.0
        self.topo_maps_std = 0.0

        # NOTE: hard-coded global constants
        # original sample size is 2um
        self.img_size_um = IMG_SIZE_UM
        # all data (current + topo maps) normalized to -> [0, 1]
        self.normalized_data_range: Tuple[float, float] = NORMALIZED_DATA_RANGE

        # load all data from src files
        self._load_imgs()

        # remove L -> R gradients; remove back contact bias
        # self._remove_gradients()

        # find the mean/std of current and topo maps
        self._calculate_mean_std()

    def _load_imgs(self) -> None:
        """
        TODO: make less clunky and hard-coded.
        Load current-map + topo-map data from source dir.
        """

        current_map_regex = f"{SRC_DIR}/*Current*.npy"
        topo_map_regex = f"{SRC_DIR}/*Height*.npy"

        self._raw_current_fps = sorted(glob(current_map_regex))
        self._raw_topo_fps = sorted(glob(topo_map_regex))

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
            os.path.basename(fp)[:4] for fp in self._raw_current_fps
        ]
        _topo_fps_basenames = [os.path.basename(fp)[:4] for fp in self._raw_topo_fps]
        assert (
            _current_fps_basenames == _topo_fps_basenames
        ), f"Error: misalignment of current maps and topo maps during dataloading"

        # convert maps to type -> float64
        self.current_maps = [cm.astype(np.float64) for cm in self.current_maps]
        self.topo_maps = [tm.astype(np.float64) for tm in self.topo_maps]

    def __remove_gradient(self, current_map: np.ndarray) -> np.ndarray:
        """
        Find a line of best fit through the column-wise average current of a sample y.
        This method helps to remove the bias create by the back-contact; a global bias
        a model could not be expected to remove without additional information.
        """
        corrected_map = np.copy(current_map)
        H, W = current_map.shape
        # column indices from 0..W-1
        x = np.arange(W)
        # 1. shape: (W,)
        column_means = [np.mean(current_map[:, w]) for w in range(W)]
        # 2. fit a line (degree=1 polynomial) to these means
        # polyfit returns [slope, intercept] for a degree=1 polynomial
        slope, intercept = np.polyfit(x, column_means, deg=1)
        # evaluate the fitted line at each column index
        # shape: (W,)
        best_fit_line = slope * x + intercept
        # 3. subtract the fitted line from each pixel in the column
        # for column w, best_fit_line[w] is the "gradient" we want to remove
        for w in range(W):
            corrected_map[:, w] -= best_fit_line[w]
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
        self.current_maps_mean = np.mean(np.array(self.current_maps))
        self.current_maps_std = np.std(np.array(self.current_maps))
        self.current_maps_max = np.amax(np.array(self.current_maps))
        self.current_maps_min = np.amin(np.array(self.current_maps))
        self.topo_maps_mean = np.mean(np.array(self.topo_maps))
        self.topo_maps_std = np.std(np.array(self.topo_maps))
        self.topo_maps_max = np.amax(np.array(self.topo_maps))
        self.topo_maps_min = np.amin(np.array(self.topo_maps))

    def _create_augmentation_pipeline(self, resize_to_og_height=True):
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
                "sparse_mask": "mask",
            },
        )

    def __len__(self) -> int:
        """
        len(self) == self.steps_per_epoch
        """
        return self.steps_per_epoch

    def get_item_p_y_bar_y_sparse_cn(self, index: int) -> Dict:
        """
        Get items for ControlNet (image -> image translation)
        Predict a current map y_hat from given topology map X
        - Do NOT resize images after taking a random crop.
        """
        items = self.get_item_p_y_bar_y_sparse(index)
        y: torch.Tensor = items["y"]

        # [-1, 1] -> [0, 1]
        # y_sig = (y - y.min()) / (y.max() - y.min())
        # -> [0, 1]; y is already normalized
        y_sig = y.clone()
        y_mask: torch.Tensor = items["mask"]
        y_sparse = (y_sig * y_mask).float()

        # HACK: unconditional training
        y_sparse = y_sparse * 0

        # -> [H, W, C]
        # [H, W] -> [H, W, 1]
        y_img_like = y_sig.clone()
        y_img_like = y_img_like.unsqueeze(-1)
        # [H, W, 1] -> [H, W, 3]
        y_img_like = y_img_like.repeat(1, 1, 3)
        # [H, W] -> [H, W, 1]
        y_sparse_img_like = y_sparse.clone()
        y_sparse_img_like = y_sparse_img_like.unsqueeze(-1)
        # [H, W, 1] -> [H, W, 3]
        y_sparse_img_like = y_sparse_img_like.repeat(1, 1, 3)
        # [0, 1] -> [-1, 1]
        y_img_like = (y_img_like * 2) - 1

        return dict(jpg=y_img_like, txt="", hint=y_sparse_img_like)

    def get_item_p_y_bar_y_sparse(self, index: int) -> Dict:
        """
        Item getter method for p(y|y_sparse) formulation.
        Partially mask the original current map y; currently row-wise masking.
        """

        p_y_bar_x_augmentation_pipeline = self._create_augmentation_pipeline(
            resize_to_og_height=False
        )

        # NOTE: we only consider samples: [0, 1, 2, 3];
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
        
        # [H, W]; un-normalized topography map
        X: np.ndarray = self.topo_maps[sample_idx]

        # [H, W]; get sparse mask w/ shape
        sparse_mask = np.ones(tuple(X.shape))
        sparse_mask[:: self.masking_ratio + 1, :] = 0
        # TODO: add a better way to allow differnt sparse ratio selection

        # [H, W]; get un-normed current map
        y: np.ndarray = self.current_maps[sample_idx]

        # ---- augment samples ----
        augmented = p_y_bar_x_augmentation_pipeline(
            image=X, y=y, sparse_mask=sparse_mask
        )

        # H' < H | W' < W
        # [H', W']
        X: np.ndarray = augmented["image"]
        X = torch.tensor(X).float()
        # [H', W']
        y: np.ndarray = augmented["y"]
        y = torch.tensor(y).float()
        # [H', W']
        mask: torch.Tensor = torch.Tensor(augmented["sparse_mask"]).bool()

        # normalize X, y -> [0, 1]
        X = (X - self.topo_maps_min) / (self.topo_maps_max - self.topo_maps_min)

        y_unnorm = y.clone()
        y = (y - self.current_maps_min) / (
            self.current_maps_max - self.current_maps_min
        )
        
        assert X.max() <= 1.0 and X.min() >= 0.0, f"Error normalizing X sample: {X.shape}"
        assert y.max() <= 1.0 and y.min() >= 0.0, f"Error normalizing y sample: {y.shape}"

        return {
            "X": X,
            "y": y,
            "y_unnorm": y_unnorm,
            "mask": mask,
        }

    def get_item_p_y_bar_y_sparse_deterministic(self, index: int) -> Dict:
        """
        Deterministic variant of the p(y|y_sparse) get item method.
        """

        # save current rng states
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()

        seed = 12345 + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        item = self.get_item_p_y_bar_y_sparse(index)

        # restore previous rng states
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)

        return item

    def get_item_p_y_bar_y_sparse_deterministic_cn(self, index: int) -> Dict:
        """
        ...
        """

        items = self.get_item_p_y_bar_y_sparse_deterministic(index)

        y: torch.Tensor = items["y"]

        # [-1, 1] -> [0, 1]
        # y_sig = (y - y.min()) / (y.max() - y.min())
        # -> [0, 1]; y is already normalized
        y_sig = y.clone()
        y_mask: torch.Tensor = items["y_mask"]
        y_sparse = (y_sig * y_mask).float()

        # -> [H, W, C]
        # [H, W] -> [H, W, 1]
        y_img_like = y_sig.clone()
        y_img_like = y_img_like.unsqueeze(-1)
        # [H, W, 1] -> [H, W, 3]
        y_img_like = y_img_like.repeat(1, 1, 3)
        # [H, W] -> [H, W, 1]
        y_sparse_img_like = y_sparse.clone()
        y_sparse_img_like = y_sparse_img_like.unsqueeze(-1)
        # [H, W, 1] -> [H, W, 3]
        y_sparse_img_like = y_sparse_img_like.repeat(1, 1, 3)
        # [0, 1] -> [-1, 1]
        y_img_like = (y_img_like * 2) - 1

        return dict(jpg=y_img_like, txt="", hint=y_sparse_img_like)
    
    def get_item_p_older_bar_y_y_aug(self, index: int) -> dict: 
        """
        Item getter method for p(y|y_sparse) formulation.
        Partially mask the original current map y; currently row-wise masking.
        """

        augmentation_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
                A.Resize(
                    width=self.side_length,
                    height=self.side_length,
                    interpolation=cv2.INTER_AREA,
                ),
            ],
            additional_targets={
                "y": "mask",
            },
        )

        # NOTE: we only consider samples: [0, 1, 2, 3];
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

        # [H, W]; get un-normed current map
        y: np.ndarray = self.current_maps[sample_idx]
        y_aug = y.copy()

        # ---- augment samples ----
        augmented = augmentation_pipeline(image=y_aug, y=y)

        # H' < H | W' < W
        # [H', W']
        y_aug: np.ndarray = augmented["image"]
        y_aug = torch.tensor(y_aug).float()
        y: np.ndarray = augmented["y"]
        y = torch.tensor(y).float()
        
        # normalize y -> [0, 1]
        y = (y - self.current_maps_min) / (self.current_maps_max - self.current_maps_min)
        y_aug = (y_aug - self.current_maps_min) / (self.current_maps_max - self.current_maps_min)
        y_aug = (y_aug - y_aug.flatten().min()) / (y_aug.flatten().max() - y_aug.flatten().min())
        
        assert y.max() <= 1.0 and y.min() >= 0.0, f"Error normalizing y sample: {y.shape}"
        assert y_aug.max() <= 1.0 and y_aug.min() >= 0.0, f"Error normalizing y sample: {y_aug.shape}"

        return {
            "y": y,
            "y_aug": y_aug,
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
            Formulation.P_Y_BAR_Y_SPARSE: self.get_item_p_y_bar_y_sparse,
            Formulation.P_Y_BAR_Y_SPARSE_CN: self.get_item_p_y_bar_y_sparse_cn,
            Formulation.P_Y_BAR_Y_SPARSE_BENCHMARK: self.get_item_p_y_bar_y_sparse_deterministic,
            Formulation.P_Y_BAR_Y_SPARSE_BENCHMARK_CN: self.get_item_p_y_bar_y_sparse_deterministic_cn,
            Formulation.P_OLDER_BAR_Y_Y_AUG: self.get_item_p_older_bar_y_y_aug,
        }
        if self.formulation not in fn_map:
            raise Exception(
                f"Error: invalid problem problem formulation: {self.formulation}"
            )
        f = fn_map[self.formulation]
        return f(index)


if __name__ == "__main__":
    _ = MOS2SEFDataset()
