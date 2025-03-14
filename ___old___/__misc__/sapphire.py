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
CROPPED_IMG_SIDE_LENGTH = 64
SRC_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/raw-data/11-19-24/2. MoS2 on Sapphire"
EXT = "tiff"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


class Formulation(Enum):
    P_Z_BAR_X = 0
    P_Y_BAR_X = 1
    P_Y_BAR_X_FIXED_GRID = 2
    P_Z_BAR_X_Y = 3
    P_Z_BAR_X_PLUS_Y_BAR_X = 4
    P_Z_BAR_Y = 5


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
        self.formulation: Formulation = formulation
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

    def _create_augmentation_pipeline(self, resize_to_og_height=True):
        # HACK: optionaly resize image to original height after taking random crop.
        # We do not resize images when training a ControlNet, hence the need for the conditional.
        if resize_to_og_height:
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomCrop(
                        width=self.side_length, height=self.side_length, p=1.0
                    ),
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
        else:
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomCrop(
                        width=self.side_length, height=self.side_length, p=1.0
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

        # HACK: use fixed-grid sampling
        if self.formulation == Formulation.P_Y_BAR_X_FIXED_GRID:
            if self.split == "train":
                return 64 * 4
            elif self.split == "val":
                return 64
            else:
                raise Exception

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

    def get_item_p_y_bar_x(self, index: int) -> Dict:
        """
        Get item method for ControlNet models.
        Predict a current map y_hat from given topology map X.

        - Do NOT resize images after taking a random crop.
        """

        p_y_bar_x_augmentation_pipeline = self._create_augmentation_pipeline(
            resize_to_og_height=False
        )

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
        # do not resize after random crop; 64x64 -> 64x64
        augmented = p_y_bar_x_augmentation_pipeline(
            image=X, mask=y_raw, y=y, X_og=X_og, y_og=y_og
        )

        # convert all data -> tensor
        X = torch.tensor(augmented["image"]).permute(2, 0, 1).float()
        y = torch.tensor(augmented["y"]).permute(2, 0, 1).float()
        X_og = torch.tensor(augmented["X_og"])
        y_og = torch.tensor(augmented["y_og"])
        # stays as a np.ndarray
        y_raw = augmented["mask"]

        # normalize X between [0, 1]
        X = torch.stack(
            [
                (channel - channel.min()) / (channel.max() - channel.min())
                for channel in X
            ]
        )
        # normalize y between [-1, 1]
        y = (y / 127.5) - 1.0

        # [C, H, W] -> [H, W, C]
        X = X.permute(1, 2, 0)
        y = y.permute(1, 2, 0)

        # MARK: calculate values of z
        # z: sum(pixels < self.epsilon) / total num pixels
        z = (y_raw.flatten() < self.epsilon).sum() / (y_raw.shape[0] * y_raw.shape[1])

        # TODO: what is the ideal way to normalize z?
        z = torch.tensor(z).float() * self.z_mult

        # z should always be in range: [0, 1.0 * Z_MULT]
        assert z >= 0.0 and z <= (1.0 * self.z_mult)

        return dict(jpg=y, txt=" ", hint=X)

    def get_item_p_y_bar_x_fixed_grid_sampling(self, index: int) -> Dict:
        pass

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
            Formulation.P_Y_BAR_X_FIXED_GRID: self.get_item_p_y_bar_x_fixed_grid_sampling,
        }
        if self.formulation not in fn_map:
            raise Exception(
                f"Error: invalid problem problem formulation: {self.formulation}"
            )
        f = fn_map[self.formulation]
        return f(index)


class SapphireDatasetFixedGridSampling(Dataset):
    """
    Used for fixed-grid sub-sampling of our dataset.

    :Definitions:
    - X: topography map (height, width, depth)
    - y: current map (height, width, current)
    - y_hat: predicted current map [via ControlNet] (height, width, depth)
    - z: scalar, current-under-threshold
    - epsilon: bottom 10th percentile threshold of current values
    """

    def __init__(
        self,
        split: str = "train",
        formulation: Formulation = Formulation.P_Y_BAR_X,
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

        super(SapphireDatasetFixedGridSampling, self).__init__()
        self.side_length: int = side_length
        self.steps_per_epoch: int = steps_per_epoch
        self.split: str = split
        self.formulation: Formulation = formulation
        self.device: int = device
        self.z_mult: int = z_mult
        self.original_image_size: Tuple[int, int] = original_image_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()

        # (B, C, H, W)
        self.topo_maps: tuple = None
        self.current_maps: tuple = None
        self.predicted_current_maps: tuple = None
        self._raw_current_maps: tuple = None
        
        # load all data from src files
        self._load_imgs()

        # calculate the value of ε: the bottom 10th percentile of raw current-map readings
        self.epsilon: Optional[float] = None
        self._calculate_epsilon()

        # (N, 64, C/D, H, W)
        self.X_patches: tuple = None
        self.y_patches: tuple = None
        # (N, 64, H, W)
        self.y_raw_patches: tuple = None
        self._patchify_images()

        # (N, 64, C, H, W)
        self.y_hat_patches: tuple = None
        self._load_y_hat_patches()

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

    def _load_y_hat_patches(self) -> None:

        src_dir = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/full-sized-c-asm-data/target/y_hat"
        if self.split == "train":
            src_dir = os.path.join(src_dir, "train")
        elif self.split == "val":
            src_dir = os.path.join(src_dir, "val")
        else:
            raise Exception("Invalid split")

        all_img_file_paths = glob(src_dir + "/**.png")
        all_img_file_paths = sorted(all_img_file_paths)

        y_hat_patches = []
        for fp in all_img_file_paths:
            arr = cv2.imread(fp)
            y_hat_patches.append(arr)

        # (N, H, W, C)
        self.y_hat_patches = np.array(y_hat_patches)

    def _patchify_images(self) -> None:
        """
        Split original (512, 512) data samples -> 64x (64, 64)
        """

        # want to create four arrays; aligned by dim
        # X_patches:                        (N, 64, D, H, W)
        # y_patches:                        (N, 64, C, H, W)
        # y_hat_patches:                    (N, 64, C, H, W)
        # y_raw_patches:                    (N, 64, H, W)

        self.topo_maps = torch.tensor(self.topo_maps).permute(0, 3, 1, 2)
        self.current_maps = torch.tensor(self.current_maps).permute(0, 3, 1, 2)
        
        # shape: (5, 512, 512)
        self._raw_current_maps = torch.tensor(self._raw_current_maps)

        # patchify X
        N, C, H, W = self.topo_maps.shape
        subimages = []
        for i in range(0, H, 64):
            for j in range(0, W, 64):
                # (N, C, 64, 64)
                chunk = (
                    self.topo_maps[:, :, i : i + 64, j : j + 64].detach().cpu().numpy()
                )
                subimages.append(chunk)

        # patchify y
        # (64, 5, 3, 64, 64)
        self.X_patches = np.array(subimages)

        N, C, H, W = self.current_maps.shape
        subimages = []
        for i in range(0, H, 64):
            for j in range(0, W, 64):
                # (N, C, 64, 64)
                chunk = (
                    self.topo_maps[:, :, i : i + 64, j : j + 64].detach().cpu().numpy()
                )
                subimages.append(chunk)

        # (64, 5, 3, 64, 64)
        # X -> controlnet -> y_hat
        self.y_patches = np.array(subimages)
        
        N, H, W = self._raw_current_maps.shape
        subimages = []
        for i in range(0, H, 64):
            for j in range(0, W, 64):
                # (N, 64, 64)
                chunk = (
                    self._raw_current_maps[:, i : i + 64, j : j + 64].detach().cpu().numpy()
                )
                subimages.append(chunk)

        # (64, 5, 64, 64)
        self.y_raw_patches = np.array(subimages)

    def _create_augmentation_pipeline(self, resize_to_og_height=True):
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(
                    width=self.original_image_size[0],
                    height=self.original_image_size[1],
                ),
            ],
        )

    def __len__(self):
        """
        Hard coded.

        HACK:
        - train:    64 * 4
        - val:      64
        """

        # HACK: use fixed-grid sampling
        if self.split == "train":
            return 64 * 4
        elif self.split == "val":
            return 64
        else:
            raise Exception

    def get_item_p_z_bar_x(self, index: int) -> Dict:
        """
        Formulation 1/4: predict z from topography map X.
        :returns:
            ```
                {
                    'X': torch.Tensor,      # topo-map w/ shape [C, H, W]
                    'y': torch.Tensor,      # target current-map w/ shape [C, H, W]
                    'z': torch.Tensor,      # scalar-valued target denoting 'current-under-threshold'
                }
        """

        # x/y patches shape: (64, 5, 3, 64, 64)
        X_patches = self.X_patches
        y_patches = self.y_patches
        
        # raw current readings shape: (64, 5, 64, 64)
        y_raw_patches = self.y_raw_patches

        # HACK: hard-coded train/val splits
        if self.split == TRAIN_SPLIT:
            X_patches: np.ndarray = X_patches[:, :5, :, :, :]
            y_patches: np.ndarray = y_patches[:, :5, :, :, :]
            y_raw_patches: np.ndarray = y_raw_patches[:, :5, :, :]
        elif self.split == VAL_SPLIT:
            X_patches = X_patches[:, -1, :, :, :]
            X_patches = np.expand_dims(X_patches, 1)
            y_patches = y_patches[:, -1, :, :, :]
            y_patches = np.expand_dims(y_patches, 1)
            y_raw_patches = y_raw_patches[:, -1, :, :]
            y_raw_patches = np.expand_dims(y_raw_patches, 1)
        else:
            raise Exception(f"Invalid split: {self.split}")
        
        whole_sample_idx = index // 64
        offset_idx = index - (64 * whole_sample_idx)

        # shape: (3, 64, 64)
        X: np.ndarray = X_patches[offset_idx, whole_sample_idx, :, :, :]
        # (3, 64, 64) -> (64, 64, 3)
        X = np.transpose(X, axes=(1, 2, 0))
        
        # (64, 64)
        # no need to augment this array, z will be unchanged by our current transforms
        y_raw: np.ndarray = y_raw_patches[offset_idx, whole_sample_idx, :, :]
        
        # (N * 64, 64, 64, 3) -> (64, 64, 3)
        y_hat: np.ndarray = self.y_hat_patches[index, :, :, :]
        
        # apply augmentations predicted current-map (y_hat)
        aug = self.augmentation_pipeline(image=y_hat)
        
        # (64, 64, 3) -> (224, 224, 3) -> (3, 224, 224)
        y_hat = torch.tensor(aug["image"]).float().permute(2, 1, 0)

        # apply augmentations to topo-map (X), current-map (y),
        aug = self.augmentation_pipeline(image=X)
        
        # (64, 64, 3) -> (224, 224, 3) -> (3, 224, 224)
        X = torch.tensor(aug["image"]).float().permute(2, 1, 0)

        # calculate value of z
        z = (y_raw.flatten() < self.epsilon).sum() / (y_raw.shape[0] * y_raw.shape[1])

        # HACK: normalize Z
        z = torch.tensor(z).float() * self.z_mult

        # z should always be in range: [0, 1.0 * Z_MULT]
        assert z >= 0.0 and z <= (1.0 * self.z_mult)

        return {
            "X": X,
            "z": z,
            "y_hat": y_hat,
            "epsilon": self.epsilon,
        }

    def get_item_p_y_bar_x(self, index: int) -> Dict:
        """
        TODO: clean up this method

        Get item method for ControlNet models.
        Predict a current map y_hat from given topology map X.

        - Do NOT resize images after taking a random crop.
        """

        p_y_bar_x_augmentation_pipeline = self._create_augmentation_pipeline(
            resize_to_og_height=False
        )

        sample_idx = -1
        offset = -1

        # HACK: hard-coded train/val splits
        if self.split == TRAIN_SPLIT:
            X_patches: np.ndarray = X_patches[:, :5, :, :, :]
            y_patches: np.ndarray = y_patches[:, :5, :, :, :]
        elif self.split == VAL_SPLIT:
            X_patches = X_patches[:, -1, :, :, :]
            y_patches = y_patches[:, -1, :, :, :]
        else:
            raise Exception(f"Invalid split: {self.split}")

        whole_sample_idx = index // 64
        offset_idx = index - (64 * whole_sample_idx)

        # get topography map with normalized depth dim
        X: np.ndarray = X_patches[offset_idx, whole_sample_idx, :, :, :]
        y: np.ndarray = y_patches[offset_idx, whole_sample_idx, :, :, :]
        y_hat: np.ndarray = self.y_hat_patches[index, :, :, :]

        aug = self.augmentation_pipeline(image=X, y=y, y_hat=y_hat)
        X = torch.tensor(aug["image"]).permute(1, 2, 0).float()
        y = torch.tensor(aug["y"]).permute(1, 2, 0).float()
        y_hat = torch.tensor(aug["y_hat"]).permute(1, 2, 0).float()

    def get_item_p_z_bar_y_hat(self, index: int) -> Dict:
        """
        Forumulation 2/4. Predict z from y_hat.

        :param index: currently unused, necessiary for batch data-loading
        :returns:
            ```
                {
                    'X': torch.Tensor,      # topo-map w/ shape [C, H, W]
                    'y_hat': torch.Tensor,  # ControlNet predicted current-map w/ shape [C, H, W]
                    'z': torch.Tensor,      # scalar-valued target denoting 'current-under-threshold
                }
        """
        
        # x/y patches shape: (64, 5, 3, 64, 64)
        X_patches = self.X_patches
        y_patches = self.y_patches
        
        # raw current readings shape: (64, 5, 64, 64)
        y_raw_patches = self.y_raw_patches

        # HACK: hard-coded train/val splits
        if self.split == TRAIN_SPLIT:
            X_patches: np.ndarray = X_patches[:, :5, :, :, :]
            y_patches: np.ndarray = y_patches[:, :5, :, :, :]
            y_raw_patches: np.ndarray = y_raw_patches[:, :5, :, :]
        elif self.split == VAL_SPLIT:
            X_patches = X_patches[:, -1, :, :, :]
            X_patches = np.expand_dims(X_patches, 1)
            y_patches = y_patches[:, -1, :, :, :]
            y_patches = np.expand_dims(y_patches, 1)
            y_raw_patches = y_raw_patches[:, -1, :, :]
            y_raw_patches = np.expand_dims(y_raw_patches, 1)
        else:
            raise Exception(f"Invalid split: {self.split}")
        
        whole_sample_idx = index // 64
        offset_idx = index - (64 * whole_sample_idx)

        # shape: (3, 64, 64)
        X: np.ndarray = X_patches[offset_idx, whole_sample_idx, :, :, :]
        # (3, 64, 64) -> (64, 64, 3)
        X = np.transpose(X, axes=(1, 2, 0))
        
        # (N * 64, 64, 64, 3) -> (64, 64, 3)
        y_hat: np.ndarray = self.y_hat_patches[index, :, :, :]
        
        # (64, 64)
        # no need to augment this array, z will be unchanged by our current transforms
        y_raw: np.ndarray = y_raw_patches[offset_idx, whole_sample_idx, :, :]
        
        # apply augmentations predicted current-map (y_hat)
        aug = self.augmentation_pipeline(image=y_hat)
        
        # (64, 64, 3) -> (224, 224, 3) -> (3, 224, 224)
        y_hat = torch.tensor(aug["image"]).float().permute(2, 1, 0)
        
        # apply augmentations to topo-map (X), current-map (y),
        aug = self.augmentation_pipeline(image=X)
        
        # (64, 64, 3) -> (224, 224, 3) -> (3, 224, 224)
        X = torch.tensor(aug["image"]).float().permute(2, 1, 0)

        # calculate value of z
        z = (y_raw.flatten() < self.epsilon).sum() / (y_raw.shape[0] * y_raw.shape[1])

        # HACK: normalize Z
        z = torch.tensor(z).float() * self.z_mult

        # z should always be in range: [0, 1.0 * Z_MULT]
        assert z >= 0.0 and z <= (1.0 * self.z_mult)

        return {
            "X": X,
            "y_hat": y_hat,
            "z": z,
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
            Formulation.P_Z_BAR_Y: self.get_item_p_z_bar_y_hat,
        }
        if self.formulation not in fn_map:
            raise Exception(
                f"Error: invalid problem problem formulation: {self.formulation}"
            )
        f = fn_map[self.formulation]
        return f(index)
