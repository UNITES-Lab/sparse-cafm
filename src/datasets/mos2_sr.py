import os
import torch
import random
import numpy as np
import albumentations as A
import torch.nn.functional as F

from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List, Union
from glob import glob

MOS2_SYNTHETIC             = "data/synth-datasets"
MOS2_SAPPHIRE_DIR          = "data/raw-data/11-19-24/2. MoS2 on Sapphire"
MOS2_SILICON_DIR           = "data/raw-data/11-19-24/2. MoS2 on Sapphire"
MOS2_SEF_FULL_RES_SRC_DIR  = "data/raw-data/1-23-25"
MOS2_SEF_MANY_RES_SRC_DIR  = "/playpen/mufan/levi/tianlong-chen-lab/sparse-cafm/data/raw-data/2-6-25"
BTO_MANY_RES               = "data/raw-data/3-12-25"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
ORIGINAL_IMAGE_SIZE = (512, 512)
CROPPED_IMG_SIDE_LENGTH = 64
IMG_SIZE_UM = 2.0
NORMALIZED_DATA_RANGE = (0.0, 1.0)


class MOS2SRDataset(Dataset):
    """
    Dataset class for sparse-sampling of MoS2 samples collected on various substrates.

    :Definitions:
    - X: surface height map | (H, W)
    - y: current map        | (H, W)
    """

    def __init__(
        self,
        src_dir: str = MOS2_SEF_FULL_RES_SRC_DIR,
        split: str = "train",
        upsample_factor: int = 2,
        steps_per_epoch: int = 100,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        Parameters
        ---
        split : str
            Dataset split; one of {'train', 'val', 'test'}.
                - 'train': Uses synthetic downsampling for training samples.
                - 'val': Uses synthetic downsampling for validation samples.
                - 'test': Uses only real downsampled data; supported by `MOS2_SEF_MANY_RES_SRC_DIR` and `BTO_MANY_RES` datasets.
        steps_per_epoch : int
            Number of batches per epoch. Data is randomly augmented, so the number of samples per epoch is arbitrary.
        upsample_factor : int
            Upsampling factor; must be one of {1, 2, 4, 8}.
        original_image_size : tuple of int
            Size of the original images in the dataset, e.g., (512, 512).
        """

        super(MOS2SRDataset, self).__init__()
        self.steps_per_epoch: int = steps_per_epoch

        assert split.lower() in ["train", "val", "test"], f"Error: invalid split. Expected 'train' or 'val'"
        self.split: str = split.lower()

        assert upsample_factor in [1, 2, 4, 8], f"Error: expected upsample_factor in: [1, 2, 4, 8]"
        self.upsample_factor = upsample_factor

        # size of subsamples to crop from original (512, 512) data
        self.side_length = 128
        if self.upsample_factor == 2:
            # [64, 64] -> [128, 128]
            self.side_length == 64 * 2
        if self.upsample_factor == 4:
            # [64, 64] -> [256, 256]
            self.side_length = 64 * 4
        if self.upsample_factor == 8:
            # [48, 48] -> [384, 384]
            self.side_length = 48 * 8

        assert os.path.isdir(src_dir), f"Error: invalid src_dir: {src_dir}"
        self.src_dir = src_dir

        self.original_image_size: Tuple[int, int] = original_image_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()

        # (B, C, H, W)
        self.current_maps = None
        self.topo_maps = None

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

        # original sample size is 2um
        self.img_size_um = IMG_SIZE_UM

        # all data (current + topo maps) normalized to -> [0, 1]
        self.normalized_data_range: Tuple[float, float] = NORMALIZED_DATA_RANGE

        # load all data from src files
        if self.src_dir == MOS2_SEF_FULL_RES_SRC_DIR:
            self._load_imgs_mos2_sef()
        elif src_dir == MOS2_SILICON_DIR or src_dir == MOS2_SAPPHIRE_DIR:
            self._load_imgs_sil_saf()
        elif src_dir == BTO_MANY_RES:
            self._load_bto_many_res()
        elif src_dir == MOS2_SYNTHETIC:
            self._load_imgs_mos2_synth()
        else:
            raise Exception(f"Error: unsupported dataset: {src_dir}")

        # TODO: experiment with this
        # remove L -> R gradients; remove back contact bias
        # self._remove_gradients()

        # find the mean/std of current and topo maps
        self._calculate_mean_std()

    def _load_bto_many_res(self) -> None:
        """
        BTO dataset only contains surface morphology maps.
        - 4x scans @{512, 256, 128, 64}
        """

        raise Exception("BTO dataset is not supported with this dataloader.")

    def _load_imgs_mos2_synth(self) -> None: 
        
        # current_map_regex = f"{self.src_dir}/current/{self.split}/current-maps/*.npy"
        # topo_map_regex = f"{self.src_dir}/topology/{self.split}/topo-maps/*.npy"

        # HACK: only load in train samples
        current_map_regex = f"{self.src_dir}/current/train/current-maps/*.npy"
        topo_map_regex = f"{self.src_dir}/topology/train/topo-maps/*.npy"

        NUM_SAMPLES = min(len(glob(current_map_regex)), len(glob(topo_map_regex)))
        self._raw_current_fps = sorted(glob(current_map_regex))[:NUM_SAMPLES]
        self._raw_topo_fps = sorted(glob(topo_map_regex))[:NUM_SAMPLES]

        assert (len(self._raw_current_fps) > 0), f"Error: could not load images using regex: {current_map_regex}"
        assert (len(self._raw_topo_fps) > 0), f"Error: could not load images using regex: {current_map_regex}"

        # [H, W, C]
        self.current_maps: List[np.ndarray] = [np.load(fp) for fp in self._raw_current_fps]
        self.topo_maps: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_fps]

        # validate current, topo map paris are aligned
        _current_fps_basenames = [os.path.basename(fp) for fp in self._raw_current_fps]
        _topo_fps_basenames = [os.path.basename(fp) for fp in self._raw_topo_fps]
        assert (_current_fps_basenames == _topo_fps_basenames), f"Error: misalignment of current maps and topo maps during dataloading"

        # convert maps to type -> float64
        self.current_maps = [cm.astype(np.float64) for cm in self.current_maps]
        self.topo_maps = [tm.astype(np.float64) for tm in self.topo_maps]
        
        # [H, W, C] -> [H, W] by averaging across the channel dimension
        self.current_maps = [np.mean(cm, axis=-1) for cm in self.current_maps]
        self.topo_maps = [np.mean(tm, axis=-1) for tm in self.topo_maps]

    def _load_imgs_mos2_sef(self) -> None:
        """
        Load current-map + topo-map data from source dir.
        """

        current_map_regex = f"{self.src_dir}/*Current*.npy"
        topo_map_regex = f"{self.src_dir}/*Height*.npy"

        self._raw_current_fps = sorted(glob(current_map_regex))
        self._raw_topo_fps = sorted(glob(topo_map_regex))

        assert (len(self._raw_current_fps) > 0), f"Error: could not load images using regex: {current_map_regex}"
        assert (len(self._raw_topo_fps) > 0), f"Error: could not load images using regex: {current_map_regex}"

        # [H, W, C]
        self.current_maps: List[np.ndarray] = [np.load(fp) for fp in self._raw_current_fps]
        self.topo_maps: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_fps]

        # validate current, topo map paris are aligned
        _current_fps_basenames = [os.path.basename(fp)[:4] for fp in self._raw_current_fps]
        _topo_fps_basenames = [os.path.basename(fp)[:4] for fp in self._raw_topo_fps]
        assert (_current_fps_basenames == _topo_fps_basenames), f"Error: misalignment of current maps and topo maps during dataloading"

        # convert maps to type -> float64
        self.current_maps = [cm.astype(np.float64) for cm in self.current_maps]
        self.topo_maps = [tm.astype(np.float64) for tm in self.topo_maps]


    def _load_imgs_sil_saf(self) -> None:
        """
        TODO: make less clunky and hard-coded.
        Load current-map + topo-map data from source dir.
        """

        current_map_regex = f"{self.src_dir}/*/*Current*.npy"
        topo_map_regex = f"{self.src_dir}/*/*Topo*.npy"

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

        # HACK: only use samples: [0, 1, 2, 3]
        self.current_maps = self.current_maps[0:4]
        self.topo_maps = self.topo_maps[0:4]

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

    def _create_augmentation_pipeline(self):
        return A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Rotate(limit=15, p=0.5),
                # A.ElasticTransform(),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
            ],
            additional_targets={
                "X":      "image",
                "X_mask": "mask",
                "y":      "mask",
            },
        )

    def __len__(self) -> int:
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
                    'X'       : torch.Tensor, topo-map w/ shape    [H, W]
                    'X_sparse': torch.Tensor, topo-map w/ shape    [H / upsample_factor, W / upsample_factor]
                    'X_unnorm': torch.Tensor, topo-map w/ shape    [H / upsample_factor, W / upsample_factor]
                    'y'       : torch.Tensor, current-map w/ shape [H, W]
                    'y_sparse': torch.Tensor, current-map w/ shape [H / upsample_factor, W / upsample_factor]
                    'y_unnorm': torch.Tensor, current-map w/ shape [H / upsample_factor, W / upsample_factor]
                }
        """
        
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
        
        # [512, 512]; un-normalized, full-sized topography map
        X: np.ndarray = self.topo_maps[sample_idx]
        
        # [512, 512]; un-normalized, full-sized current map
        y: np.ndarray = self.current_maps[sample_idx]

        # ---- select a [128, 128] subset from full-sample ----
        augmented: np.ndarray = self.augmentation_pipeline(image=y, X=X, X_mask=X, y=y)

        # [512, 512] -> [128, 128] + apply augs
        # HACK: always apply augmentations
        if self.split == "train":
            X: np.ndarray = augmented["X"]
            y: np.ndarray = augmented["image"]
        elif self.split == "val":
            X: np.ndarray = augmented["X_mask"]
            y: np.ndarray = augmented["y"]
        else:
            raise Exception("Something has gone very wrong")
        
        X: torch.Tensor = torch.Tensor(X).float()
        y: torch.Tensor = torch.Tensor(y).float()
        
        # [128, 128]
        X_unnorm = X.clone()
        y_unnorm = y.clone()

        # -> [0, 1]
        X = (X - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )

        # -> [0, 1]
        y = (y - self.current_maps_min) / (
            self.current_maps_max - self.current_maps_min
        )

        # ---- bicubic downsampling ----

        # -> [1, 1, 128, 128]
        X_unsqueezed = X.unsqueeze(0).unsqueeze(0)
        # -> [H', W']
        X_sparse = F.interpolate(
            X_unsqueezed, 
            scale_factor=1/self.upsample_factor, 
            mode='bicubic', 
            align_corners=False
            )
        X_sparse = X_sparse.squeeze(0).squeeze(0)
        
        # -> [1, 1, 128, 128]
        y_unsqueezed = y.unsqueeze(0).unsqueeze(0)
        # -> [H', W']
        y_sparse = F.interpolate(
            y_unsqueezed, 
            scale_factor=1/self.upsample_factor, 
            mode='bicubic', 
            align_corners=False
            )
        y_sparse = y_sparse.squeeze(0).squeeze(0)
        
        assert (X.max() <= 1.0 and X.min() >= 0.0), f"Error normalizing X sample: {X.shape}"
        assert (y.max() <= 1.0 and y.min() >= 0.0), f"Error normalizing y sample: {y.shape}"
        
        return {
            "X": X,
            "X_sparse": X_sparse,
            "X_unnorm": X_unnorm,
            "y": y,
            "y_sparse": y_sparse,
            "y_unnorm": y_unnorm,
        }
    

class BTOSRDataset(Dataset):
    """
    Dataset class used for sparse-sampling of BTO surface morphology maps.

    :Definitions:
    - X: surface height map | (H, W)
    """

    def __init__(
        self,
        src_dir: str = BTO_MANY_RES,
        split: str = "train",
        upsample_factor: int = 2,
        steps_per_epoch: int = 100,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        Parameters
        ---
        split : str
            Dataset split; one of {'train', 'val', 'test'}.
                - 'train': Uses synthetic downsampling for training samples.
                - 'val': Uses synthetic downsampling for validation samples.
                - 'test': Uses only real downsampled data.
        steps_per_epoch : int
            Number of batches per epoch. Data is randomly augmented, so the number of samples per epoch is arbitrary.
        upsample_factor : int
            Upsampling factor; must be one of {2, 4, 8}.
        original_image_size : tuple of int
            Size of the original images in the dataset, e.g., (512, 512).
        """

        super(BTOSRDataset, self).__init__()
        self.steps_per_epoch: int = steps_per_epoch

        assert split.lower() in ["train", "val", "test"], f"Error: invalid split. Expected 'train' or 'val'"
        self.split: str = split.lower()

        assert upsample_factor in [2, 4, 8], f"Error: expected upsample_factor in: [2, 4, 8]"
        self.upsample_factor = upsample_factor

        # size of subsamples to crop from original (512, 512) data
        self.side_length = 128
        if self.upsample_factor == 2:
            # [64, 64] -> [128, 128]
            self.side_length == 64 * 2
        if self.upsample_factor == 4:
            # [64, 64] -> [256, 256]
            self.side_length = 64 * 4
        if self.upsample_factor == 8:
            # [48, 48] -> [384, 384]
            self.side_length = 48 * 8

        assert os.path.isdir(src_dir), f"Error: invalid src_dir: {src_dir}"
        self.src_dir = src_dir

        self.original_image_size: Tuple[int, int] = original_image_size

        # dedicated train/val augmentation pipelines
        self.train_augmentation_pipeline = self._create_train_augmentation_pipeline()
        self.val_augmentation_pipeline   = self._create_val_augmentation_pipeline()

        # (B, H, W)
        self.topo_maps = None

        # paths to un-normalized, high-precision current maps
        self._raw_topo_fps: Optional[List[str]] = None

        # use these vals to normalize all data -> [0, 1]
        self.topo_maps_mean = 0.0
        self.topo_maps_std = 0.0

        # original sample size is 2umx2um (512x512)
        self.img_size_um = IMG_SIZE_UM

        # all data (current + topo maps) normalized to -> [0, 1]
        self.normalized_data_range: Tuple[float, float] = NORMALIZED_DATA_RANGE

        self._load_bto_many_res()

        # find the mean/std of current and topo maps
        self._calculate_mean_std()

    def _load_bto_many_res(self) -> None:
        """
        BTO dataset only contains surface morphology maps.
        - 4x scans @{512, 256, 128, 64}
        """

        topo_map_regex_64  = f"{self.src_dir}/*64*.npy"
        topo_map_regex_128 = f"{self.src_dir}/*128*.npy"
        topo_map_regex_256 = f"{self.src_dir}/*256*.npy"
        topo_map_regex_512 = f"{self.src_dir}/*512*.npy"

        self._raw_topo_64_fps  = sorted(glob(topo_map_regex_64))
        self._raw_topo_128_fps = sorted(glob(topo_map_regex_128))
        self._raw_topo_256_fps = sorted(glob(topo_map_regex_256))
        self._raw_topo_512_fps = sorted(glob(topo_map_regex_512))

        assert (len(self._raw_topo_64_fps) > 0), f"Error: could not load images using regex: {topo_map_regex_64}"

        # [H, W]
        self.topo_maps_64 : List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_64_fps]
        self.topo_maps_128: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_128_fps]
        self.topo_maps_256: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_256_fps]
        self.topo_maps_512: List[np.ndarray] = [np.load(fp) for fp in self._raw_topo_512_fps]

        # convert maps to type -> float64
        self.topo_maps_64  = [tm.astype(np.float64) for tm in self.topo_maps_64]
        self.topo_maps_128 = [tm.astype(np.float64) for tm in self.topo_maps_128]
        self.topo_maps_256 = [tm.astype(np.float64) for tm in self.topo_maps_256]
        self.topo_maps_512 = [tm.astype(np.float64) for tm in self.topo_maps_512]

    def _calculate_mean_std(self) -> None:
        """
        Calculate the mean and std of topo/curr maps.
        Saves results as internal vars.
        """

        self.topo_maps_mean = np.mean(np.array(self.topo_maps_512))
        self.topo_maps_std  = np.std(np.array(self.topo_maps_512))
        self.topo_maps_max  = np.amax(np.array(self.topo_maps_512))
        self.topo_maps_min  = np.amin(np.array(self.topo_maps_512))

    def _create_train_augmentation_pipeline(self):
        return A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Rotate(limit=15, p=0.5),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
            ],
            additional_targets={
                "X":      "image",
                "X_mask": "mask",
            },
        )
    
    def _create_val_augmentation_pipeline(self):
        return A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Rotate(limit=15, p=0.5),
                A.RandomCrop(width=self.side_length, height=self.side_length, p=1.0),
            ],
            additional_targets={
                "X":      "image",
                "X_mask": "mask",
            },
        )

    def __len__(self) -> int:
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
                    'X_64'    : torch.Tensor, topo-map w/ shape    [H, W]
                    'X_128'   : torch.Tensor, topo-map w/ shape    [H, W]
                    'X_256'   : torch.Tensor, topo-map w/ shape    [H, W]
                    'X_512'   : torch.Tensor, topo-map w/ shape    [H, W]
                    'X_sparse': torch.Tensor, topo-map w/ shape    [H / upsample_factor, W / upsample_factor]
                    'X_unnorm': torch.Tensor, topo-map w/ shape    [H / upsample_factor, W / upsample_factor]
                }
        """
        
        # NOTE: we only consider samples: [0, 1, 2, 3];
        # choose a random sample idx
        if self.split == TRAIN_SPLIT:
            # randint is inclusive: [a, b]
            # select a random sample from self.data[:-1]
            sample_idx = random.randint(0, len(self.topo_maps_512) - 2)
        elif self.split == VAL_SPLIT:
            # select the final data sample: self.data[-1]
            sample_idx = len(self.topo_maps_512) - 1
        elif self.split == TEST_SPLIT:
            sample_idx = len(self.topo_maps_512) - 1
        else:
            raise Exception(f"Invalid split: {self.split}")
        
        # [512, 512]; un-normalized, full-sized topography map
        X_512: np.ndarray = self.topo_maps_512[sample_idx]
        X_256: np.ndarray = self.topo_maps_256[sample_idx]
        X_128: np.ndarray = self.topo_maps_128[sample_idx]
        X_64 : np.ndarray = self.topo_maps_64[sample_idx]

        X: np.ndarray = X_512.copy()

        # ---- select a [128, 128] subset from full-sample ----
        # TODO: create a validation set augmentation pipeline
        if self.split == "train":
            augmented: np.ndarray = self.train_augmentation_pipeline(image=X, X=X, X_mask=X)
        elif self.split == "val":
            augmented: np.ndarray = self.val_augmentation_pipeline(image=X, X=X, X_mask=X)
        else:
            raise Exception()

        # [512, 512] -> [128, 128] + apply augs
        if self.split   == TRAIN_SPLIT:
            X: np.ndarray = augmented["X"]
        elif self.split == VAL_SPLIT:
            X: np.ndarray = augmented["X_mask"]
        elif self.split == TEST_SPLIT:
            # don't apply augs to test set
            pass
        else:
            raise Exception("Something has gone very wrong")
        
        X    : torch.Tensor = torch.Tensor(X).float()
        X_512: torch.Tensor = torch.Tensor(X_512).float()
        X_256: torch.Tensor = torch.Tensor(X_256).float()
        X_128: torch.Tensor = torch.Tensor(X_128).float()
        X_64 : torch.Tensor = torch.Tensor(X_64).float()
        
        X_unnorm = X_512.clone()

        # -> [0, 1]
        X = (X - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )
        X_512 = (X_512 - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )
        X_256 = (X_256 - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )
        X_128 = (X_128 - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )
        X_64  = (X_64 - self.topo_maps_min) / (
            self.topo_maps_max - self.topo_maps_min
        )

        # ---- bicubic downsampling ----

        # -> [1, 1, 128, 128]
        X_unsqueezed = X.unsqueeze(0).unsqueeze(0)
        # -> [H', W']
        X_sparse = F.interpolate(
            X_unsqueezed, 
            scale_factor=1/self.upsample_factor, 
            mode='bicubic', 
            align_corners=False
            )
        X_sparse = X_sparse.squeeze(0).squeeze(0)
        
        assert (X.max()     <= 1.0 and X.min()     >= 0.0), f"Error normalizing X sample: {X.shape}"
        assert (X_512.max() <= 1.0 and X_512.min() >= 0.0), f"Error normalizing X sample: {X_512.shape}"
        # assert (X_256.max() <= 1.0 and X_256.min() >= 0.0), f"Error normalizing X sample: {X_256.shape}"
        # assert (X_128.max() <= 1.0 and X_128.min() >= 0.0), f"Error normalizing X sample: {X_128.shape}"
        # assert (X_64.max()  <= 1.0 and X_64.min()  >= 0.0), f"Error normalizing X sample: {X_64.shape}"
        
        return {
            "X"       : X,
            "X_sparse": X_sparse,
            "X_512"   : X_512,
            "X_256"   : X_256,
            "X_128"   : X_128,
            "X_64"    : X_64,
            "X_unnorm": X_unnorm,
        }


class UnifiedMOS2SRDataset(Dataset):
    """
    A horrible abomination that contains all datasets in one.
    """

    def  __init__(
        self,
        split: str = "train",
        upsample_factor: int = 2,
        steps_per_epoch: int = 100,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    ):
        """
        :param split: "train" or "val"
        :param steps_per_epoch: data is sampled using random augmentations, therefore the # sample per epoch is arbitrary
        :param upsample_factor: 1, 2, 4 or 8x
        :param original_image_size: size of the original images in the dataset: e.g., (512, 512)
        """

        super(UnifiedMOS2SRDataset, self).__init__()
        
        self.mos2_sef_dataset = MOS2SRDataset(
            src_dir=MOS2_SEF_SRC_DIR,
            split=split,
            upsample_factor=upsample_factor,
            steps_per_epoch=steps_per_epoch,
            original_image_size=original_image_size,
        )

        self.sapphire_dataset = MOS2SRDataset(
            src_dir=MOS2_SAPPHIRE_DIR,
            split=split,
            upsample_factor=upsample_factor,
            steps_per_epoch=steps_per_epoch,
            original_image_size=original_image_size,
        )
        
        self.silicon_datset = MOS2SRDataset(
            src_dir=MOS2_SILICON_DIR,
            split=split,
            upsample_factor=upsample_factor,
            steps_per_epoch=steps_per_epoch,
            original_image_size=original_image_size,
        )

    def __len__(self): 
        return len(self.mos2_sef_dataset)

    def __getitem__(self, index: int) -> dict:
        """
        [HACK]: currently returning items for unconditional ControlNet training.
        Return a random item from one of three datasets.
        """

        item = {}
        choice = random.random()
        
        # if choice   < .33: item = self.mos2_sef_dataset.__getitem__(index)
        # elif choice < .66: item = self.sapphire_dataset.__getitem__(index)
        # else:              item =   self.silicon_datset.__getitem__(index)
        
        # HACK: only use two datasets for transfer learning ablation
        if choice   < .50: item = self.mos2_sef_dataset.__getitem__(index)
        else             : item = self.sapphire_dataset.__getitem__(index)
        # else:              item =   self.silicon_datset.__getitem__(index)

        return item


if __name__ == "__main__":

    dataset = BTOSRDataset(
        src_dir=BTO_MANY_RES,
        split="train", 
        upsample_factor=2, 
    )
    dataset[0]
