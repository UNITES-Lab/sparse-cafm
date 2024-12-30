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
from PIL import Image


class ImageExtensions(Enum):
    JPG = ".jpg"
    PNG = ".png"


class DatasetPaths(Enum):
    TEST_PLACES_MEDIUM_256 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_medium_256"
    TEST_PLACES_MEDIUM_512 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_medium_512"
    TEST_PLACES_THICK_256 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_thick_256"
    TEST_PLACES_THICK_512 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_thick_512"
    TEST_PLACES_THIN_256 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_thin_256"
    TEST_PLACES_THIN_512 = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365/benchmark-set/random_thin_512"


class InpaintingEvaluationDataset(Dataset):

    def __init__(
        self,
        root_dir: str = DatasetPaths.TEST_PLACES_MEDIUM_256.value,
        img_suffix: str = ImageExtensions.PNG.value,
        pad_img_to_mod_by: Optional[int] = None,
        scale_img_by: Optional[int] = None,
    ):
        """
        ...
        
        Args:
            :param root_dir: path to dir of images to evaluate model on
            :param img_suffix: ext of images in `datadir`
            :param pad_img_to_mod_by: no fucking clue
            :param scale_img_by:
        """

        self.root_dir = root_dir
        assert os.path.isdir(self.root_dir)
        self.img_suffix: str = img_suffix

        # load image masks
        # TODO: pre-compute all mask varients; make mask type an option
        masks_regex = os.path.join(self.root_dir, f"*mask*{self.img_suffix}")
        self.mask_fps = glob(masks_regex)
        assert (
            len(self.mask_fps) > 0
        ), f"Error: could not load images from dir {root_dir}"

        self.img_fps = []
        for fp in self.mask_fps:
            base_name = fp.rsplit("_mask", 1)[0]
            img_filename = base_name + self.img_suffix
            self.img_fps.append(img_filename)
        assert len(self.img_fps) == len(
            self.mask_fps
        ), f"Error: {len(self.mask_fps)} masks found != {len(self.img_fps)} images found"

        self.pad_img_to_mod_by: Optional[int] = pad_img_to_mod_by
        self.scale_img_by: Optional[int] = scale_img_by

    def __len__(self) -> int:
        return len(self.mask_fps)

    def __getitem__(self, index: int) -> dict:
        """
        : Returns :
        ```
        {
            'image':                    ...,
            'mask':                     ...,
            'original_image_shape':     ...,
        }
        ```
        """
        # TODO: image loading could be a major bottlneck; consider pre-loading imgs/keeping a buffer
        image = Image.open(self.img_fps[index]).convert("RGB")
        mask = Image.open(self.mask_fps[index]).convert("L")
        # (H, W, C)
        image_arr = np.array(image)
        # [H, C]
        original_image_shape: Tuple = image_arr.shape[1:]
        # optionally pad image to modulo factor
        # need to make image shapes place nice with some models
        if self.pad_img_to_mod_by != None:
            # (H, W, C -> MOD)
            image_arr = InpaintingEvaluationDataset.pad_img_to_modulo(
                image_arr, self.pad_img_to_mod_by
            )
        # (1, H, W)
        mask_arr = np.array(mask)[None, ...]
        return {
            "original_image_shape": original_image_shape,
            "image": torch.Tensor(image_arr),
            "mask": torch.Tensor(mask_arr),
        }

    @staticmethod
    def pad_img_to_modulo(img: np.ndarray, mod: int):
        """ """
        _, height, width = img.shape
        # smallest multiple of mod >= height and width
        out_height = ((height + mod - 1) // mod) * mod
        out_width = ((width + mod - 1) // mod) * mod
        pad_height = out_height - height
        pad_width = out_width - width
        # pad image
        padded_img = np.pad(
            img,
            pad_width=((0, 0), (0, pad_height), (0, pad_width)),
            mode="symmetric",
        )
        return padded_img