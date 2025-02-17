from glob import glob
import cv2
import pprint
import torch
import random
import numpy as np
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from typing import Tuple, Dict
from torch.utils.data import Dataset
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation
from src.util.celano_lab_scripts import process_image

NUM_CHAR_FEATURES = 9
CROPPED_IMAGE_SIDE_LENGTH = 128
ORIGINAL_IMAGE_SIZE = (512, 512)


class MOS2SefOLDERSurrogateDataset(Dataset):
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
        normalize_on_init: bool = False,
    ):
        self.split = split
        self.formulation = formulation
        self.side_length = side_length
        self.masking_ratio = masking_ratio
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.original_image_size = original_image_size
        
        self.dataset = MOS2SEFDataset(
            split=split,
            formulation=formulation,
            side_length=side_length,
            masking_ratio=masking_ratio,
            steps_per_epoch=steps_per_epoch,
            device=device,
            original_image_size=original_image_size,
        )
        
        # dictionary of {"mean": float, "std": float} values
        self.normalization_dict: Dict[str, Dict] = {}
        
        # optional: run a short benchmark to determine normalization mean/std
        if normalize_on_init: self.normalize()
    
    @staticmethod
    def scale_image(image: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scales the input image by `scale` (using bilinear interpolation)
        and then crops a central patch of the original size.
        
        Args:
            image (torch.Tensor): A 2D tensor of shape (H, W).
            scale (float): Scaling factor.
            
        Returns:
            torch.Tensor: Processed image of shape (H, W).
        """

        H, W = image.shape
        new_H, new_W = int(H * scale), int(W * scale)
        image_scaled = F.interpolate(image.unsqueeze(0).unsqueeze(0).float(),
                                    size=(new_H, new_W),
                                    mode='bilinear',
                                    align_corners=False)
        image_scaled = image_scaled.squeeze(0).squeeze(0)
        start_H = (new_H - H) // 2
        start_W = (new_W - W) // 2
        cropped_image = image_scaled[start_H:start_H+H, start_W:start_W+W]
        return cropped_image

    def __len__(self) -> int: return len(self.dataset)

    def normalize(self) -> None:
        """
        Run a short test proceedure to calculate the mean and std of train/val samples;
        set global values for mean/std so that all samples are normalized roughly to the std normal.
        We make the apriori assumption that train/val samples belong to roughly the same distribution.
        """
        
        NUM_BENCHMARK_STEPS = 1000
        train_dataset = MOS2SEFDataset(
            split="train",
            formulation=self.formulation,
            side_length=self.side_length,
            masking_ratio=self.masking_ratio,
            steps_per_epoch=NUM_BENCHMARK_STEPS,
            device=self.device,
            original_image_size=self.original_image_size,
        )
        val_dataset = MOS2SEFDataset(
            split="val",
            formulation=self.formulation,
            side_length=self.side_length,
            masking_ratio=self.masking_ratio,
            steps_per_epoch=NUM_BENCHMARK_STEPS,
            device=self.device,
            original_image_size=self.original_image_size,
        )
        
        samples = {}
        
        # samples from train/val datasets
        for idx in tqdm(range(NUM_BENCHMARK_STEPS), total=NUM_BENCHMARK_STEPS, desc="Calculating global mean/stds.."):
            
            train_item = train_dataset.__getitem__(idx)
            val_item = val_dataset.__getitem__(idx)
            train_y = train_item["y"]; val_y = val_item["y"]
            
            # characterize train/val current-maps
            train_char = process_image(train_y, self.dataset.img_size_um)
            val_char = process_image(val_y, self.dataset.img_size_um)
            
            for k, v in train_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)
            
            for k, v in val_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)

        for k, v in samples.items():
            self.normalization_dict[k] = {
                "mean": np.mean(v),
                "std": np.std(v),
            }


    def __getitem__(self, index: int) -> Dict:
        """
        Provide a current-map y and a "target" Tensor.
        
        Returns
        ---
        {
            "y": torch.Tensor: [H, W]
            "target": torch.Tensor: [9]
                - All nine Celano-Lab characterisitics normalized to standard normal.
        }
        """
        
        batch: dict  = self.dataset[index]
        
        # [H, W]
        y: torch.Tensor = batch["y"]
        y_unnorm: torch.Tensor = batch["y_unnorm"]
        
        # get the celano-lab characterization of a raw current-map sample
        y_char = process_image(y_unnorm, self.dataset.img_size_um)
        
        # ---- bootstrap y_char 10x ----
        # NUM_BOOTSTRAPS = 10

        # for i in range(NUM_BOOTSTRAPS - 1):
            
        #     y_aug = y.clone()
        #     if random.random() < 0.5:
        #         y_aug = torch.flip(y_aug, dims=[1])
        #     if random.random() < 0.5:
        #         y_aug= torch.flip(y_aug, dims=[0])
        #     if y.shape[0] == y.shape[1] and random.random() < 0.5:
        #         y_aug = torch.rot90(y_aug, k=1, dims=(0, 1))
        #     if random.random() < 0.5:
        #         factor = random.uniform(0.9, 1.1)
        #         y_aug = y_aug * factor
        #     if random.random() < 0.5:
        #         noise_std = 0.05 * (y_aug.max() - y_aug.min())
        #         noise = torch.randn_like(y_aug) * noise_std
        #         y_aug = y_aug + noise
        #     if random.random() < 0.5:
        #         # scale randomly 1x-1.3x
        #         y_aug = MOS2SefOLDERSurrogateDataset.scale_image(y_aug, 1 + (random.random() * 0.3))

        #     y_char_bootstrapped = process_image(y_aug, self.dataset.img_size_um)
            
        #     for k, v in y_char.items():
        #         y_char[k] = (y_char[k] + y_char_bootstrapped[k])
        
        # for k, v in y_char.items():
        #     y_char[k] = (y_char[k] + y_char_bootstrapped[k]) / NUM_BOOTSTRAPS
        
        # ---- normalize all vals -> ~std-normal ----
        for k in y_char:
            val = y_char[k]
            mean = self.normalization_dict[k]['mean']
            std = self.normalization_dict[k]['std']
            y_char[k] = (val - mean) / std
        
        # NOTE: remove high variance features
        y_char.pop("num_curved_lines")
        y_char.pop("num_extended_shapes")
        y_char.pop("total_area_extended_shapes")
        
        # for peace of mind; manually select features for target array
        target_arr = [None] * NUM_CHAR_FEATURES
        target_arr[0] = y_char['coverage_percentage']
        target_arr[1] = y_char['total_len_detected_curves']
        target_arr[2] = y_char['total_area_circular_shapes']
        target_arr[3] = y_char['total_defect_area']
        target_arr[4] = y_char['num_circular_shapes']
        target_arr[5] = y_char['average_surface_current']
        target = torch.Tensor(target_arr).float()
        
        item = {}
        item['y'] = y
        item['y_char'] = y_char
        item['target'] = target

        return item
    
class SyntheticMOS2SefOLDERSurrogateDataset(Dataset):
    """
    Synthetic MOS2-SEF dataset generated via ControlNet.
    Intended for pre-training the OLDER-Surrogate model.
    """

    ROOT_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/mos2-cafm-controlnet-synthetic-dataset"

    def __init__(self,
        split: str = "train",
        formulation: Formulation = Formulation.P_Y_BAR_Y_SPARSE,
        side_length: int = CROPPED_IMAGE_SIDE_LENGTH,
        masking_ratio: int = 0,
        steps_per_epoch: int = 100,
        device: int = 0,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
        normalize_on_init: bool = True,
        ):
        super().__init__()

        assert split in ["train", "val"], f"Error: expected `split` value in [train, val], got: {split}"
        self.split = split

        self.dataset = MOS2SEFDataset(
            split=split,
            formulation=formulation,
            side_length=side_length,
            masking_ratio=masking_ratio,
            steps_per_epoch=steps_per_epoch,
            device=device,
            original_image_size=original_image_size,
        )

        self.train_img_buffer = []
        self.train_current_map_buffer = []
        self.val_img_buffer = []
        self.val_current_map_buffer = []
        self.__load__()
        
        # dictionary of {"mean": float, "std": float} values
        self.normalization_dict: Dict[str, Dict] = {}

        # optional: run a short benchmark to determine normalization mean/std
        if normalize_on_init: self.normalize()

    def normalize(self) -> None:
        """
        Run a short test proceedure to calculate the mean and std of train/val samples;
        set global values for mean/std so that all samples are normalized roughly to the std normal.
        We make the apriori assumption that train/val samples belong to roughly the same distribution.
        """
        
        NUM_BENCHMARK_STEPS = 100
        
        samples = {}
        
        # samples from train/val datasets
        for idx in tqdm(range(NUM_BENCHMARK_STEPS), total=NUM_BENCHMARK_STEPS, desc="Calculating global mean/stds.."):
            
            # HACK: change train -> val buffer 
            train_item_fp = self.train_current_map_buffer[idx]
            val_item_fp = self.train_current_map_buffer[idx]

            train_y = np.load(train_item_fp)
            val_y = np.load(val_item_fp)

            # HACK: a super lazy way of nuking nan values that leak through
            train_y[np.isnan(train_y)] = np.nanmedian(train_y)
            train_y[np.isneginf(train_y)] = np.nanmedian(train_y)
            
            # HACK: a super lazy way of nuking nan values that leak through
            val_y[np.isnan(val_y)] = np.nanmedian(val_y)
            val_y[np.isneginf(val_y)] = np.nanmedian(val_y)

            train_y = torch.Tensor(train_y).float()
            val_y = torch.Tensor(val_y).float()
            
            # characterize train/val current-maps
            train_char = process_image(train_y, self.dataset.img_size_um)
            val_char = process_image(val_y, self.dataset.img_size_um)
            
            for k, v in train_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)
            
            for k, v in val_char.items():
                if k not in samples: samples[k] = [v]
                else: samples[k].append(v)

        for k, v in samples.items():
            self.normalization_dict[k] = {
                "mean": np.mean(v),
                "std": np.std(v),
            }
    
    def __load__(self) -> None:

        self.train_current_map_buffer = glob(f"{self.ROOT_DIR}/train/normalized-current-maps/*.npy")
        self.train_img_buffer = glob(f"{self.ROOT_DIR}/train/images/*.png")
        self.val_current_map_buffer = glob(f"{self.ROOT_DIR}/val/normalized-current-maps/*.npy")
        self.val_img_buffer = glob(f"{self.ROOT_DIR}/val/images/*.png")
        
        self.train_current_map_buffer.sort()
        self.train_img_buffer.sort()

        # ---- HACK: val data is not ready yet, ----

        # well assign an 80/20 split for now
        total_train_items = len(self.train_current_map_buffer)
        train_split = int(.80 * total_train_items)
        
        self.val_current_map_buffer = self.train_current_map_buffer[:train_split]
        self.val_img_buffer = self.train_img_buffer[:train_split]
        self.train_current_map_buffer = self.train_current_map_buffer[:train_split]
        self.train_img_buffer = self.train_img_buffer[:train_split]

        # ------------------------------------------

        self.val_current_map_buffer.sort()
        self.val_img_buffer.sort()

        assert len(self.train_current_map_buffer) == len(self.train_img_buffer)
        assert len(self.val_current_map_buffer) == len(self.val_img_buffer)

    def __len__(self): 
        return len(self.train_current_map_buffer) if self.split == "train" else len(self.val_current_map_buffer)
    
    def __getitem__(self, index: int) -> Dict:

        # select train/val buffer
        buffer = self.train_current_map_buffer if self.split == "train" else self.val_current_map_buffer
        
        y_unnormed = np.load(buffer[index])
        # HACK: a super lazy way of nuking nan values that leak through
        y_unnormed[np.isnan(y_unnormed)] = np.nanmedian(y_unnormed)
        y_unnormed[np.isneginf(y_unnormed)] = np.nanmedian(y_unnormed)
        y_unnormed = torch.Tensor(y_unnormed).float()
        
        # normalized -> ~std normal
        y = (y_unnormed - self.dataset.current_maps_mean) / self.dataset.current_maps_std
        
        try:
            y_char = process_image(y_unnormed, self.dataset.img_size_um)
        except: breakpoint()
        
        # ---- bootstrap y_char 10x ----
        # NUM_BOOTSTRAPS = 10

        # for i in range(NUM_BOOTSTRAPS - 1):

        #     y_aug = y.clone()
        #     if random.random() < 0.5:
        #         y_aug = torch.flip(y_aug, dims=[1])
        #     if random.random() < 0.5:
        #         y_aug= torch.flip(y_aug, dims=[0])
        #     if y.shape[0] == y.shape[1] and random.random() < 0.5:
        #         y_aug = torch.rot90(y_aug, k=1, dims=(0, 1))
        #     if random.random() < 0.5:
        #         factor = random.uniform(0.9, 1.1)
        #         y_aug = y_aug * factor
        #     if random.random() < 0.5:
        #         noise_std = 0.05 * (y_aug.max() - y_aug.min())
        #         noise = torch.randn_like(y_aug) * noise_std
        #         y_aug = y_aug + noise
        #     if random.random() < 0.5:
        #         # scale randomly 1x-1.3x
        #         y_aug = MOS2SefOLDERSurrogateDataset.scale_image(y_aug, 1 + (random.random() * 0.3))

        #     y_char_bootstrapped = process_image(y_aug, self.dataset.img_size_um)
            
        #     for k, v in y_char.items():
        #         y_char[k] = (y_char[k] + y_char_bootstrapped[k])
        
        # for k, v in y_char.items():
        #     y_char[k] = (y_char[k] + y_char_bootstrapped[k]) / NUM_BOOTSTRAPS

        # ---- normalize all vals -> ~std-normal ----
        for k in y_char:
            val = y_char[k]
            mean = self.normalization_dict[k]['mean']
            std = self.normalization_dict[k]['std']
            y_char[k] = (val - mean) / std
        
        # # NOTE: remove high variance features
        # y_char.pop("num_curved_lines")
        # y_char.pop("num_extended_shapes")
        # y_char.pop("total_area_extended_shapes")
        
        # for peace of mind; manually select features for target array
        target_arr = [None] * NUM_CHAR_FEATURES
        target_arr[0] = y_char['coverage_percentage']
        target_arr[1] = y_char['total_len_detected_curves']
        target_arr[2] = y_char['total_area_circular_shapes']
        target_arr[3] = y_char['total_defect_area']
        target_arr[4] = y_char['num_circular_shapes']
        target_arr[5] = y_char['average_surface_current']
        target_arr[6] = y_char['num_curved_lines']
        target_arr[7] = y_char['num_extended_shapes']
        target_arr[8] = y_char['total_area_extended_shapes']
        target = torch.Tensor(target_arr).float()

        item = {}
        item['y'] = y
        item['y_char'] = y_char
        item['target'] = target

        return item


if __name__ == "__main__": 
    dataset = MOS2SefOLDERSurrogateDataset()
    dataset[0]
