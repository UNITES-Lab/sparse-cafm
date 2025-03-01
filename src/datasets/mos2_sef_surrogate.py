import json
import torch
import random
import concurrent
import numpy as np
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from glob import glob
from tqdm import tqdm
from typing import Tuple, Dict
from torch.utils.data import Dataset
from src.datasets.mos2_sr import MOS2SRDataset
from src.util.celano_lab_scripts import process_image

NUM_CHAR_FEATURES = 1
CROPPED_IMAGE_SIDE_LENGTH = 128
ORIGINAL_IMAGE_SIZE = (512, 512)
EXPERT_FEATURE_NORMS = "data/raw-data/1-23-25/_surrogate_norms.json"
EXPERT_FEATURES = [
    "average_surface_current",
    "coverage_percentage",
    "total_area_extended_shapes",
    "total_len_detected_curves",
    "total_area_circular_shapes",
    "total_defect_area",
    "num_extended_shapes",
    "num_circular_shapes",
    "num_curved_lines",
]


def augment_and_process(y: torch.Tensor, img_size_um: float) -> torch.Tensor:
    """
    Performs a series of random augmentations on the image tensor `y_unnorm` and processes it.
    Returns the dictionary from process_image.
    """

    y_aug = y.clone()
    if random.random() < 0.5:
        y_aug = torch.flip(y_aug, dims=[1])
    if random.random() < 0.5:
        y_aug = torch.flip(y_aug, dims=[0])
    if y.shape[0] == y.shape[1] and random.random() < 0.5:
        y_aug = torch.rot90(y_aug, k=1, dims=(0, 1))
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        y_aug = y_aug * factor
    if random.random() < 0.5:
        noise_std = 0.05 * (y_aug.max() - y_aug.min())
        noise = torch.randn_like(y_aug) * noise_std
        y_aug = y_aug + noise
    if random.random() < 0.5:
        scale_factor = 1 + (random.random() * 0.3)
        y_aug = ExpertSurrogateDataset.scale_image(y_aug, scale_factor)

    return process_image(y_aug, img_size_um)


class ExpertSurrogateDataset(Dataset):
    """
    Dataset class used to train an expert-surrogate model.
    """

    def __init__(
        self,
        split: str = "train",
        upsample_factor: int = 2,
        steps_per_epoch: int = 100,
        original_image_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
        normalize_on_init: bool = False,
        surrogate_norms_fp: str = EXPERT_FEATURE_NORMS,
        expert_features: list = EXPERT_FEATURES,
    ):
        self.split = split
        self.upsample_factor = upsample_factor
        self.steps_per_epoch = steps_per_epoch
        self.original_image_size = original_image_size
        self.expert_features = expert_features

        self.dataset = MOS2SRDataset(
            split=split,
            upsample_factor=upsample_factor,
            steps_per_epoch=steps_per_epoch,
            original_image_size=original_image_size,
        )

        # dictionary of {"mean": float, "std": float} values
        self.normalization_dict: Dict[str, Dict] = {}

        if surrogate_norms_fp != None:
            with open(surrogate_norms_fp, "r") as f:
                self.normalization_dict = json.load(f)

        # optional: run a short benchmark to determine normalization mean/std
        if normalize_on_init and surrogate_norms_fp == None:
            self.normalize()

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
        image_scaled = F.interpolate(
            image.unsqueeze(0).unsqueeze(0).float(),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        )
        image_scaled = image_scaled.squeeze(0).squeeze(0)
        start_H = (new_H - H) // 2
        start_W = (new_W - W) // 2
        cropped_image = image_scaled[start_H : start_H + H, start_W : start_W + W]
        return cropped_image

    def __len__(self) -> int:
        return len(self.dataset)

    def normalize(self) -> None:
        """
        TODO: just run this test once, save the results somewhere and load as needed.
        
        Run a short test proceedure to calculate the mean and std of train/val samples;
        set global values for mean/std so that all samples are normalized roughly to the std normal.
        We make the apriori assumption that train/val samples belong to roughly the same distribution.
        """

        NUM_BENCHMARK_STEPS = 1000

        # Initialize datasets using direct indexing
        train_dataset = MOS2SEFDataset(
            split="train",
            formulation=self.formulation,
            side_length=self.side_length,
            masking_ratio=self.masking_ratio,
            steps_per_epoch=NUM_BENCHMARK_STEPS,
            device=self.device,
            original_image_size=self.original_image_size,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=8)

        # Use defaultdict to avoid membership checks
        samples = defaultdict(list)

        for train_item in tqdm(train_dataloader, total=NUM_BENCHMARK_STEPS, desc="Calculating global mean/stds..."):
            
            # Process images
            train_char = process_image(train_item["y_unnorm"], self.dataset.img_size_um)
            
            # Accumulate values for each key
            for k, v in train_char.items():
                samples[k].append(v)

        # Compute normalization statistics using numpy vectorized operations
        self.normalization_dict = {
            k: {"mean": np.mean(values), "std": np.std(values)}
            for k, values in samples.items()
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

        batch: dict = self.dataset[index]

        # [H, W]
        y: torch.Tensor = batch["y"]
        y_unnorm: torch.Tensor = batch["y_unnorm"]

        # get the celano-lab characterization of a raw current-map sample
        y_char = process_image(y_unnorm, self.dataset.img_size_um)

        # get a copy so that we can use non-bootstraped `average_surface_current`
        y_char_og = y_char.copy()

        # NOTE: we reduce variance by sampling multiple times from the expert charcterization script
        # using slight augmentations of the original input image
        NUM_BOOTSTRAPS = 20
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(augment_and_process, y_unnorm, self.dataset.img_size_um)
                for _ in range(NUM_BOOTSTRAPS - 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # add each result from the concurrent iterations.
        for res in results:
            for key in y_char:
                y_char[key] += res[key]

        # average over the total number of bootstraps.
        for key in y_char:
            y_char[key] /= NUM_BOOTSTRAPS

        # ---- normalize all vals -> ~std-normal ----
        for k in y_char:
            val = y_char[k]
            mean = self.normalization_dict[k]["mean"]
            std = self.normalization_dict[k]["std"]
            y_char[k] = (val - mean) / std

        for k in y_char_og:
            val = y_char_og[k]
            mean = self.normalization_dict[k]["mean"]
            std = self.normalization_dict[k]["std"]
            y_char_og[k] = (val - mean) / std

        # for peace of mind; manually select features for target array
        target_arr = [None] * len(self.expert_features)

        for i, feat in enumerate(self.expert_features):
            target_arr[i] = y_char[feat]
            
        target = torch.Tensor(target_arr).float()

        item = {}
        item["y"] = y
        item["y_char"] = y_char
        item["target"] = target
        return item


if __name__ == "__main__":
    dataset = ExpertSurrogateDataset()
    dataset[0]
