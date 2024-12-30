# this class should provide flexible support for evaluating
# many different models on the same benchmarks/metrics

import yaml
import cv2
import torch
import os
import tqdm
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from src.util.logger import ExperimentLogger
from src.util.config import parse_config, LOSS_FUNCTIONS
from src.util.torch_helpers import move_to
from src.datasets.inpainting import InpaintingEvaluationDataset


class InpaintingEvaluator:

    def __init__(
        self,
        experiment_logger: ExperimentLogger,
        model: torch.nn.Module,
        dataset: InpaintingEvaluationDataset,
        config: dict,
        out_dir: str,
        device: int = 0,
    ):
        """
        ...
        """
        self.experiment_logger: ExperimentLogger = experiment_logger
        self.model: torch.nn.Module = model
        self.dataset: InpaintingEvaluationDataset = dataset
        self.config: dict = config
        self.dataloader: DataLoader = DataLoader(
            dataset=dataset,
            batch_size=self.config["dataloader"]["batch_size"],
            shuffle=self.config["dataloader"]["shuffle"],
            num_workers=self.config["dataloader"]["num_workers"],
        )
        # assert os.path.isdir(out_dir), f"Error: out_dir @ {out_dir} does not exist"
        self.out_dir: str = out_dir
        self.device: int = device

    @torch.no_grad()
    def eval(self):
        """
        Evaluate a model a suite of pre-defined metrics
        1. FID
        2. LPIPS
        """

        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            # batch -> GPU
            batch = move_to(batch, self.device)
            # (N, H, W, 3)
            image: torch.Tensor = batch["image"]
            # -> # (N, 3, H, W)
            image = image.permute(0, 3, 1, 2)
            # (N, 1, H, W)
            mask: torch.Tensor = (batch["mask"] > 0) * 1
            # mask out pixels in original image
            masked_image = image * (1 - mask)
            # forward
            predicted_pixels: torch.Tensor = self.model(masked_image)
            predicted_image = (mask * predicted_pixels) + (1 - mask) * batch["image"]
            # unpad image if needed
            unpad_to_size = batch["original_image_shape"]
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]


if __name__ == "__main__":
    pass
