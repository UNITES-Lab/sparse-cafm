import cv2
import os
import torch
import datetime
import pandas as pd
import yaml
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from src.util.torch_helpers import convert_to_img_like

EXPS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/"
FIGURES_DIR_NAME = "figures"
RESULTS_CSV_NAME = "results.csv"


class ExperimentLogger:
    """
    A flexible logger used to record and organize experimental runs.
    """

    def __init__(
        self,
        config_fp: str,
        root: str = EXPS_DIR,
        exp_name: Optional[str] = "",
        log_interval: int = 100,
        enable_tensorboard=False,
        enable_wandb=False,
        wandb_proj_name: Optional[str] = None,
    ) -> None:
        """
        :param config_fp:           path to a `.yaml` config file containing all hps
        :param root:                path to top experiment dir
        :param exp_name:            name of the experiment
        :param log_interval:        how often to write log results to .csv file
        :param enable_tensorboard:  flag to enable tensorboard logging
        :param enable_wandb:        flag to enable W&B logging [NOT SUPPORTED]
        :param wandb_project_name:  name of W&B project (e.g. "my-project")
        """

        assert config_fp.endswith(".yaml")
        self.config_fp: str = config_fp
        self.exp_name: str = exp_name
        self.results = pd.DataFrame()
        self.log_interval: int = log_interval
        self.log_counter = 0
        self.root: str = root
        self.enable_tensorboard: bool = enable_tensorboard
        self.exp_dir: Optional[str] = None
        # tensorboard support
        self.results_out_path: Optional[str] = None
        self.summary_writer: Optional[SummaryWriter] = None
        # wandb support
        self.enable_wandb = enable_wandb
        if self.enable_wandb == True:
            assert (
                wandb_proj_name != None
            ), f"Error: must provide a valid name for wandb_proj_name"
        self.wandb_proj_name = wandb_proj_name
        self.wandb_run = None
        self._setup_exp_dir()

    def _update_csv(self) -> None:
        self.results.to_csv(self.results_out_path, index=False)

    def _setup_exp_dir(self) -> None:

        # get date and time as a string
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name = date_time_str + "_" + self.exp_name
        exp_out_dir = os.path.join(self.root, subdir_name)
        self.exp_dir = exp_out_dir

        # make new subdir if needed
        os.makedirs(exp_out_dir, exist_ok=True)
        # save config in subdir
        config_save_fp = os.path.join(exp_out_dir, "config.yaml")
        with open(config_save_fp, "w") as f:
            with open(self.config_fp, "r") as g:
                f.write(g.read())
        self.config_fp = config_save_fp

        # path to results csv file
        self.results_out_path = os.path.join(exp_out_dir, RESULTS_CSV_NAME)

        # optional: create a tensorboard writer object
        if self.enable_tensorboard:
            tb_log_dir = os.path.join(self.exp_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=tb_log_dir)

        # optional: create a wandb run
        if self.enable_wandb:
            with open(self.config_fp, "r") as f:
                config_dict = yaml.safe_load(f)
            wandb.init(
                project=self.wandb_proj_name,
                name=self.exp_name,
                config=config_dict,
                dir=self.exp_dir,
            )
            self.wandb_run = wandb.run

    def add_result_column(self, name: str) -> None:
        self.results[name] = None
        self._update_csv()

    def add_result_columns(self, names: List[str]) -> None:
        for name in names:
            self.add_result_column(name)
        self._update_csv()

    def log(self, **kwargs) -> None:
        # log -> csv
        self.results = pd.concat(
            [self.results, pd.DataFrame.from_records([kwargs])], ignore_index=True
        )
        if self.log_counter % self.log_interval == 0:
            self._update_csv()
        self.log_counter += 1
        # optional: log -> tensorboard
        if self.enable_tensorboard:
            if step is None:
                step = self.log_counter
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self.summary_writer.add_scalar(k, v, step)
        # optional: log -> wandb
        if self.enable_wandb:
            step = self.log_counter
            wandb_dict = {
                k: v for k, v in kwargs.items() if isinstance(v, (int, float))
            }
            wandb.log(wandb_dict, step=step)

    def save_weights(self, x: torch.nn.Module, name: str = "best") -> None:
        """
        Save model weights of a `torch.nn.Module` object to the current exp dir.

        :param x: model to save
        """
        model_out_path = os.path.join(self.exp_dir, f"{self.exp_name}_{name}.pth")
        torch.save(x.state_dict(), model_out_path)

    def save_tensorlike_data(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray],
        subdir: Optional[str] = None,
    ) -> None:
        """
        Log `torch.Tensor`-like to data to the current exp dir.

        Currently supports:
            - `.npy`

        :param name: name of the image
        :param img_like: image to log
        :param subdir: subdirectory to save to
        """

        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        # optionally, save in a subdir
        if subdir is not None:
            outdir = os.path.join(outdir, subdir)
            os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # TODO: support other data formats
        if name.endswith(".npy"):
            np.save(out_fp, data)

    def log_original_masked_predicted_sample_triplet(
        self,
        y: torch.Tensor,
        y_sparse: torch.Tensor,
        y_hat: torch.Tensor,
        name: str,
    ) -> None:
        """
        Expect inputs with shapes (B, H, W).
        """
        # (B, H, W) -> (H, W)
        y = y[0, ...]
        y_sparse = y_sparse[0, ...]
        y_hat = y_hat[0, ...]
        # (H, W) -> (H, W, C)
        y, y_sparse, y_hat = convert_to_img_like(y, y_sparse, y_hat)
        combined_image = np.concatenate([y, y_sparse, y_hat], axis=1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_image)
        ax.axis("off")
        h, w = y.shape[:2]
        labels = ["Original", "Masked", "Predicted"]
        for i, label in enumerate(labels):
            x_pos = i * w + w // 2
            ax.text(x_pos, -4, label, fontsize=14, ha="center", color="black")
        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        plt.savefig(out_fp, bbox_inches='tight', pad_inches=0.1, dpi=300)

    def log_original_masked_predicted_sample_triplet_controlnet(
        self,
        y: torch.Tensor,
        y_sparse: torch.Tensor,
        y_hat: torch.Tensor,
        name: str,
    ) -> None:
        """
        Expect img-like inputs with shapes (H, W, C).
        """
        combined_image = np.concatenate([y, y_sparse, y_hat], axis=1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_image)
        ax.axis("off")
        h, w = y.shape[:2]
        labels = ["Original", "Masked", "Predicted"]
        for i, label in enumerate(labels):
            x_pos = i * w + w // 2
            ax.text(x_pos, -4, label, fontsize=14, ha="center", color="black")
        outdir = os.path.join(self.exp_dir, FIGURES_DIR_NAME)
        # create the figures dir if it does not already exist
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, name)
        plt.savefig(out_fp, bbox_inches='tight', pad_inches=0.1, dpi=300)
