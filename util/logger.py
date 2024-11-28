import cv2
import math
import os
import csv
import torch
import yaml
import datetime
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import List, Dict, Optional

EXPS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/p(z | X)/ablation-loss-fn"


# TODO: implement an experiment launcher for easy, co-ordinated ablations
class ExperimentLauncher:
    pass


class ExperimentLogger:
    """
    A flexible logger used for recording and organizing experimental runs.
    """

    def __init__(
        self, config_fp: str, exp_name: Optional[str] = "", log_interval: int = 100
    ) -> None:
        """
        :param config_fp: path to a `.yaml` config file containing all hps
        :param exp_name: name of the experiment
        :param log_interval: how often to write log results to .csv file
        """
        assert config_fp.endswith(".yaml")
        self.config_fp = config_fp
        self.exp_dir: Optional[str] = None
        self.exp_name = exp_name
        self.results_out_path: Optional[str] = None
        self.results = pd.DataFrame()
        self.log_interval = log_interval
        self.log_counter = 0
        self._setup_exp_dir()

    def _update_csv(self):
        self.results.to_csv(self.results_out_path, index=False)

    def _setup_exp_dir(self):

        # get date and time as a string
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name = date_time_str + "_" + self.exp_name
        exp_out_dir = os.path.join(EXPS_DIR, subdir_name)
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
        self.results_out_path = os.path.join(exp_out_dir, "results.csv")

    def add_result_column(self, name: str):
        self.results[name] = None
        self._update_csv()

    def add_result_columns(self, names: List[str]):
        for name in names:
            self.add_result_column(name)
        self._update_csv()

    def log(self, **kwargs):
        self.results = pd.concat(
            [self.results, pd.DataFrame.from_records([kwargs])], ignore_index=True
        )
        if self.log_counter % self.log_interval == 0:
            self._update_csv()
        self.log_counter += 1

    def save_sample(self, X: torch.Tensor, epoch: int, name: Optional[str] = ""):
        """
        :param X: (B, H, W, C)
        :param epoch: int
        :param name: str
        """
        figures_dir = os.path.join(self.exp_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        x_out_path = os.path.join(self.exp_dir, "figures", f"{name}_{epoch}.png")
        # save img
        X_np = X.detach().cpu().numpy()
        X_np = X_np[0, :, :, :]
        cv2.imwrite(x_out_path, X_np)
