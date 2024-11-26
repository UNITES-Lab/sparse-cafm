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

EXPS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__"


# TODO: implement an experiment launcher for easy, co-ordinated ablations
class ExperimentLauncher:
    pass


class ExperimentLogger:
    """
    A flexible logger used for recording and organizing experimental runs.
    """

    def __init__(self, config_fp: str, exp_name: Optional[str] = "") -> None:
        """

        :param config_fp: path to a `.yaml` config file containing all hps
        """
        assert config_fp.endswith(".yaml")
        self.config_fp = config_fp
        self.exp_name = exp_name
        self.results_out_path: Optional[str] = None
        self.results = pd.DataFrame()
        self._setup_exp_dir()

    def _update_csv(self):
        self.results.to_csv(self.results_out_path, index=False)

    def _setup_exp_dir(self):

        # get date and time as a string
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name = date_time_str + "_" + self.exp_name
        exp_out_dir = os.path.join(EXPS_DIR, subdir_name)

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
        self._update_csv()
