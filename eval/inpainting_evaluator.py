# this class should provide flexible support for evaluating
# many different models on the same benchmarks/metrics

import yaml
import torch
from typing import Optional
from torch.utils.data import DataLoader
from src.util.logger import ExperimentLogger
from src.util.config import parse_config, LOSS_FUNCTIONS
from src.datasets.inpainting import InpaintingEvaluationDataset


class InpaintingEvaluator:

    def __init__(
        self, model: torch.nn.Module, dataset: InpaintingEvaluationDataset, config: dict
    ):
        """
        We can simply pass in a config and dynamically instantiate the model.
        """
        self.model: torch.nn = model
        self.config: dict = config
        self.dataset: InpaintingEvaluationDataset = dataset
        self.dataloader: DataLoader = DataLoader(
            dataset=dataset,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=self.config['dataloader']['shuffle'],
            num_workers=self.config['dataloader']['num_workers'],
        )
        self.experiment_logger: Optional[ExperimentLogger] = None

    @torch.no_grad()
    def eval(self):
        """
        Evaluate a model a suite of pre-defined metrics
        1. FID
        2. LPIPS
        """
        pass

    def register_logger(self, logger: ExperimentLogger):
        self.experiment_logger = logger
        
if __name__ == "__main__":
    pass
