import warnings

warnings.filterwarnings("ignore")

import torch
from eval.inpainting_evaluator import InpaintingEvaluator
from src.datasets.inpainting import InpaintingEvaluationDataset
from src.util.config import parse_config
from src.models.inpainting.toy import ToyModel


def main():
    FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/eval/example_config.yaml"
    dataset = InpaintingEvaluationDataset()
    model = ToyModel()
    config = parse_config(FP)
    out_dir = ""
    experimental_logger = None
    evaluator = InpaintingEvaluator(
        experimental_logger, model, dataset, config, out_dir
    )
    evaluator.eval()


if __name__ == "__main__":
    main()
