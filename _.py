import warnings
warnings.filterwarnings("ignore")

import torch
from eval.inpainting_evaluator import InpaintingEvaluator
from src.datasets.inpainting import InpaintingEvaluationDataset
from src.util.config import parse_config

def main():
    FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/eval/example_config.yaml"
    dataset = InpaintingEvaluationDataset()
    model = torch.nn.Module()
    config = parse_config(FP)
    evaluator = InpaintingEvaluator(model, dataset, config)
    evaluator.eval()
    
if __name__ == "__main__":
    main()