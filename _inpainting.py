import warnings

warnings.filterwarnings("ignore")

import torch
from eval.inpainting_evaluator import InpaintingEvaluator
from src.datasets.inpainting import InpaintingEvaluationDataset
from src.util.config import parse_config
from src.models.inpainting.toy import ToyModel
from _lama.saicinpainting.training.modules.ffc import FFCResNetGenerator


def main():
    FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/eval/example_config.yaml"
    dataset = InpaintingEvaluationDataset(pad_img_to_mod_by=4)
    
    model = FFCResNetGenerator(
        input_nc=4,
        output_nc=3,
        init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        downsample_conv_kwargs={
            "ratio_gin": 0,
            "ratio_gout": 0,
            "enable_lfu": False,
        },
        resnet_conv_kwargs={
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "enable_lfu": False,
        },
    )
    
    config = parse_config(FP)
    out_dir = ""
    experimental_logger = None
    evaluator = InpaintingEvaluator(
        experimental_logger, model, dataset, config, out_dir
    )
    evaluator.eval()


if __name__ == "__main__":
    main()
