import os
import sys
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader
from src.models.our_method.older_surrogate import  MultiHeadOLDERSurrogate
from src.datasets.mos2_sef import Formulation as F
from src.datasets.mos2_sef_surrogate import MOS2SefOLDERSurrogateDataset, SyntheticMOS2SefOLDERSurrogateDataset
from src.util.logger import ExperimentLogger
from src.util.config import (
    TrainConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    MODELS,
)
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from src.util.metrics import OLDER

EVAL_CONFIG_FP = os.path.abspath("/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/configs/train-configs/eval_older_surrogate_standalone.yaml")


def setup_logger(train_config: TrainConfig, model_config: Optional[ModelConfig]) -> ExperimentLogger:
    logger = ExperimentLogger(
        train_config_dict=train_config.to_dict(),
        model_config_dict = model_config.to_dict() if model_config != None else None,
        root=train_config.log_root,
        exp_name=train_config.exp_name,
        log_interval=train_config.log_interval,
    )
    logger.add_result_columns(train_config.result_columns)
    return logger


def create_model(config: TrainConfig) -> nn.Module:
    model_fn = MODELS[config.model_name]["fn"]
    model_weights = MODELS[config.model_name]["weights"]
    if model_weights:
        model = model_fn(weights=model_weights)
    elif config.model_name == "hiera":
        model = model_fn
        model.freeze()
    else:
        model = model_fn()
    assert isinstance(model, nn.Module)
    return model.cuda(config.device).float()


def create_dataloader(config: TrainConfig, split: str) -> DataLoader:
    img_size = int(config.image_size)
    dataset = MOS2SefOLDERSurrogateDataset(
        split=split,
        side_length=int(config.crop_size),
        formulation=F.get_formulation_from_str(config.formulation),
        steps_per_epoch=(
            config.steps_per_epoch if split == "train" else config.val_steps_per_epoch
        ),
        device=config.device,
        original_image_size=(img_size, img_size),
        masking_ratio=int(config.masking_ratio),
        normalize_on_init=True if split == "train" else False,
    )
    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def train(args: argparse.Namespace, config: TrainConfig, model_config: Optional[ModelConfig] = None) -> None:
    """
    Train OLDER surrogate model.
    
    Given:
        1. y
    Predict: 
        1. OLDER: characterization of y
    """

    logger = setup_logger(config, model_config)
    older_surrogate_model: MultiHeadOLDERSurrogate = create_model(config)
    
    train_dataloader = create_dataloader(config, "train")
    val_dataloader = create_dataloader(config, "val")
    train_dataset: MOS2SefOLDERSurrogateDataset = train_dataloader.dataset
    val_dataset: MOS2SefOLDERSurrogateDataset = val_dataloader.dataset

    # NOTE: use the same mean/std vals normalize both dataloaders to ~std normal
    # val_dataset.val_current_map_buffer = train_dataset.val_current_map_buffer
    val_dataset.normalization_dict = train_dataset.normalization_dict

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config.train_loss]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config.val_loss]()

    best_loss = sys.maxsize
    num_epochs = config.epochs
    device = config.device

    # load weights from checkpoint
    if config.weights != None:
        # load weights only:
        # model.load_state_dict(torch.load(config["model"]["weights"]), strict=False)
        # load enitre model object:
        older_surrogate_model = torch.load(config.weights).float().cuda()
    
    older_surrogate_model.cuda(device)
    older_surrogate_model.float()

    # ---------- test loop ----------

    # validation
    older_surrogate_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(val_dataloader, desc=f"Evaluating..:")
        ):
            # input: y
            y: torch.Tensor = batch["y"].cuda(device)

            # char
            y_char: dict = batch['y_char']
            
            # targets
            target: torch.Tensor = batch['target'].cuda(device)

            # ---- forward: [H, W] ----
            pred = older_surrogate_model(y)

            # HACK: calculate errors by feature category; assume BS=1
            errors = (target - pred).detach().cpu().numpy().tolist()[0]
            
            OLDER_l1_loss: torch.Tensor = torch.nn.functional.l1_loss(pred, target)
            
            running_loss += OLDER_l1_loss.item() * y.size(0)
            
            logger.log(
                **{
                    "step": i,
                    "OLDER_l1_loss": OLDER_l1_loss,
                    "features_loss": None,
                    "y_char": y_char,
                    "errors": errors,
                }
            )
        
            # log a triplet (original, masked, predicted) every 100 steps
            if i % 100 != 0: continue
            triplet_name = f"eval_step_{i}.png"
            logger.log_colorized_tensors(
                (y, "Input (y)"),
                file_name=triplet_name
            )


def main(args: argparse.Namespace) -> None:
    
    # load training config
    config = TrainConfig(EVAL_CONFIG_FP)
    model_config: Optional[ModelConfig] = None
    
    # optional: parse model config
    if config.model_config_file != None:
        model_config_abs_path = os.path.join(
            Path(EVAL_CONFIG_FP).parent.__str__(), config.model_config_file
        )
        assert os.path.isfile(
            model_config_abs_path
        ), f"Bad path to model config: {model_config_abs_path}"
        model_config = ModelConfig(model_config_abs_path)
        
    # -------------------- training config args --------------------
    config.exp_name = args.exp_name
    # -------------------- model config args --------------------
    if model_config != None:
        # transformer block depths; e.g., [6, 6, 6, 6, 6, 6]
        model_config.depths = [args.depths] * args.num_blocks
        # num heads per block; e.g., [6, 6, 6, 6, 6, 6]
        model_config.num_heads = [args.num_heads] * args.num_blocks
        # size of sifted-attention window
        model_config.window_size = args.window_size
        model_config.drop_path_rate = args.drop_path_rate
        model_config.norm_layer = args.norm_layer    
    
    # train
    train(args, config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------- training config args --------------------
    parser.add_argument("-e", "--exp_name", type=str, help="Experiment directory name", default="my-experiment")
    # -------------------- model config args --------------------
    parser.add_argument("-dps", "--depths", type=int, help="Depths of RSTB blocks", default=6)
    parser.add_argument("-nbs", "--num_blocks", type=int, help="Number of RSTB blocks", default=6)
    parser.add_argument("-nhs", "--num_heads", type=int, help="Number of heads per RSTB block", default=6)
    parser.add_argument("-wsz", "--window_size", type=int, help="Size of shifted attention window", default=8)
    parser.add_argument("-dpr", "--drop_path_rate", type=float, help="", default=0.1)
    parser.add_argument("-nlr", "--norm_layer", type=str, help="", default="torch.nn.LayerNorm")
    args = parser.parse_args()
    main(args)