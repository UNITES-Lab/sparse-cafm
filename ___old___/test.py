import os
import argparse
import random
from typing import Optional
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from src.datasets.mos2_sr import MOS2SRDataset, MOS2_SYNTHETIC, MOS2_SAPPHIRE_DIR, MOS2_SEF_SRC_DIR, MOS2_SILICON_DIR, UnifiedMOS2SRDataset
from src.models.our_method.swin_cafm import SwinCAFM
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from src.util.logger import ExperimentLogger
from src.util.loss import ImageInpaintingL1Loss
from src.util.metrics import OLDER, PSNR, MSE, MAE, SSIM
from src.util.config import (
    EvalConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    MODELS,
)

EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")


def setup_logger(
    train_config: EvalConfig, model_config: Optional[ModelConfig]
) -> ExperimentLogger:
    logger = ExperimentLogger(
        train_config_dict=train_config.to_dict(),
        model_config_dict=model_config.to_dict() if model_config != None else None,
        root=train_config.log_root,
        exp_name=train_config.exp_name,
        log_interval=train_config.log_interval,
    )
    logger.add_result_columns(train_config.result_columns)
    return logger


def create_model(config: EvalConfig) -> nn.Module:
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
    return model


def create_dataloader(args, config: EvalConfig, split: str) -> DataLoader:
    
    assert str(args.dataset) in ['all', 'synthetic', 'mos2-sef', 'sapphire', 'silicon']
    
    src_dir = {
        "all": None,
        "synthetic": MOS2_SYNTHETIC,
        "mos2-sef": MOS2_SEF_SRC_DIR,
        "sapphire": MOS2_SAPPHIRE_DIR,
        "silicon": MOS2_SILICON_DIR
    }[args.dataset]
    
    dataset = None
    if str(args.dataset) == 'all':
        dataset = UnifiedMOS2SRDataset(
            split=split,
            steps_per_epoch=(
                int(config.steps_per_epoch * config.train_batch_size)
                if split == "train"
                else config.val_steps_per_epoch
            ),
            upsample_factor=int(args.upsample_factor)
        )
    else:
        dataset = MOS2SRDataset(
            src_dir=src_dir,
            split=split,
            steps_per_epoch=(
                int(config.steps_per_epoch * config.train_batch_size)
                if split == "train"
                else config.val_steps_per_epoch
            ),
            upsample_factor=int(args.upsample_factor)
        )
    return DataLoader(
        dataset,
        batch_size=(
            config.train_batch_size if split == "train" else config.val_batch_size
        ),
        shuffle=False,
        num_workers=config.num_workers,
    )

@torch.no_grad()
def eval(args, config: EvalConfig, model_config: ModelConfig) -> None:

    logger = setup_logger(config, model_config)
    model = create_model(config)
    val_dataloader = create_dataloader(args, config, "val")
    val_dataset: MOS2SRDataset = val_dataloader.dataset
    device = config.device

    # load weights from checkpoint
    if config.weights != None:
        model = torch.load(config.weights)
        
    assert isinstance(model, torch.nn.Module)

    # validation loop
    model.eval()

    for step, batch in enumerate(tqdm(val_dataloader, desc=f"Evaluating...:")):
        
        F = args.formulation
        assert F in ['X', 'y', 'both']

        y, y_sparse = None, None
        if F == 'both':
            _F = "y" if random.random() < 0.5 else "X"
            y: torch.Tensor = batch[_F].cuda(device)
            # current-map: y_sparse; [64, 64]
            y_sparse: torch.Tensor = batch[f"{_F}_sparse"].cuda(device)
        else:
            # current-map: y; [128, 128]
            y: torch.Tensor = batch[F].cuda(device)
            # current-map: y_sparse; [64, 64]
            y_sparse: torch.Tensor = batch[f"{F}_sparse"].cuda(device)
    
        # ---- forward: p(y | y_sparse) ----
        y_hat: torch.Tensor = model(y_sparse)
        # ----------------------------------

        # get final predicted image
        triplet_name = f"eval_step_{step}.png"

        final_pred = y_hat

        # 1. MAE
        mae = MAE(final_pred, y)
        # 2. MSE
        mse = MSE(final_pred, y)
        # 3. PSNR; assume data in range [0, 1]
        psnr = PSNR(final_pred, y, val_dataset.normalized_data_range)

        # (B, H, W) -> (B, 1, H, W)
        final_pred_img_like = final_pred.clone()
        final_pred_img_like = final_pred_img_like.unsqueeze(1)
        # (B, 1, H, W) -> (B, 3, H, W)
        final_pred_img_like = final_pred_img_like.repeat(1, 3, 1, 1)

        # (B, H, W) -> (B, 1, H, W)
        y_img_like = y.clone()
        y_img_like = y_img_like.unsqueeze(1)
        # (B, 1, H, W) -> (B, 3, H, W)
        y_img_like = y_img_like.repeat(1, 3, 1, 1)

        # 4. SSIM
        ssim_val = SSIM(final_pred_img_like, y_img_like, val_dataset.normalized_data_range)

        mean, std = val_dataset.current_maps_mean, val_dataset.current_maps_std

        # 5a. characterize(y)
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (y * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)

        try:
            y_char = celano_lab_characterization(data, val_dataset.img_size_um)
        except:
            y_char = None

        # 5b. characterize(y_sparse)
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (final_pred * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)

        # HACK: very rarely we run into an index OOB error b/c there are not peaks in an output map
        # we'll try to ignore these for now
        try:
            y_hat_char = celano_lab_characterization(data, val_dataset.img_size_um)
        except:
            y_hat_char = None

        logger.log(
            **{
                "step": step,
                "mae": mae.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
                "ssim": ssim_val.item(),
                "older": OLDER(y_char, y_hat_char) if y_char != None and y_hat_char != None else None,
                "celano_script_y": y_char,
                "celano_script_y_sparse": y_hat_char,
            }
        )
        
        if step % 100 == 0:
            logger.log_colorized_tensors(
                # (X, "Topology Map (X)"),
                # (X_sparse, "Model Input (X_sparse)"),
                (y, "Target (y)"),
                (y_sparse, "Model Input (y_sparse)"),
                (y_hat, "Raw Model Prediction"),
                (final_pred, "Model Prediction With Given Prior (y_hat)"),
                file_name=triplet_name
            )


def main(args: argparse.Namespace):

    # load training config
    config = EvalConfig(EVAL_CONFIG_FP)
    config.weights = args.weights

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
    config.log_root = args.root
    # config.learning_rate = str(args.learning_rate)
    # config.train_batch_size = int(args.batch_size)
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

    args.upsample_factor = int(args.upsample_factor)
    
    # run eval
    eval(args, config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------- training config args --------------------
    parser.add_argument("-e","--exp_name",type=str,help="Experiment directory name.",default="my-experiment",)
    parser.add_argument("-r","--root", type=str, help="Root directory to save experiment in.",default="__exps__/",)
    parser.add_argument("-ds", "--dataset", type=str, help="'synthetic', 'mos2-sef', 'sapphire', 'silicon', 'all']", default="mos2-sef")
    parser.add_argument("-ws", "--weights", type=str, help="Path to model checkpoints", default="")
    parser.add_argument("-fm", "--formulation", type=str, help="['X', 'y', 'both']", default="y")
    # -------------------- model config args --------------------
    parser.add_argument("-dps", "--depths", type=int, help="Depths of RSTB blocks", default=6)
    parser.add_argument("-nbs", "--num_blocks", type=int, help="Number of RSTB blocks", default=6)
    parser.add_argument("-nhs","--num_heads",type=int,help="Number of heads per RSTB block",default=6,)
    parser.add_argument("-wsz","--window_size",type=int,help="Size of shifted attention window",default=8,)
    parser.add_argument("-dpr", "--drop_path_rate", type=float, help="", default=0.1)
    parser.add_argument("-nlr", "--norm_layer", type=str, help="", default="torch.nn.LayerNorm")
    # -------------------- ablation args --------------------
    parser.add_argument("-sw", "--surrogate_weights", type=str, help="", default="")
    parser.add_argument("-lr", "--learning_rate", type=float, help="", default=1e-5)
    parser.add_argument("-bs", "--batch_size", type=int, help="", default=1)
    parser.add_argument("-sr", "--upsample_factor", type=int, help="", default=2)
    args = parser.parse_args()
    main(args)
