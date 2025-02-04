import os
import argparse
from typing import Optional
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
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

EVAL_CONFIG_FP = os.path.abspath("configs/train-configs/older_surrogate.yaml")


def setup_logger(train_config: EvalConfig, model_config: Optional[ModelConfig]) -> ExperimentLogger:
    logger = ExperimentLogger(
        train_config_dict=train_config.to_dict(),
        model_config_dict = model_config.to_dict() if model_config != None else None,
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
    return model.cuda(config.device).float()


def create_dataloader(config: EvalConfig, split: str) -> DataLoader:
    img_size = int(config.image_size)
    dataset = MOS2SEFDataset(
        split=split,
        side_length=int(config.crop_size),
        formulation=F.get_formulation_from_str(config.formulation),
        steps_per_epoch=(
            config.steps_per_epoch if split == "train" else config.val_steps_per_epoch
        ),
        device=config.device,
        original_image_size=(img_size, img_size),
        masking_ratio=int(config.masking_ratio),
    )
    return DataLoader(
        dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


@torch.no_grad()
def eval(args: argparse.Namespace, config: EvalConfig, model_config: ModelConfig) -> None:

    logger = setup_logger(config, model_config)
    model = create_model(config)
    
    val_dataloader = create_dataloader(config, "val")
    val_dataset: MOS2SEFDataset = val_dataloader.dataset
    device = config.device

    # load weights from checkpoint
    if config.weights != None:
        model = torch.load(config.weights)
    assert isinstance(model, torch.nn.Module)

    # validation loop
    model.eval()

    for step, batch in enumerate(tqdm(val_dataloader, desc=f"Evaluating...:")):

        # target: y
        y: torch.Tensor = batch["y"].cuda(device)

        # mask
        y_mask: torch.Tensor = batch["y_mask"].cuda(device)
        y_sparse = (y * y_mask).float()

        # ---- forward : p(y|y_sparse) ----
        outputs = model(y_sparse)
            
        # forward : p(y|y_sparse)
        # y_hat: torch.Tensor = model(y_sparse, y, y_mask)
        # ---------------------------------

        # log final predicted image
        triplet_name = f"eval_step_{step}.png"
        
        y_hat = ImageInpaintingL1Loss.get_final_prediction(
            predicted_image=outputs, target_image=y, mask=y_mask
        )
                
        mean, std = val_dataset.current_maps_mean, val_dataset.current_maps_std
        
        # ---- characterize(y) ----
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (y * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)
        y_char = celano_lab_characterization(data, val_dataset.img_size_um)
        
        # ---- characterize(y_sparse) ----
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (y_hat * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)
        y_sparse_char = celano_lab_characterization(data, val_dataset.img_size_um)
        
        # calculate older scores
        older_gt = OLDER(y_char, y_sparse_char)
        older_pred = older_surrogate_model(y_sparse, y_hat)
        
        # HACK: [y-y=0]
        # ---- minimize older w.r.t. denoising model weights ----
        loss = val_loss(older_pred, older_pred * 0)
        
        # ----  minimize || older_pred - older_gt || w.r.t. surrogate model weights  ----
        B = y.shape[0]
        older_gt_tensor = torch.Tensor([[older_gt]] * B).float().cuda()
        surrogate_loss = torch.nn.functional.l1_loss(older_pred, older_gt_tensor)
    
        val_running_loss += loss.item() * y_sparse.size(0)
        logger.log(
            **{
                "global_train_step": None,
                "global_val_step": len(val_dataloader) * (epoch) + i,
                "epoch": epoch,
                "train_denoising_loss": None,
                "val_denoising_loss": loss.item(),
                "train_surrogate_loss": None,
                "val_surrogate_loss": surrogate_loss.item(),
            }
        )
    
        # log a triplet (original, masked, predicted) every 100 steps
        if i % 100 == 0:
            triplet_name = f"val_epoch_{epoch}_step_{i}.png"
            final_pred = ImageInpaintingL1Loss.get_final_prediction(
                predicted_image=outputs, target_image=y, mask=y_mask
            )
            logger.log_original_masked_predicted_sample_triplet(
                y, y_sparse, final_pred, triplet)

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
        y_char = celano_lab_characterization(data, val_dataset.img_size_um)

        # 5b. characterize(y_sparse)
        # z: [0, 1] -> [-1, 1] (i.e., standard normal)
        z = (final_pred * 2) - 1
        # [-1, 1] -> original dist
        # x' = mu + (sigma * z)
        data = mean + (std * z)
        y_sparse_char = celano_lab_characterization(data, val_dataset.img_size_um)

        logger.log(
            **{
                "step": step,
                "mae": mae.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
                "ssim": ssim_val.item(),
                "older": OLDER(y_char, y_sparse_char),
                "celano_script_y": y_char,
                "celano_script_y_sparse": y_sparse_char,
            }
        )


def main(args: argparse.Namespace):

    config = EvalConfig(EVAL_CONFIG_FP)
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
    config.weights = args.model_weights_path
    
    # run eval
    eval(config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------- eval run config args --------------------
    parser.add_argument("-e", "--exp_name", type=str, help="Experiment directory name", default="my-experiment")
    parser.add_argument("-osp", "--older_surrogate_model_weights_path", type=str, help="Path to older surrogate model checkpoint to evaluate.")
    parser.add_argument("-dmp", "--denoising_model_weights_path", type=str, help="Path to denosing model checkpoint to evaluate.")
    # --------------------------------------------------------------
    args = parser.parse_args()
    main(args)
