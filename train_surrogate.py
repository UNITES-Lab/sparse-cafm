import os
import sys
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader
from src.models.our_method.swin_cafm import SwinCAFM
from src.models.our_method.older_surrogate import OlderSurrogate
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.logger import ExperimentLogger
from src.util.loss import ImageInpaintingL1Loss
from src.util.config import (
    TrainConfig,
    ModelConfig,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    MODELS,
)
from src.util.celano_lab_scripts import process_image as celano_lab_characterization
from src.util.metrics import OLDER

TRAIN_CONFIG_FP = os.path.abspath("/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/configs/older_surrogate.yaml")


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
    split_str = "training" if split == "train" else "validation"
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
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def train(config: TrainConfig, model_config: Optional[ModelConfig] = None) -> None:
    """
    Train OLDER surrogate model.
    
    TODO: verify this isn't just dumb
    It may be easier to attempt to train two models at the same time.
    1. Model-A: p(y | y_sparse)
    2. Model-B  p(older | y_sparse, y_hat)
    
    Given:
        1. y_sparse
        2. y_hat
    Predict: 
        1. OLDER: [0, inf)
    """

    logger = setup_logger(config, model_config)
    infilling_model = create_model(config)
    older_surrogate_model = OlderSurrogate()
    
    train_dataloader = create_dataloader(config, "train")
    val_dataloader = create_dataloader(config, "val")

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config.train_loss]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config.val_loss]()

    best_loss = sys.maxsize
    num_epochs = config.epochs
    device = config.device

    # create model using model config obj
    # NOTE: only supported for SwinCAFM atm
    if config.model_config_file != None:
        assert isinstance(infilling_model, SwinCAFM), f"Only SwinCAFM supports init from config."
        infilling_model = SwinCAFM.init_from_config(model_config.to_dict())

    # load weights from checkpoint
    if config.weights != None:
        # load weights only:
        # model.load_state_dict(torch.load(config["model"]["weights"]), strict=False)
        # load enitre model object:
        infilling_model = torch.load(config.weights).float().cuda()
    
    infilling_model.cuda(device)
    infilling_model.float()
    older_surrogate_model.cuda(device)
    older_surrogate_model.float()
    
    surrogate_optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=older_surrogate_model.parameters(),
        lr=1e-4
    )
    infilling_optimizer: torch.optim.Optimizer = OPTIMIZERS[config.optimizer](
        infilling_model.parameters(), lr=float(config.learning_rate)
    )
   
    train_dataset: MOS2SEFDataset = train_dataloader.dataset
    val_dataset: MOS2SEFDataset = val_dataloader.dataset
    
    # ---------- training loop ----------
    for epoch in range(num_epochs):
        
        infilling_model.train()
        older_surrogate_model.train()
        
        running_loss = 0.0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):
            
            # target: y
            y: torch.Tensor = batch["y"].cuda(device)

            # mask
            y_mask: torch.Tensor = batch["y_mask"].cuda(device)
            y_sparse = (y * y_mask).float()

            # zero gradients
            infilling_optimizer.zero_grad()
            surrogate_optimizer.zero_grad()

            # p(y_hat|y_sparse)]
            # forward: [H, W]
            outputs = infilling_model(y_sparse)
            y_hat = ImageInpaintingL1Loss.get_final_prediction(
                predicted_image=outputs, target_image=y, mask=y_mask
            )
            
            mean, std = train_dataset.current_maps_mean, train_dataset.current_maps_std
            
            # ---- characterize(y) ----
            # z: [0, 1] -> [-1, 1] (i.e., standard normal)
            z = (y * 2) - 1
            # [-1, 1] -> original dist
            # x' = mu + (sigma * z)
            data = mean + (std * z)
            y_char = celano_lab_characterization(data, train_dataset.img_size_um)
            
            # ---- characterize(y_sparse) ----
            # z: [0, 1] -> [-1, 1] (i.e., standard normal)
            z = (y_hat * 2) - 1
            # [-1, 1] -> original dist
            # x' = mu + (sigma * z)
            data = mean + (std * z)
            y_sparse_char = celano_lab_characterization(data, train_dataset.img_size_um)
            
            # calculate older scores
            older_gt = OLDER(y_char, y_sparse_char)
            older_pred = older_surrogate_model(y_sparse, y_hat)
            
            # HACK: [y-y=0]
            # ---- minimize older w.r.t. denoising model weights ----
            # 1. OLDER
            # infilling_loss = train_loss(older_pred, older_pred * 0)
            
            # 2. OLDER + L1
            # infilling_loss = train_loss(older_pred, older_pred * 0) + torch.nn.functional.l1_loss(y, y_hat)
            
            # 3. sigmoid(OLDER) + L1
            _older_pred_norm = torch.nn.functional.sigmoid(older_pred)
            infilling_loss: torch.Tensor = train_loss(_older_pred_norm, _older_pred_norm * 0) + torch.nn.functional.l1_loss(y, y_hat)
            # ------------------------------------------------------
            
            # NOTE: must retain graph, we will backprop again using surrogate model
            infilling_loss.backward(retain_graph=True)
            
            # ----  minimize || older_pred - older_gt || w.r.t. surrogate model weights  ----
            B = y.shape[0]
            older_gt_tensor = torch.Tensor([[older_gt]] * B).float().cuda()
            surrogate_loss = torch.nn.functional.l1_loss(older_pred, older_gt_tensor)
            surrogate_loss.backward()
            
            surrogate_optimizer.step()
            infilling_optimizer.step()
            
            running_loss += infilling_loss.item() * y_sparse.size(0)
            
            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + i,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_denoising_loss": infilling_loss.item(),
                    "val_denoising_loss": None,
                    "train_surrogate_loss": surrogate_loss.item(),
                    "val_surrogate_loss": None,
                }
            )
         
            # log a triplet (original, masked, predicted) every 100 steps
            if i % 100 == 0:
                triplet_name = f"train_epoch_{epoch}_step_{i}.png"
                final_pred = ImageInpaintingL1Loss.get_final_prediction(
                    predicted_image=outputs, target_image=y, mask=y_mask
                )
                logger.log_original_masked_predicted_sample_triplet(
                    y, y_sparse, final_pred, triplet_name
                )

        # validation
        infilling_model.eval()
        older_surrogate_model.eval()
        val_running_loss = 0.0
        
        avg_val_loss = 0.0
        num_val_steps = 0

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):
                
                # target: y
                y: torch.Tensor = batch["y"].cuda(device)

                # mask
                y_mask: torch.Tensor = batch["y_mask"].cuda(device)
                y_sparse = (y * y_mask).float()

                # forward
                # p(y_hat | y_sparse)
                outputs = infilling_model(y_sparse)
                y_hat = ImageInpaintingL1Loss.get_final_prediction(
                    predicted_image=outputs, target_image=y, mask=y_mask
                )
                
                mean, std = train_dataset.current_maps_mean, train_dataset.current_maps_std
                
                # ---- characterize(y) ----
                # z: [0, 1] -> [-1, 1] (i.e., standard normal)
                z = (y * 2) - 1
                # [-1, 1] -> original dist
                # x' = mu + (sigma * z)
                data = mean + (std * z)
                y_char = celano_lab_characterization(data, train_dataset.img_size_um)
                
                # ---- characterize(y_sparse) ----
                # z: [0, 1] -> [-1, 1] (i.e., standard normal)
                z = (y_hat * 2) - 1
                # [-1, 1] -> original dist
                # x' = mu + (sigma * z)
                data = mean + (std * z)
                y_sparse_char = celano_lab_characterization(data, train_dataset.img_size_um)
                
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
                        y, y_sparse, final_pred, triplet_name
                    )
    
            # optionally log best/epoch model weights
            if num_val_steps > 0:
                avg_val_loss = val_running_loss / num_val_steps
            
            if bool(config.save_weights):
                if bool(config.save_only_best_weights):
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logger.save_weights(infilling_model, "best_infilling_model")
                        logger.save_weights(older_surrogate_model, "best_older_surrogate")
                    else:
                        # NOTE: we overwrite previous "latest" weights
                        logger.save_weights(infilling_model, "latest_infilling_model")
                        logger.save_weights(older_surrogate_model, "latest_older_surrogate")
                else:
                    logger.save_weights(infilling_model, f"epoch_{epoch}_infilling_model")
                    logger.save_weights(older_surrogate_model, f"epoch_{epoch}_surrogate")


def main(args: argparse.Namespace) -> None:
    
    # load training config
    config = TrainConfig(TRAIN_CONFIG_FP)
    model_config: Optional[ModelConfig] = None
    
    # optional: parse model config
    if config.model_config_file != None:
        model_config_abs_path = os.path.join(
            Path(TRAIN_CONFIG_FP).parent.__str__(), config.model_config_file
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
    train(config, model_config)


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
