import os
import yaml
import sys
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.datasets.mos2_sef import MOS2SEFDataset, Formulation as F
from src.util.logger import ExperimentLogger
from src.util.config import LOSS_FUNCTIONS, OPTIMIZERS, MODELS, parse_config
from src.util.loss import ImageInpaintingL1Loss

TRAIN_CONFIG_FP = os.path.abspath("configs/train.yaml")
EVAL_CONFIG_FP = os.path.abspath("configs/eval.yaml")
Z_MULT = 1


def setup_logger(config: dict) -> ExperimentLogger:
    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])
    return logger


def create_model(config: dict) -> nn.Module:
    model_fn = MODELS[config["model"]["name"]]["fn"]
    model_weights = MODELS[config["model"]["name"]]["weights"]
    if model_weights:
        model = model_fn(weights=model_weights)
    elif config["model"]["name"] == "hiera":
        model = model_fn
        model.freeze()
    else:
        model = model_fn()
    assert isinstance(model, nn.Module)
    return model.cuda(config["global"]["device"]).float()


def create_dataloader(config: dict, split: str) -> DataLoader:
    split_str = "training" if split == "train" else "validation"
    img_size = int(config["dataset"]["image_size"])
    dataset = MOS2SEFDataset(
        split=split,
        side_length=int(config['dataset']['crop_size']),
        formulation=F.get_formulation_from_str(config["global"]["formulation"]),
        steps_per_epoch=config[split_str]["steps_per_epoch"],
        device=config["global"]["device"],
        original_image_size=(img_size, img_size),
        masking_ratio=int(config["dataset"]["masking_ratio"]),
    )
    return DataLoader(
        dataset,
        batch_size=config[split_str]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )


@torch.no_grad()
def eval(config: dict) -> None:

    logger = setup_logger(config)
    model = create_model(config)
    val_dataloader = create_dataloader(config, "val")

    # define loss function and optimizer
    val_loss = LOSS_FUNCTIONS[config["validation"]["loss"]]()

    best_loss = sys.maxsize
    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        model = torch.load(config["model"]["weights"])

    # validation
    model.eval()
    val_running_loss = 0.0
    num_val_steps = 0
    epoch = 0

    for i, batch in enumerate(
        tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
    ):
        # target: y
        y: torch.Tensor = batch["y"].cuda(device)
        
        # mask
        y_mask: torch.Tensor = batch["y_mask"].cuda(device)
        y_sparse = (y * y_mask).float()
        
        # forward : p(y | y_sparse)
        outputs: torch.Tensor = model(y_sparse)
       
       # NOTE: standard loss (e.g., L1)
        loss = val_loss(outputs, y)
        
        # # NOTE: inpainting loss
        # loss = val_loss(predicted_image=outputs, target_image=y, mask=y_mask)
       
        val_running_loss += loss.item() * y_sparse.size(0)
        logger.log(
            **{
                "global_train_step": None,
                "global_val_step": len(val_dataloader) * (epoch) + i,
                "epoch": epoch,
                "train_loss": None,
                "val_loss": loss.item(),
            }
        )
        logger.save_sample(f"{i}.npy", y.squeeze(0), "ground-truth-current-maps")
        logger.save_sample(f"{i}.npy", outputs.squeeze(0), "predicted-current-maps")
        num_val_steps += 1

    # optionally log best/epoch model weights
    avg_val_loss = val_running_loss / num_val_steps
    if bool(config["logging"]["save_weights"]):
        if bool(config["logging"]["save_only_best_weights"]):
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                logger.save_weights(model, "best")
        else:
            logger.save_weights(model, f"epoch_{epoch}")


def train(config: dict) -> None:

    logger = setup_logger(config)
    model = create_model(config)
    train_dataloader = create_dataloader(config, "train")
    val_dataloader = create_dataloader(config, "val")

    # define loss function and optimizer
    train_loss: torch.nn.Module = LOSS_FUNCTIONS[config["training"]["loss"]]()
    val_loss: torch.nn.Module = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["training"]["optimizer"]](
        model.parameters(), lr=float(config["training"]["lr"])
    )

    best_loss = sys.maxsize
    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]

    model.cuda(device)
    model.float()

    # load weights from checkpoint
    if config["model"]["weights"] != None:
        # load weights only:
        # model.load_state_dict(torch.load(config["model"]["weights"]), strict=False)
        # load enitre model object:
        model = torch.load(config["model"]["weights"]).float().cuda()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):
            # feature: X
            # X: torch.Tensor = batch["X"].cuda(device)
            # target: y
            y: torch.Tensor = batch["y"].cuda(device)
            # mask
            y_mask: torch.Tensor = batch["y_mask"].cuda(device)
            y_sparse = (y * y_mask).float()
            # forward
            # zero gradients
            optimizer.zero_grad()
            # P(y | y_sparse)
            outputs = model(y_sparse)
            
            # # NOTE: standard loss (e.g., L1)
            # loss = train_loss(outputs, y)
            
            # NOTE: inpainting loss
            loss: torch.Tensor = train_loss(
                predicted_image=outputs, target_image=y, mask=y_mask
            )
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y_sparse.size(0)
            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + i,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": None,
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
        model.eval()
        val_running_loss = 0.0
        num_val_steps = 0

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):
                # feature: X
                # X: torch.Tensor = batch["X"].cuda(device)
                # target: y
                y: torch.Tensor = batch["y"].cuda(device)
                # mask
                y_mask: torch.Tensor = batch["y_mask"].cuda(device)
                y_sparse = (y * y_mask).float()
                # forward : p(y | y_sparse)
                outputs = model(y_sparse)
                
                # # NOTE: standard loss (e.g., L1)
                # loss = val_loss(outputs, y)
               
                # NOTE: inpainting loss
                loss = val_loss(predicted_image=outputs, target_image=y, mask=y_mask)

                val_running_loss += loss.item() * y_sparse.size(0)
                logger.log(
                    **{
                        "global_train_step": None,
                        "global_val_step": len(val_dataloader) * (epoch) + i,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": loss.item(),
                    }
                )
                num_val_steps += 1
                
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
            avg_val_loss = val_running_loss / num_val_steps
            if bool(config["logging"]["save_weights"]):
                if bool(config["logging"]["save_only_best_weights"]):
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logger.save_weights(model, "best")
                else:
                    logger.save_weights(model, f"epoch_{epoch}")


def main():
    config = parse_config(TRAIN_CONFIG_FP)
    if config["global"]["mode"] == "eval":
        eval(config)
    elif config["global"]["mode"] == "train":
        train(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
