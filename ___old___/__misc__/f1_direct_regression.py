import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.sapphire import SapphireDataset
from util.logger import ExperimentLogger
from util.config import LOSS_FUNCTIONS, OPTIMIZERS

CONFIG_FP = (
    "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/config.yaml"
)


class RegressionHead(nn.Module):
    """
    Custom classification head used for predicting the final output value z.
    """

    def __init__(self, in_channels):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.fc1(x)


def parse_config(fp: str) -> dict:
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():

    config = parse_config(CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=CONFIG_FP,
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(config["logging"]["result_columns"])

    # create model + change classification head
    model: torch.nn.Module = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet152", pretrained=True
    )
    in_features = model.fc.in_features

    # change classification to have size 1
    model.fc = RegressionHead(in_features)

    # create train/val dataset and dataloader
    train_dataset = SapphireDataset(
        split="train",
        steps_per_epoch=config["training"]["steps_per_epoch"],
        device=config["global"]["device"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    val_dataset = SapphireDataset(
        split="val",
        steps_per_epoch=config["validation"]["steps_per_epoch"],
        device=config["global"]["device"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    # define loss function and optimizer
    train_loss = LOSS_FUNCTIONS[config["training"]["loss"]]()
    val_loss = LOSS_FUNCTIONS[config["validation"]["loss"]]()
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["training"]["optimizer"]](
        model.parameters(), lr=float(config["training"]["lr"])
    )

    num_epochs = config["training"]["epochs"]
    device = config["global"]["device"]
    model = model.cuda(device)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training: Epoch {epoch+1}/{num_epochs}")
        ):

            X: torch.Tensor = batch["X"]
            X = X.cuda(device)
            X_og = batch["X_og"]

            # original current map
            # (B, H, W, C)
            y_og: torch.Tensor = batch["y_og"]
            y_og = y_og.cuda(device)

            z: torch.Tensor = batch["z"]
            z = z.cuda(device)

            # zero gradients
            optimizer.zero_grad()

            # forward
            outputs = model(X)

            # TODO: do we need this?
            if z.dim() == 1:
                z = z.unsqueeze(1)

            loss = train_loss(outputs, z)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            logger.log(
                **{
                    "global_train_step": len(train_dataloader) * (epoch) + i,
                    "global_val_step": None,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": None,
                }
            )

            # optional sample logging
            if bool(config["logging"]["log_figures"]):
                logger.save_sample(X_og, epoch, name="train_X")
                logger.save_sample(y_og, epoch, name="train_y")

        # validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_dataloader, desc=f"Validation: Epoch {epoch+1}/{num_epochs}")
            ):

                # move inputs -> device
                X = batch["X"].to(device)
                X_og = batch["X_og"]
                y_og = batch["y_og"]
                z = batch["z"].to(device)

                # forward pass
                outputs = model(X)

                # ensure the target has the correct shape
                if z.dim() == 1:
                    z = z.unsqueeze(1)

                # compute loss
                loss = val_loss(outputs, z)

                # accumulate validation loss
                val_running_loss += loss.item() * X.size(0)
                logger.log(
                    **{
                        "global_train_step": None,
                        "global_val_step": len(val_dataloader) * (epoch) + i,
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": loss.item(),
                    }
                )

                if bool(config["logging"]["log_figures"]):
                    logger.save_sample(X_og, epoch, name="val_X")
                    logger.save_sample(y_og, epoch, name="val_y")


if __name__ == "__main__":
    main()
