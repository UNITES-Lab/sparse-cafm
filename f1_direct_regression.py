from calendar import c
from math import e
import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from util.logger import ExperimentLogger
from datasets.sapphire import SapphireDataset
from torch.utils.data import DataLoader

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
    logger = ExperimentLogger(config_fp=CONFIG_FP, exp_name="f1_direct_regression")
    logger.add_result_columns(
        ["global_train_step", "global_val_step", "epoch", "train_loss", "val_loss"]
    )

    # create model + change classification head
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)
    in_features = model.fc.in_features

    # change classification to have size 1
    model.fc = RegressionHead(in_features)

    # create train/val dataset and dataloader
    train_dataset = SapphireDataset(
        split="train", steps_per_epoch=config["training"]["steps_per_epoch"]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )
    val_dataset = SapphireDataset(
        split="val", steps_per_epoch=config["validation"]["steps_per_epoch"]
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], shuffle=False
    )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))

    num_epochs = config["training"]["epochs"]
    device = 4
    model = model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_dataloader, desc="Batches")):

            X = batch["X"].to(device)
            z = batch["z"].to(device)
            optimizer.zero_grad()
            # forward
            outputs = model(X)
            # TODO: do we need this?
            if z.dim() == 1:
                z = z.unsqueeze(1)

            loss = criterion(outputs, z)
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

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for i, batch in enumerate(tqdm(val_dataloader, desc="Batches")):
                # Move inputs and targets to device
                X = batch["X"].to(device)
                z = batch["z"].to(device)
                # Forward pass
                outputs = model(X)
                # Ensure the target has the correct shape
                if z.dim() == 1:
                    z = z.unsqueeze(1)
                # Compute loss
                loss = criterion(outputs, z)
                # Accumulate validation loss
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

        # Compute average validation loss
        val_loss = val_running_loss / len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
