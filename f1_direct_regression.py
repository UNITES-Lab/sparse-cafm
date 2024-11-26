import torch
import torch.nn as nn

from tqdm import tqdm
from datasets.sapphire import SapphireDataset
from torch.utils.data import DataLoader


class RegressionHead(nn.Module):
    """
    Custom classification head used for predicting the final output value z.
    """

    def __init__(self, in_channels):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.fc1(x)


def main():

    # create model + change classification head
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)
    in_features = model.fc.in_features

    # change classification to have size 1
    model.fc = RegressionHead(in_features)

    # create train/val dataset and dataloader
    train_dataset = SapphireDataset(split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataset = SapphireDataset(split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    device = 4
    model = model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc="Batches"):
            # Move inputs and targets to device
            X = batch["X"].to(device)
            z = batch["z"].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X)
            # Ensure the target has the correct shape
            if z.dim() == 1:
                z = z.unsqueeze(1)
            # Compute loss
            loss = criterion(outputs, z)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Accumulate running loss
            running_loss += loss.item() * X.size(0)

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(val_dataloader, desc="Batches"):
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

        # Compute average validation loss
        val_loss = val_running_loss / len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
