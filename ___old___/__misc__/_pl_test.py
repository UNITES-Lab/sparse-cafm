import os
import pytorch_lightning as pl
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class AE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def main():
    ae = AE()
    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)
    trainer = pl.Trainer(
        limit_train_batches=100, max_epochs=1, accelerator="gpu", devices=1
    )
    trainer.fit(model=ae, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
