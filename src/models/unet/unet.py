# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet.py
""" Full assembly of the parts to form the complete network """

import torch.utils
import torch.utils.checkpoint
from src.models.unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        # what if this was a vit-enc
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ... and this was a vit-dec
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes, activation=nn.Tanh())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    @staticmethod
    def get(weights=None):
        model = UNet(3, 3)
        return model


class ThickUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ThickUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        # what if this was a vit-enc
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ... and this was a vit-dec
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes, activation=nn.Tanh())

    def wide_forward(self, X, y_sparse):
        # [B, 3, H, W], [B, 3, H, W]
        x = torch.cat([X, y_sparse], dim=1)
        return self.forward(x)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    @staticmethod
    def get(weights=None):
        model = ThickUNet(6, 3)
        return model


class HieraUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Step 1: Reduce channels from 512 to something smaller
        self.conv_reduce = nn.Conv2d(512, 128, kernel_size=1)
        # Step 2: Upsample in stages
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)   # 28x28 -> 56x56
        # Step 3: Special handling to go 56x56 -> 64x64 (can do partial upsample + conv)
        self.conv_64 = nn.ConvTranspose2d(64, 64, kernel_size=9, stride=1, padding=1)
        # Final: map 64 channels to 3 channels
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)        # -> [B, 512, 14, 14]
        x = self.conv_reduce(x)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        # Maybe do an interpolation or partial upsample
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x = F.relu(self.conv_64(x))
        x = self.conv_out(x)
        x = F.tanh(x)
        return x

    @staticmethod
    def get(weights=None):
        model = HieraUNetDecoder()
        return model
