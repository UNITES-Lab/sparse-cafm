import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# 2) Down-sampling Block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


# 3) Up-sampling Block
#    We use bilinear upsampling to reach the desired scale_factor,
#    then follow with a DoubleConv. Optionally, we can accept a skip connection.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # We use bilinear upsampling for more flexible scaling
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip=None):
        # 1) Upsample
        x = self.up(x)
        # 2) If a skip connection is provided, concatenate along channel dimension
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        # 3) DoubleConv
        x = self.conv(x)
        return x


class HieraUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution block to reduce from 512 -> 256 at 14x14
        self.inc = DoubleConv(512, 256)
        # Down to 7x7
        self.down1 = Down(256, 256)
        # Bottleneck at 7x7
        self.bottleneck = DoubleConv(256, 512)
        # Up: from 7 -> 14 (skip from down1 output's DoubleConv)
        self.up1 = Up(512, 256, scale_factor=2)
        # Up: 14 -> 28
        self.up2 = Up(256, 128, scale_factor=2)
        # Up: 28 -> 56
        self.up3 = Up(128, 64, scale_factor=2)
        # Up: 56 -> 64 (custom scale_factor = 64/56)
        self.up4 = Up(64, 32, scale_factor=(64 / 56))
        # Final 1×1 conv to get 3 output channels
        self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)           # -> [B, 512, 14, 14]
        # 1) Initial conv at 14x14
        x1 = self.inc(x)                    # (B, 256, 14, 14)
        # 2) Down to 7x7
        x2 = self.down1(x1)                 # (B, 256, 7, 7)
        # 3) Bottleneck still at 7x7
        x3 = self.bottleneck(x2)            # (B, 512, 7, 7)
        # 4) Up to 14x14, skip with x2
        x4 = self.up1(x3) + x1              # (B, 256, 14, 14)
        # 5) Up to 28x28
        x5 = self.up2(x4)                   # (128, 28, 28)
        # 6) Up to 56x56
        x6 = self.up3(x5)                   # (64, 56, 56)
        # 7) Up to 64x64
        x7 = self.up4(x6)                   # (32, 64, 64)
        # 8) Final 3-channel output
        out = self.outc(x7)                 # (3, 64, 64)
        out = nn.functional.tanh(out)
        return out
    
    @staticmethod
    def get(weights=None):
        model = HieraUNetDecoder()
        return model