import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Handle possible odd image dimensions.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = nn.functional.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.input_block = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.bottleneck = DoubleConv(c4, c4 * 2)

        self.up1 = UpBlock(c4 * 2, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

        self.output_layer = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        skip1 = self.input_block(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)

        x = self.bottleneck(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return torch.sigmoid(self.output_layer(x))