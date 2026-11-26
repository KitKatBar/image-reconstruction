"""Small residual CNN for same-size image reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualCNN(nn.Module):
    """Predict a residual correction and add it to the pixelated input."""

    def __init__(self, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.head(x))
        features = self.body(features)
        residual = self.tail(features)
        return torch.clamp(x + residual, 0.0, 1.0)
