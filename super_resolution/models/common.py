"""Shared building blocks for generator architectures (plan section 8: models/common.py)."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Paper SRResNet block: Conv3x3 + BN + ReLU, Conv3x3 + BN, residual addition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class UpsampleBlock(nn.Module):
    """One factor-of-2 upsampling stage: Conv3x3(C, 4C) + PixelShuffle(2) + activation."""

    def __init__(self, channels: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            activation if activation is not None else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def upsample_stages(channels: int, scale: int, activation: nn.Module | None = None) -> nn.Sequential:
    """log2(scale) consecutive factor-of-2 upsampling stages."""
    if scale & (scale - 1):
        raise ValueError(f"scale must be a power of 2, got {scale}")
    n_stages = scale.bit_length() - 1
    return nn.Sequential(*[UpsampleBlock(channels, activation) for _ in range(n_stages)])
