"""A0: paper-faithful SRResNet generator and its input/output variants (plan section 4.1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_resolution.models.common import ResidualBlock, upsample_stages
from super_resolution.representations.color import RGBToYCbCr, YCbCrToRGB


class SRResNetGenerator(nn.Module):
    """Paper-style SRResNet: 16 residual blocks, two PixelShuffle(2) stages for r = 4."""

    def __init__(
        self,
        channels_input: int = 3,
        channels_output: int = 3,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels_input, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            *[ResidualBlockNorm(width, normalize) for _ in range(num_blocks)]
        )
        post_norm: nn.Module = nn.BatchNorm2d(width) if normalize else nn.Identity()
        self.post_residual = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            post_norm,
        )
        self.upsample = upsample_stages(width, scale)
        self.out_layer = nn.Conv2d(width, channels_output, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_x = self.head(x)
        trunk_x = head_x + self.post_residual(self.residual(head_x))
        return self.out_layer(self.upsample(trunk_x))


class ResidualBlockNorm(nn.Module):
    """Conv3x3 + (BN) + ReLU, Conv3x3 + (BN), residual addition; BN per paper."""

    def __init__(self, channels: int, normalize: bool = True) -> None:
        super().__init__()
        norm: type[nn.Module] = nn.BatchNorm2d if normalize else nn.Identity
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class YCbCrSRResNet(nn.Module):
    """A0-YCbCr: full-resolution YCbCr input/output, converted back to RGB."""

    def __init__(self, width: int = 64, num_blocks: int = 16, scale: int = 4) -> None:
        super().__init__()
        self.generator = SRResNetGenerator(3, 3, width, num_blocks, scale)
        self.to_ycbcr = RGBToYCbCr()
        self.to_rgb = YCbCrToRGB()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.to_rgb(self.generator(self.to_ycbcr(rgb)))


class LumaSRResNet(nn.Module):
    """A0-Y: luma-only SR; chroma is bicubic-upsampled and recombined to RGB."""

    def __init__(self, width: int = 64, num_blocks: int = 16, scale: int = 4) -> None:
        super().__init__()
        self.generator = SRResNetGenerator(1, 1, width, num_blocks, scale)
        self.to_ycbcr = RGBToYCbCr()
        self.to_rgb = YCbCrToRGB()
        self.scale = scale

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        ycbcr = self.to_ycbcr(rgb)
        sr_y = self.generator(ycbcr[:, :1])
        sr_chroma = F.interpolate(
            ycbcr[:, 1:],
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        return self.to_rgb(torch.cat([sr_y, sr_chroma], dim=1))
