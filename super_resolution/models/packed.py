"""A2: packed-block control network — pixel unshuffle into 64C channels, NO DCT (plan 4.3)."""

import torch
import torch.nn as nn

from super_resolution.models.common import ResidualBlock, upsample_stages
from super_resolution.representations.dct import DEFAULT_BLOCK_SIZE, pixel_shuffle_blocks, pixel_unshuffle_blocks


class PackedBlockGenerator(nn.Module):
    """Rearranges LR 8x8 blocks into 64*C channels, runs a residual trunk on the block
    grid, expands the grid by r, and unpacks directly into an HR spatial image."""

    def __init__(
        self,
        channels: int = 3,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.block_size = block_size
        packed_channels = channels * block_size * block_size
        self.input_projection = nn.Conv2d(packed_channels, width, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(width, packed_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        packed = pixel_unshuffle_blocks(lr_rgb, self.block_size)
        features = self.trunk(self.input_projection(packed))
        hr_packed = self.head(features)
        return pixel_shuffle_blocks(hr_packed, self.block_size, self.channels)
