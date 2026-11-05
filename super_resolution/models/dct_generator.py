"""A3: pure DCT-domain generator and A4: hybrid DCT + spatial refinement (plan 4.3, 4.4)."""

import torch
import torch.nn as nn

from super_resolution.models.common import ResidualBlock
from super_resolution.representations.dct import (
    DEFAULT_BLOCK_SIZE,
    BlockDCT,
    InverseBlockDCT,
)


class DCTResidualBlock(nn.Module):
    """Residual block on the block grid: 1x1 mixes frequencies/channels, 3x3 mixes
    neighboring blocks (plan 4.3)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class DCTResNetGenerator(nn.Module):
    """A3: LR -> block DCT -> packed-coefficient trunk -> HR coefficients -> inverse DCT.

    The head maps each LR block-grid location to an r x r neighborhood of HR blocks;
    packed channels survive PixelShuffle because the channel order per block is preserved.
    """

    def __init__(
        self,
        channels: int = 3,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
        coefficient_groups: tuple[str, ...] = ("dc", "ac"),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.block_size = block_size
        packed_channels = channels * block_size * block_size
        self.dct = BlockDCT(block_size)
        self.inverse_dct = InverseBlockDCT(block_size, channels)
        self.input_projection = nn.Conv2d(packed_channels, width, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[DCTResidualBlock(width) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(width, packed_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward_coefficients(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """LR RGB -> packed predicted HR DCT coefficients [B, 64C, H/8, W/8]."""
        packed = self.dct(lr_rgb)
        features = self.trunk(self.input_projection(packed))
        return self.head(features)

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return self.inverse_dct(self.forward_coefficients(lr_rgb))


class DCTHybridGenerator(nn.Module):
    """A4: DCT trunk -> inverse DCT -> 1-3 lightweight spatial residual blocks (plan 4.4)."""

    def __init__(
        self,
        channels: int = 3,
        dct_width: int = 64,
        dct_num_blocks: int = 16,
        spatial_width: int = 32,
        n_spatial_blocks: int = 2,
        scale: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        if not 1 <= n_spatial_blocks <= 3:
            raise ValueError("The spatial tail should be 1-3 lightweight blocks")
        self.dct_generator = DCTResNetGenerator(channels, dct_width, dct_num_blocks, scale, block_size)
        self.spatial_refinement = nn.Sequential(
            nn.Conv2d(channels, spatial_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(spatial_width) for _ in range(n_spatial_blocks)],
            nn.Conv2d(spatial_width, channels, kernel_size=3, padding=1),
        )

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        preliminary = self.dct_generator(lr_rgb)
        return preliminary + self.spatial_refinement(preliminary)
