"""Orthonormal block DCT-II / DCT-III as differentiable, buffer-based modules.

Canonical packed representation (plan section 3.3):

    Input:  [B, C, H, W]           with H, W divisible by block_size
    Blocks: [B, C, H/8, W/8, 8, 8]
    DCT:    [B, C, H/8, W/8, 8, 8]
    Packed: [B, 64*C, H/8, W/8]    channel = c * 64 + u * 8 + v

where (u, v) indexes vertical and horizontal frequency.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, einsum

DEFAULT_BLOCK_SIZE = 8


def dct_matrix(block_size: int = DEFAULT_BLOCK_SIZE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal 1D DCT-II matrix D with D[u, n] = a_u cos(pi (2n+1) u / (2N))."""
    n = torch.arange(block_size, dtype=dtype)
    u = n.unsqueeze(1)  # [N, 1] frequency index
    mat = torch.cos(math.pi * (2 * n + 1) * u / (2 * block_size))  # [u, n]
    mat[0] *= math.sqrt(1.0 / block_size)
    mat[1:] *= math.sqrt(2.0 / block_size)
    return mat


def _blocks_from_image(images: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, C, H/8, W/8, 8, 8] without any transform."""
    height, width = images.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError(
            f"H={height} and W={width} must be divisible by block_size={block_size}"
        )
    return rearrange(
        images,
        "b c (bh bi) (bw bj) -> b c bh bw bi bj",
        bi=block_size,
        bj=block_size,
    )


def _image_from_blocks(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B, C, h, w, 8, 8] -> [B, C, H, W]; exact inverse of _blocks_from_image."""
    return rearrange(
        blocks,
        "b c bh bw bi bj -> b c (bh bi) (bw bj)",
        bi=block_size,
        bj=block_size,
    )


def pack_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """[B, C, h, w, 8, 8] -> [B, 64*C, h, w] with channel = c * 64 + u * 8 + v."""
    batch, channels, h, w, u, v = blocks.shape
    if u != v:
        raise ValueError("DCT blocks must be square")
    return rearrange(blocks, "b c h w u v -> b (c u v) h w")


def unpack_blocks(packed: torch.Tensor, channels: int) -> torch.Tensor:
    """[B, 64*C, h, w] -> [B, C, h, w, 8, 8]; exact inverse of pack_blocks."""
    block_area = packed.shape[1] // channels
    block_size = int(math.isqrt(block_area))
    if block_size * block_size != block_area:
        raise ValueError(f"Packed channel count {packed.shape[1]} is not 64*C")
    return rearrange(packed, "b (c u v) h w -> b c h w u v", c=channels, u=block_size)


class BlockDCT(nn.Module):
    """Differentiable orthonormal block DCT: [B, C, H, W] -> [B, 64*C, H/8, W/8].

    Coeff = D @ X @ D.T per block; the two einsum contractions keep batch, channel,
    and block-grid axes untouched.
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        super().__init__()
        self.block_size = block_size
        self.register_buffer("dct", dct_matrix(block_size))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        matrix = self.dct.to(device=images.device, dtype=images.dtype)
        blocks = _blocks_from_image(images, self.block_size)  # [b,c,bh,bw,i,j]
        tmp = einsum(matrix, blocks, "u i, b c bh bw i j -> b c bh bw u j")
        coeffs = einsum(matrix, tmp, "v j, b c bh bw u j -> b c bh bw u v")
        return pack_blocks(coeffs)


class InverseBlockDCT(nn.Module):
    """Differentiable inverse block DCT: [B, 64*C, H/8, W/8] -> [B, C, H, W].

    X = D.T @ Coeff @ D per block (DCT-III through the transpose matrix).
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, channels: int = 3) -> None:
        super().__init__()
        self.block_size = block_size
        self.channels = channels
        self.register_buffer("dct", dct_matrix(block_size))

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        matrix = self.dct.to(device=packed.device, dtype=packed.dtype)
        coeffs = unpack_blocks(packed, self.channels)  # [b,c,bh,bw,u,v]
        tmp = einsum(matrix, coeffs, "u i, b c bh bw u v -> b c bh bw i v")
        blocks = einsum(matrix, tmp, "v j, b c bh bw i v -> b c bh bw i j")
        return _image_from_blocks(blocks, self.block_size)


def block_dct(images: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    return BlockDCT(block_size).to(device=images.device)(images)


def inverse_block_dct(
    packed: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE, channels: int = 3
) -> torch.Tensor:
    return InverseBlockDCT(block_size, channels).to(device=packed.device)(packed)


def pixel_unshuffle_blocks(images: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """Pack spatial blocks into channels WITHOUT a DCT (A2 control input)."""
    return pack_blocks(_blocks_from_image(images, block_size))


def pixel_shuffle_blocks(
    packed: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE, channels: int = 3
) -> torch.Tensor:
    """Unpack channel-packed blocks into a spatial image WITHOUT a DCT."""
    return _image_from_blocks(unpack_blocks(packed, channels), block_size)
