"""Full-range BT.601 RGB <-> YCbCr conversion as differentiable nn.Modules.

Convention (see plan section 3.2):
- Input RGB is gamma-encoded, float, nominally in [0, 1].
- Y approximately in [0, 1]; Cb and Cr centered around zero, approximately in [-0.5, 0.5].
- Conversion matrices are persistent buffers so device/dtype move with the module.
"""

import torch
import torch.nn as nn
from einops import einsum

# Full-range BT.601, RGB in [0, 1] -> Y in [0, 1], Cb/Cr centered in [-0.5, 0.5].
_RGB_TO_YCBCR = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312],
    ],
    dtype=torch.float32,
)

_YCBCR_TO_RGB = torch.inverse(_RGB_TO_YCBCR)


class RGBToYCbCr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("transform", _RGB_TO_YCBCR.clone())

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor [B, 3, H, W], got {tuple(rgb.shape)}")
        matrix = self.transform.to(device=rgb.device, dtype=rgb.dtype)
        return einsum(matrix, rgb, "out in, b in h w -> b out h w")


class YCbCrToRGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("transform", _YCBCR_TO_RGB.clone())

    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        if ycbcr.ndim != 4 or ycbcr.shape[1] != 3:
            raise ValueError(f"Expected YCbCr tensor [B, 3, H, W], got {tuple(ycbcr.shape)}")
        matrix = self.transform.to(device=ycbcr.device, dtype=ycbcr.dtype)
        return einsum(matrix, ycbcr, "out in, b in h w -> b out h w")


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    return RGBToYCbCr().to(device=rgb.device)(rgb)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    return YCbCrToRGB().to(device=ycbcr.device)(ycbcr)
