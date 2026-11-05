"""Gaussian and Laplacian pyramids with fixed separable kernels (plan section 5.4).

Both are buffer-based, parameter-free nn.Modules. Levels are ordered fine-to-coarse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float = 1.0, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class GaussianPyramid(nn.Module):
    """Fixed depthwise-separable Gaussian blur + stride-2 downsampling."""

    def __init__(self, n_levels: int, sigma: float = 1.0, radius: int | None = None) -> None:
        super().__init__()
        self.n_levels = n_levels
        kernel = gaussian_kernel1d(sigma, radius)
        self.register_buffer("kernel", kernel)

    def _blur(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        pad = self.kernel.numel() // 2
        kernel_h = self.kernel.to(dtype=images.dtype).view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        kernel_w = self.kernel.to(dtype=images.dtype).view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        padded = F.pad(images, (pad, pad, 0, 0), mode="reflect")
        padded = F.pad(padded, (0, 0, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel_h, groups=channels)
        return F.conv2d(blurred, kernel_w, groups=channels)

    def downsample(self, images: torch.Tensor) -> torch.Tensor:
        """Blur, then stride-2 subsample. Spurious last row/col of odd sizes is dropped."""
        blurred = self._blur(images)
        return blurred[:, :, ::2, ::2]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return n_levels Gaussian levels, fine to coarse."""
        levels = [images]
        for _ in range(self.n_levels - 1):
            levels.append(self.downsample(levels[-1]))
        return levels


class LaplacianPyramid(nn.Module):
    """Laplacian bands, fine to coarse; the final coarse Gaussian level is returned as-is."""

    def __init__(
        self,
        n_levels: int,
        sigma: float = 1.0,
        radius: int | None = None,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.gaussian = GaussianPyramid(n_levels, sigma, radius)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        gauss = self.gaussian(images)
        bands = []
        for level in range(len(gauss) - 1):
            upsampled = F.interpolate(
                gauss[level + 1],
                size=gauss[level].shape[-2:],
                mode=self.upsample_mode,
                align_corners=False,
            )
            bands.append(gauss[level] - upsampled)
        bands.append(gauss[-1])  # final coarse level compared directly
        return bands
