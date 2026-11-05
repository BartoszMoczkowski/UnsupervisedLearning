"""PSNR / SSIM on RGB and BT.601 Y with optional border shave (plan section 11)."""

import torch

from super_resolution.representations.color import RGBToYCbCr

_SSIM_KERNEL = torch.tensor(
    [
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0],
    ]
) / 273.0


def _shave(images: torch.Tensor, shave_border: int) -> torch.Tensor:
    if shave_border <= 0:
        return images
    return images[..., shave_border:-shave_border, shave_border:-shave_border]


def psnr(sr: torch.Tensor, hr: torch.Tensor, shave_border: int = 4) -> torch.Tensor:
    """Paper-comparable: Y PSNR with a 4-pixel border removed at r = 4 (data range 1)."""
    sr = _shave(sr, shave_border)
    hr = _shave(hr, shave_border)
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=sr.device)
    return -10.0 * torch.log10(mse)


def _gaussian_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window = _SSIM_KERNEL.to(device=device, dtype=dtype)
    return window.expand(channels, 1, 5, 5)


def ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    shave_border: int = 4,
    data_range: float = 1.0,
) -> torch.Tensor:
    """One consistent SSIM implementation (Gaussian 11x11 separable via 5x5 conv, K1/K2 standard)."""
    sr = _shave(sr, shave_border)
    hr = _shave(hr, shave_border)
    channels = sr.shape[1]
    window = _gaussian_window(channels, sr.device, sr.dtype)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def filter(x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), window, groups=channels)

    mu_x, mu_y = filter(sr), filter(hr)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filter(sr * sr) - mu_x2
    sigma_y = filter(hr * hr) - mu_y2
    sigma_xy = filter(sr * hr) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


_to_ycbcr = RGBToYCbCr()


def y_psnr(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor, shave_border: int = 4) -> torch.Tensor:
    return psnr(_to_ycbcr.to(sr_rgb.device)(sr_rgb)[:, :1], _to_ycbcr.to(hr_rgb.device)(hr_rgb)[:, :1], shave_border)


def y_ssim(sr_rgb: torch.Tensor, hr_rgb: torch.Tensor, shave_border: int = 4) -> torch.Tensor:
    return ssim(_to_ycbcr.to(sr_rgb.device)(sr_rgb)[:, :1], _to_ycbcr.to(hr_rgb.device)(hr_rgb)[:, :1], shave_border)
