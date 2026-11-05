"""YCbCr-space losses: luma pyramid, chroma L1, weighted reconstruction (plan 5.8, 6.2)."""

import torch
import torch.nn as nn

from super_resolution.losses.pixel import L1Loss, LossOutput
from super_resolution.losses.pyramid import LaplacianPyramidLoss
from super_resolution.representations.color import RGBToYCbCr


class LumaPyramidLoss(nn.Module):
    """Laplacian-pyramid loss computed on the BT.601 luma channel of RGB inputs."""

    def __init__(
        self,
        n_levels: int = 3,
        band_weights: list[float] | None = None,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.to_ycbcr = RGBToYCbCr()
        self.pyramid_loss = LaplacianPyramidLoss(n_levels, band_weights, sigma)

    def forward(self, sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> LossOutput:
        sr_y = self.to_ycbcr(sr_rgb)[:, :1]
        hr_y = self.to_ycbcr(hr_rgb)[:, :1]
        output = self.pyramid_loss(sr_y, hr_y)
        output["terms"] = {f"luma_{k}": v for k, v in output["terms"].items()}
        return output


class ChromaL1Loss(nn.Module):
    """L1 on the centered Cb/Cr channels of RGB inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.to_ycbcr = RGBToYCbCr()
        self.l1 = L1Loss()

    def forward(self, sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> LossOutput:
        sr_chroma = self.to_ycbcr(sr_rgb)[:, 1:]
        hr_chroma = self.to_ycbcr(hr_rgb)[:, 1:]
        output = self.l1(sr_chroma, hr_chroma)
        output["terms"] = {"chroma_l1": output["terms"]["l1"]}
        return output


class YCbCrWeightedLoss(nn.Module):
    """lambda_y * L(Y) + lambda_cb * L(Cb) + lambda_cr * L(Cr); equal weights = control."""

    def __init__(
        self,
        lambda_y: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_cr: float = 1.0,
        inner_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.lambdas = (lambda_y, lambda_cb, lambda_cr)
        self.to_ycbcr = RGBToYCbCr()
        self.inner_loss = inner_loss if inner_loss is not None else L1Loss()

    def forward(self, sr_rgb: torch.Tensor, hr_rgb: torch.Tensor) -> LossOutput:
        sr_ycc = self.to_ycbcr(sr_rgb)
        hr_ycc = self.to_ycbcr(hr_rgb)
        total = sr_rgb.new_zeros(())
        terms: dict[str, torch.Tensor] = {}
        weight_sum = sum(self.lambdas)
        for i, (name, weight) in enumerate(zip(("y", "cb", "cr"), self.lambdas)):
            term = self.inner_loss(sr_ycc[:, i : i + 1], hr_ycc[:, i : i + 1])["total"]
            terms[name] = term
            total = total + weight * term
        terms["ycbcr_weighted"] = total / weight_sum
        return LossOutput(total=total / weight_sum, terms=terms)
