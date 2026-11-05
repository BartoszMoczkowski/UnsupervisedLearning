"""Laplacian-pyramid loss (plan section 5.4)."""

import torch
import torch.nn as nn

from super_resolution.losses.pixel import CharbonnierLoss, LossOutput
from super_resolution.representations.pyramid import LaplacianPyramid


class LaplacianPyramidLoss(nn.Module):
    """Sum of band-wise robust losses, normalized by band element counts.

    band_weights are stored and accepted fine-to-coarse to avoid ambiguity.
    """

    def __init__(
        self,
        n_levels: int = 3,
        band_weights: list[float] | None = None,
        sigma: float = 1.0,
        inner_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pyramid = LaplacianPyramid(n_levels, sigma)
        if band_weights is None:
            band_weights = [1.0] * n_levels
        if len(band_weights) != n_levels:
            raise ValueError("band_weights must have one entry per pyramid level")
        self.band_weights = band_weights
        self.inner_loss = inner_loss if inner_loss is not None else CharbonnierLoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        sr_bands = self.pyramid(sr)
        hr_bands = self.pyramid(hr)
        total = sr.new_zeros(())
        terms: dict[str, torch.Tensor] = {}
        weight_sum = sum(self.band_weights)
        for i, (sr_band, hr_band, weight) in enumerate(
            zip(sr_bands, hr_bands, self.band_weights)
        ):
            if sr_band.shape != hr_band.shape:
                raise ValueError(
                    f"Pyramid band {i} shape mismatch: {tuple(sr_band.shape)} vs {tuple(hr_band.shape)}"
                )
            band_loss = self.inner_loss(sr_band, hr_band)["total"]
            terms[f"pyramid_band_{i}"] = band_loss
            total = total + weight * band_loss
        terms["pyramid"] = total / weight_sum
        return LossOutput(total=total / weight_sum, terms=terms)
