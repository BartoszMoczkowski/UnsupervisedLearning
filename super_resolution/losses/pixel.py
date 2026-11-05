"""Pixel-domain reconstruction losses (plan section 5.1-5.3).

Each returns a dict with the total and every unweighted term (plan section 5).
"""

from typing import TypedDict

import torch
import torch.nn as nn


class LossOutput(TypedDict):
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class MSELoss(nn.Module):
    """L_mse = mean((sr - hr)^2)."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        term = torch.mean((sr - hr) ** 2)
        return LossOutput(total=term, terms={"mse": term})


class L1Loss(nn.Module):
    """L_l1 = mean(abs(sr - hr))."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        term = torch.mean(torch.abs(sr - hr))
        return LossOutput(total=term, terms={"l1": term})


class CharbonnierLoss(nn.Module):
    """L_char = mean(sqrt((sr - hr)^2 + eps^2)), eps = 1e-3 for images in [0, 1]."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        term = torch.mean(torch.sqrt((sr - hr) ** 2 + self.eps**2))
        return LossOutput(total=term, terms={"charbonnier": term})
