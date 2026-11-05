"""DCT coefficient loss with DC / AC frequency-group weighting (plan section 5.5)."""

import torch
import torch.nn as nn

from super_resolution.losses.pixel import CharbonnierLoss, LossOutput
from super_resolution.representations.dct import DEFAULT_BLOCK_SIZE, BlockDCT, unpack_blocks


def frequency_group_mask(
    block_size: int = DEFAULT_BLOCK_SIZE,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Boolean masks [block, block] selecting DC, low-AC, mid-AC, high-AC coefficients."""
    u = torch.arange(block_size, device=device).view(-1, 1)
    v = torch.arange(block_size, device=device).view(1, -1)
    freq = u + v  # diagonal index: proxy for radial frequency
    return {
        "dc": (u == 0) & (v == 0),
        "low_ac": (freq >= 1) & (freq <= 4),
        "mid_ac": (freq >= 5) & (freq <= 9),
        "high_ac": (freq >= 10) & ~((u == 0) & (v == 0)),
    }


class DCTCoefficientLoss(nn.Module):
    """Compare packed block DCT coefficients of sr and hr with per-frequency-group weights.

    group_weights keys: "dc", "low_ac", "mid_ac", "high_ac". Uniform weights make this a
    transform sanity check only (orthonormal DCT MSE == spatial MSE up to scale).
    """

    GROUPS = ("dc", "low_ac", "mid_ac", "high_ac")

    def __init__(
        self,
        group_weights: dict[str, float] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        inner_loss: nn.Module | None = None,
        luma_only: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.luma_only = luma_only
        self.dct = BlockDCT(block_size)
        if group_weights is None:
            group_weights = {group: 1.0 for group in self.GROUPS}
        self.group_weights = group_weights
        self.inner_loss = inner_loss if inner_loss is not None else CharbonnierLoss()

    def _grouped_terms(
        self, packed_diff: torch.Tensor, channels: int
    ) -> dict[str, torch.Tensor]:
        coeffs = unpack_blocks(packed_diff, channels)  # [B, C, h, w, u, v]
        masks = frequency_group_mask(self.block_size, device=packed_diff.device)
        terms = {}
        for group, weight in self.group_weights.items():
            if weight == 0:
                continue
            mask = masks[group]
            selected = coeffs[..., mask]  # [B, C, h, w, k]
            terms[group] = self.inner_loss(selected, torch.zeros_like(selected))["total"]
        return terms

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        if sr.shape != hr.shape:
            raise ValueError(f"sr and hr shapes must match, got {tuple(sr.shape)} vs {tuple(hr.shape)}")
        sr_dct = self.dct(sr)
        hr_dct = self.dct(hr)
        diff = sr_dct - hr_dct
        terms = self._grouped_terms(diff, sr.shape[1])
        total = diff.new_zeros(())
        weight_sum = 0.0
        for group, term in terms.items():
            total = total + self.group_weights[group] * term
            weight_sum += self.group_weights[group]
        terms["dct"] = total / weight_sum
        return LossOutput(total=total / weight_sum, terms=terms)
