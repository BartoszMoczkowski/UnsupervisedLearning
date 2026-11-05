"""Optional frequency-domain diagnostics (plan section 11)."""

import torch

from super_resolution.representations.dct import DEFAULT_BLOCK_SIZE, BlockDCT, unpack_blocks


def band_errors(
    sr: torch.Tensor, hr: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE
) -> dict[str, float]:
    """Mean absolute DC and grouped AC coefficient errors for DCT models."""
    from super_resolution.losses.dct import frequency_group_mask

    diff = BlockDCT(block_size).to(sr.device)(sr) - BlockDCT(block_size).to(hr.device)(hr)
    coeffs = unpack_blocks(diff, sr.shape[1])
    masks = frequency_group_mask(block_size, device=sr.device)
    return {
        group: torch.mean(torch.abs(coeffs[..., mask])).item() for group, mask in masks.items()
    }


def block_boundary_error(
    sr: torch.Tensor, hr: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE
) -> dict[str, float]:
    """MSE on block-boundary pixels vs block-interior pixels (plan: DCT artifact check)."""
    diff2 = (sr - hr) ** 2
    height, width = sr.shape[-2:]
    boundary_mask = torch.zeros(height, width, dtype=torch.bool)
    boundary_mask[::block_size, :] = True
    boundary_mask[:, ::block_size] = True
    boundary_mask = boundary_mask.to(sr.device)
    boundary = diff2[..., boundary_mask].mean()
    interior = diff2[..., ~boundary_mask].mean()
    return {"boundary_mse": boundary.item(), "interior_mse": interior.item()}
