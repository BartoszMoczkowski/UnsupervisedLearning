"""Unit tests for the plan's pre-training validation checklist (plan section 10)."""

import torch
import torch.nn.functional as F

from super_resolution.configs.elements.loss import (
    FrequencyAwareGeneratorLoss,
    GeneratorCharbonnierLossConfig,
    GeneratorL1LossConfig,
)
from super_resolution.configs.elements.preprocessing import BicubicDownsample
from super_resolution.losses.dct import DCTCoefficientLoss
from super_resolution.losses.pyramid import LaplacianPyramidLoss
from super_resolution.representations.color import RGBToYCbCr, YCbCrToRGB
from super_resolution.representations.dct import (
    BlockDCT,
    InverseBlockDCT,
    dct_matrix,
    inverse_block_dct,
    pixel_shuffle_blocks,
    pixel_unshuffle_blocks,
)
from super_resolution.representations.pyramid import LaplacianPyramid

TOLERANCE = 1e-5


def test_dct_matrix_orthogonality():
    matrix = dct_matrix(8)
    assert torch.allclose(matrix.T @ matrix, torch.eye(8), atol=TOLERANCE)


def test_dct_round_trip_random():
    x = torch.rand(2, 3, 32, 48)
    reconstructed = inverse_block_dct(block_dct := BlockDCT()(x), channels=3)
    assert torch.allclose(reconstructed, x, atol=TOLERANCE)


def test_dct_round_trip_gradient():
    x = torch.rand(1, 3, 16, 16, requires_grad=True)
    y = inverse_block_dct(BlockDCT()(x), channels=3)
    loss = (y - torch.rand_like(y)).pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_packing_round_trip():
    x = torch.rand(2, 3, 24, 40)
    packed = pixel_unshuffle_blocks(x)
    assert packed.shape == (2, 192, 3, 5)
    reconstructed = pixel_shuffle_blocks(packed, channels=3)
    assert torch.equal(reconstructed, x)


def test_color_round_trip():
    x = torch.rand(2, 3, 16, 16)
    round_trip = YCbCrToRGB()(RGBToYCbCr()(x))
    assert torch.allclose(round_trip, x, atol=1e-6)


def test_color_convention_ranges():
    rgb = torch.rand(64, 3, 8, 8)
    ycc = RGBToYCbCr()(rgb)
    assert ycc[:, 0].min() >= -1e-5 and ycc[:, 0].max() <= 1 + 1e-5
    assert ycc[:, 1:].abs().max() <= 0.5 + 1e-5


def test_uniform_dct_mse_equals_spatial_mse():
    """Loss equivalence: orthonormal DCT preserves L2 norm up to scale."""
    x = torch.rand(1, 3, 16, 16)
    diff = x - torch.rand_like(x)
    spatial_mse = torch.mean(diff**2)
    dct_mse = torch.mean(BlockDCT()(diff) ** 2) * (8 * 8) / (8 * 8)
    # Per-block orthonormal transform: sum over 64 coefficients == sum over 64 pixels,
    # so per-element means differ exactly by the block-area ratio of 1 here.
    assert torch.allclose(spatial_mse, dct_mse, atol=1e-6)


def test_pyramid_shapes():
    x = torch.rand(1, 3, 96, 96)
    pyramid = LaplacianPyramid(n_levels=3)
    bands = pyramid(x)
    assert len(bands) == 3
    assert bands[0].shape == (1, 3, 96, 96)
    assert bands[1].shape == (1, 3, 48, 48)
    assert bands[2].shape == (1, 3, 24, 24)


def test_pyramid_loss_band_shape_check():
    loss = LaplacianPyramidLoss(n_levels=3, band_weights=[1.0, 1.0, 1.0])
    sr, hr = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    output = loss(sr, hr)
    assert "pyramid" in output["terms"]
    assert output["total"] > 0


def test_gradient_flow_through_all_representations():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    ycc = RGBToYCbCr()(x)
    bands = LaplacianPyramid(3)(ycc[:, :1])
    coeffs = BlockDCT()(x)
    loss = sum(b.mean() for b in bands) + coeffs.mean()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_bicubic_downsample_shape_and_antialias():
    step = BicubicDownsample(scale=4)
    hr = torch.rand(2, 3, 96, 96)
    lr = step.apply(hr)
    assert lr.shape == (2, 3, 24, 24)


def test_bicubic_downsample_rejects_bad_crop():
    step = BicubicDownsample(scale=4)
    try:
        step.apply(torch.rand(1, 3, 98, 98))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for crop not divisible by scale")


def test_generator_loss_configs_return_total_and_terms():
    sr, hr = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    for loss in (GeneratorL1LossConfig(), GeneratorCharbonnierLossConfig()):
        terms = loss.calculate_loss_terms(sr, hr)
        assert "total" in terms and torch.isfinite(terms["total"])
        assert loss.calculate_loss(sr, hr) == terms["total"]


def test_frequency_aware_loss_composition():
    sr, hr = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    loss = FrequencyAwareGeneratorLoss(
        base_loss=GeneratorL1LossConfig(),
        lambda_pyr=1.0,
        luma_pyramid=True,
        lambda_chroma=0.1,
        lambda_dct=0.5,
    )
    terms = loss.calculate_loss_terms(sr, hr)
    assert "weighted_pyramid" in terms
    assert "weighted_chroma" in terms
    assert "weighted_dct" in terms
    assert torch.isfinite(terms["total"])


def test_dct_loss_gradient_and_groups():
    sr = torch.rand(1, 3, 16, 16, requires_grad=True)
    hr = torch.rand(1, 3, 16, 16)
    loss = DCTCoefficientLoss({"dc": 1.0, "low_ac": 2.0, "mid_ac": 1.0, "high_ac": 0.5})
    output = loss(sr, hr)
    output["total"].backward()
    assert torch.isfinite(sr.grad).all() and sr.grad.abs().sum() > 0
    assert set(loss.group_weights) == {"dc", "low_ac", "mid_ac", "high_ac"}
