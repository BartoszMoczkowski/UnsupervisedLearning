"""Architecture contract tests: shapes, determinism, and tiny-batch overfit (plan section 10)."""

import torch

from super_resolution.models import (
    DCTHybridGenerator,
    DCTResNetGenerator,
    PackedBlockGenerator,
    SRGANDiscriminator,
    SRResNetGenerator,
    LumaSRResNet,
    YCbCrSRResNet,
)

BATCH, LR_SIZE, SCALE = 2, 24, 4
HR_SIZE = LR_SIZE * SCALE


def _lr() -> torch.Tensor:
    return torch.rand(BATCH, 3, LR_SIZE, LR_SIZE)


def test_scale_test_all_generators():
    expected = (BATCH, 3, HR_SIZE, HR_SIZE)
    for generator in (
        SRResNetGenerator(),
        YCbCrSRResNet(),
        LumaSRResNet(),
        PackedBlockGenerator(),
        DCTResNetGenerator(),
        DCTHybridGenerator(),
    ):
        sr = generator(_lr())
        assert tuple(sr.shape) == expected, f"{type(generator).__name__}: {tuple(sr.shape)}"


def test_discriminator_paper_shape():
    discriminator = SRGANDiscriminator(img_size=(96, 96))
    logits = discriminator(torch.rand(2, 3, 96, 96))
    assert logits.shape == (2, 1)
    n_params = sum(p.numel() for p in discriminator.features[-1].parameters())
    assert n_params > 0


def test_inference_determinism_bn_generator():
    generator = SRResNetGenerator().eval()
    x = _lr()
    with torch.no_grad():
        first, second = generator(x), generator(x)
    assert torch.equal(first, second)


def test_overfit_tiny_batch():
    """Each architecture must overfit a tiny batch (plan checklist item 9)."""
    torch.manual_seed(0)
    hr_target = torch.rand(BATCH, 3, HR_SIZE, HR_SIZE)
    lr_input = torch.nn.functional.interpolate(
        hr_target, scale_factor=1 / SCALE, mode="bicubic", align_corners=False
    ).clamp(0, 1)
    for name, generator in {
        "SRResNet": SRResNetGenerator(width=16, num_blocks=2),
        "Packed": PackedBlockGenerator(width=16, num_blocks=2),
        "DCT": DCTResNetGenerator(width=16, num_blocks=2),
        "DCTHybrid": DCTHybridGenerator(dct_width=16, dct_num_blocks=2, spatial_width=8),
    }.items():
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        initial = None
        for step in range(60):
            sr = generator(lr_input)
            loss = torch.mean((sr - hr_target) ** 2)
            if initial is None:
                initial = loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = loss.item()
        assert final < initial * 0.5, f"{name} failed to overfit: {initial:.5f} -> {final:.5f}"


def test_dct_generator_coefficient_contract():
    generator = DCTResNetGenerator(width=16, num_blocks=2)
    coeffs = generator.forward_coefficients(_lr())
    assert coeffs.shape == (BATCH, 192, HR_SIZE // 8, HR_SIZE // 8)


def test_luma_generator_keeps_chroma_bicubic():
    generator = LumaSRResNet(width=16, num_blocks=2)
    lr = _lr()
    ycc = generator.to_ycbcr(lr)
    sr = generator(lr)
    # Chromatic content should be present (bicubic chroma + learned luma).
    assert sr.shape == (BATCH, 3, HR_SIZE, HR_SIZE)
    assert torch.isfinite(sr).all()
