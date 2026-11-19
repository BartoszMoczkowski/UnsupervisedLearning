#!/usr/bin/env python3
"""Visual test: upscale an image with a trained generator from a Stage checkpoint.

Usage:
    python -m super_resolution.generate_test_image --stage-config experiments_data/<stage_id>/<name>_config.pkl
    python -m super_resolution.generate_test_image --stage-config ... --image path/to/img.jpg --out result.png
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2 as tv2

from super_resolution.configs.elements.preprocessing import (
    BicubicDownsample,
    PreprocessingStep,
)
from super_resolution.configs.elements.stage import Stage


def _pick_image(images_root: Path, specific: str | None) -> Path:
    if specific:
        return Path(specific)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in images_root.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images under {images_root}")
    return random.choice(files)


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        img = tv2.functional.to_image(source.convert("RGB"))
    return tv2.functional.to_dtype(img, torch.float32, scale=True)  # CHW [0,1]


def _prepare_inputs(
    img: torch.Tensor,
    model_config,
    lr_preprocessing: PreprocessingStep | None,
    max_side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned HR and model-ready LR CHW tensors.

    Block-based models (DCT and packed) require LR dimensions divisible by their
    block size. Aligning HR to scale * block_size satisfies both that constraint
    and the ordinary spatial models' scale constraint.
    """
    if max_side <= 0:
        raise ValueError("max_side must be greater than zero")

    scale = getattr(model_config, "scale", 4)
    block_size = getattr(model_config, "block_size", 1)
    hr_multiple = scale * block_size

    # Cap size, then make HR dims divisible by both scale and any LR block size.
    _, h, w = img.shape
    ratio = max_side / max(h, w)
    if ratio < 1.0:
        img = tv2.functional.resize(
            img, [round(h * ratio), round(w * ratio)], antialias=True
        )
    _, h, w = img.shape
    aligned_h = h - h % hr_multiple
    aligned_w = w - w % hr_multiple
    if aligned_h == 0 or aligned_w == 0:
        raise ValueError(
            f"Image is too small for {type(model_config).__name__}: HR dimensions "
            f"must be at least {hr_multiple}x{hr_multiple}, got {h}x{w}"
        )
    img = img[:, :aligned_h, :aligned_w]

    # Match the configured training LR pipeline. The fallback supports older
    # saved Stage configs that predate explicit LR preprocessing.
    if lr_preprocessing is not None:
        lr = lr_preprocessing.apply(img.unsqueeze(0)).squeeze(0)
    else:
        lr = BicubicDownsample(scale=scale).apply(img.unsqueeze(0)).squeeze(0)
    return img, lr


def generate_test_strip(
    stage: Stage,
    generator: torch.nn.Module,
    img: torch.Tensor,
    max_side: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Build LR | bicubic | SR | HR and return it with raw SR and HR size."""
    img, lr = _prepare_inputs(
        img,
        stage.generator_model.model_config,
        stage.lr_preprocessing,
        max_side,
    )
    _, h, w = img.shape

    with torch.no_grad():
        sr = generator(lr.unsqueeze(0).to(device)).squeeze(0).cpu()

    # Bicubic baseline at the same target size.
    bicubic = tv2.functional.resize(lr, [h, w], antialias=True)

    # Save side-by-side: LR (nearest) | bicubic | SR | HR.
    lr_up = tv2.functional.resize(lr, [h, w], interpolation=tv2.InterpolationMode.NEAREST)
    strip = torch.cat([lr_up, bicubic, sr.clamp(0, 1), img], dim=2)
    return strip, sr, (h, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-config", type=Path, required=True, help="path to a saved Stage config pickle")
    ap.add_argument("--images", type=Path, default=Path("/data/wikiart-parquet-images/train"))
    ap.add_argument("--image", default=None, help="specific image instead of random")
    ap.add_argument("--max-side", type=int, default=512, help="cap input size for speed")
    ap.add_argument("--out", type=Path, default=Path("sr_test.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage: Stage = Stage.from_stage_config(args.stage_config)

    path = _pick_image(args.images, args.image)
    print(f"testing on: {path}")
    img = _load_image(path)

    generator = stage.generator_model.model_config.build().to(device)
    state = stage.load_state()
    generator.load_state_dict(state["generator"])
    generator.eval()

    strip, sr, (h, w) = generate_test_strip(stage, generator, img, args.max_side, device)
    print(
        f"raw SR min={sr.min().item():.4f}, max={sr.max().item():.4f}, mean={sr.mean().item():.4f}"
    )
    tv2.functional.to_pil_image(strip).save(args.out)
    print(f"saved {args.out}  (LR nearest | bicubic | SR | HR), {4 * w}x{h}")


if __name__ == "__main__":
    main()
