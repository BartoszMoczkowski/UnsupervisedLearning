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

from super_resolution.configs.elements.stage import Stage
from super_resolution.configs.elements.preprocessing import BicubicDownsample


def _pick_image(images_root: Path, specific: str | None) -> Path:
    if specific:
        return Path(specific)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in images_root.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images under {images_root}")
    return random.choice(files)


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
    scale = getattr(stage.generator_model.model_config, "scale", 4)

    path = _pick_image(args.images, args.image)
    print(f"testing on: {path}")

    img = tv2.functional.to_image(Image.open(path).convert("RGB"))
    img = tv2.functional.to_dtype(img, torch.float32, scale=True)  # CHW [0,1]

    # Cap size, then make dims divisible by scale.
    _, h, w = img.shape
    ratio = args.max_side / max(h, w)
    if ratio < 1.0:
        img = tv2.functional.resize(img, [round(h * ratio), round(w * ratio)], antialias=True)
    _, h, w = img.shape
    img = img[:, : h - h % scale, : w - w % scale]
    _, h, w = img.shape

    # Synthesize LR exactly like training (bicubic, antialiased, float [0,1]).
    lr = BicubicDownsample(scale=scale).apply(img.unsqueeze(0)).squeeze(0)

    generator = stage.generator_model.model_config.build().to(device)
    state = stage.load_state()
    generator.load_state_dict(state["generator"])
    generator.eval()

    with torch.no_grad():
        sr = generator(lr.unsqueeze(0).to(device)).squeeze(0).cpu()
    print(
        f"raw SR min={sr.min().item():.4f}, max={sr.max().item():.4f}, mean={sr.mean().item():.4f}"
    )
    sr = sr.clamp(0, 1)

    # Bicubic baseline at the same target size.
    bicubic = tv2.functional.resize(lr, [h, w], antialias=True)

    # Save side-by-side: LR (nearest) | bicubic | SR | HR.
    lr_up = tv2.functional.resize(lr, [h, w], interpolation=tv2.InterpolationMode.NEAREST)
    strip = torch.cat([lr_up, bicubic, sr, img], dim=2)
    tv2.functional.to_pil_image(strip).save(args.out)
    print(f"saved {args.out}  (LR nearest | bicubic | SR | HR), {4 * w}x{h}")


if __name__ == "__main__":
    main()
