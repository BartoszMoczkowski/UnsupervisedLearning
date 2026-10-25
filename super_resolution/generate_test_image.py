#!/usr/bin/env python3
"""Visual test: upscale a random wikiart image with a trained SRGAN checkpoint.

Usage:
    python sr_visual_test.py --ckpt checkpoints/srgan.pt
    python sr_visual_test.py --ckpt checkpoints/srgan.pt --image path/to/img.jpg --scale 2
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2 as tv2

from generative_adversial_network import GeneratorNetwork

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to srgan.pt")
    ap.add_argument("--images", default="/data/wikiart-parquet-images/train")
    ap.add_argument("--image", default=None, help="specific image instead of random")
    ap.add_argument("--scale", type=int, default=2, help="must match training (hr_size / lr_size)")
    ap.add_argument("--max-side", type=int, default=512, help="cap input size for speed")
    ap.add_argument("--out", default="sr_test.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- pick image ---
    if args.image:
        path = Path(args.image)
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [p for p in Path(args.images).rglob("*") if p.suffix.lower() in exts]
        path = random.choice(files)
    print(f"testing on: {path}")

    img = tv2.functional.to_image(Image.open(path).convert("RGB"))
    img = tv2.functional.to_dtype(img, torch.float32, scale=True)  # CHW [0,1]

    # cap size, then make dims divisible by scale
    _, h, w = img.shape
    ratio = args.max_side / max(h, w)
    if ratio < 1.0:
        img = tv2.functional.resize(img, [round(h * ratio), round(w * ratio)], antialias=True)
    _, h, w = img.shape
    img = img[:, : h - h % args.scale, : w - w % args.scale]
    _, h, w = img.shape

    # --- synthesize LR exactly like training (antialiased downscale) ---
    lr = tv2.functional.resize(img, [h // args.scale, w // args.scale], antialias=True)

    # --- model ---
    generator = GeneratorNetwork(3, 16, 16, 32, n_residual_blocks=4, n_tail_blocks=1)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    generator.load_state_dict(ckpt["generator"])
    generator.to(device).eval()
    print(f"loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Match gan_train.py: scale [0, 1] to [-1, 1], then invert it.
    lr_norm = lr * 2.0 - 1.0
    with torch.no_grad():
        sr = generator(lr_norm.unsqueeze(0).to(device)).squeeze(0).cpu()
    print(
        f"input mean={lr_norm.mean().item():.4f}; raw SR "
        f"min={sr.min().item():.4f}, max={sr.max().item():.4f}, "
        f"mean={sr.mean().item():.4f}"
    )
    sr = ((sr + 1.0) / 2.0).clamp(0, 1)

    # bicubic baseline at the same target size
    bicubic = tv2.functional.resize(lr, [h, w], antialias=True)

    # --- save side-by-side: LR(nearest) | bicubic | SR | HR ---
    lr_up = tv2.functional.resize(
        lr, [h, w], interpolation=tv2.InterpolationMode.NEAREST
    )
    strip = torch.cat([lr_up, bicubic, sr, img], dim=2)  # concat along width
    tv2.functional.to_pil_image(strip).save(args.out)
    print(f"saved {args.out}  (LR nearest | bicubic | SR | HR), {4*w}x{h}")


if __name__ == "__main__":
    main()
