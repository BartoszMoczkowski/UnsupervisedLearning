#!/usr/bin/env python3
"""Extract JPEGs from the downloaded wikiart parquet shards to disk, pre-scaled.

Reads parquet from TRAINING_DATA_DIR (default ./wikiart-parquet),
writes images to <TRAINING_DATA_DIR>-images/train/NNNNNNN.jpg,
downscaled so the SHORTER side is 300px (aspect ratio preserved).
Resumable: skips images that already exist.
"""

import os
import glob
import polars as pl
from io import BytesIO
from pathlib import Path
from PIL import Image

OUT_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
IMG_DIR = Path(os.environ.get("IMAGES_DIR", OUT_DIR + "-images"))
BATCH = 512              # rows decoded per step; lower if RAM is tight
TARGET_MIN_SIDE = 300    # shorter side after scaling
JPEG_QUALITY = 90


def downscale(b: bytes) -> bytes:
    with Image.open(BytesIO(b)) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = TARGET_MIN_SIDE / min(w, h)
        if scale < 1.0:  # only shrink; tiny images pass through unchanged
            img = img.resize(
                (round(w * scale), round(h * scale)),
                Image.LANCZOS,
            )
        buf = BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return buf.getvalue()


def extract_split(split_dir: str, split: str):
    out_split = IMG_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{split_dir}/*.parquet"))
    # resume: continue from the number of images already extracted
    idx = len(list(out_split.glob("*.jpg")))
    if idx:
        print(f"  resuming {split} at image {idx:,}")

    n_written = 0  # global running index across all shards
    for f in files:
        lf = pl.scan_parquet(f)
        n = lf.select(pl.len()).collect().item()
        for offset in range(0, n, BATCH):
            if n_written + BATCH <= idx:  # whole batch already extracted
                n_written += min(BATCH, n - offset)
                continue
            chunk = lf.slice(offset, BATCH).collect()
            for b in chunk["image"].struct.field("bytes"):
                if n_written < idx:        # skip already-written rows
                    n_written += 1
                    continue
                (out_split / f"{n_written:07d}.jpg").write_bytes(downscale(b))
                n_written += 1
        print(f"  {split}: {os.path.basename(f)} done ({n_written:,} total)")


def main():
    split_dir = f"{OUT_DIR}/data"
    if not glob.glob(f"{split_dir}/*.parquet"):
        raise SystemExit(f"No parquet shards found under {split_dir}/ — run the downloader first")
    print(f"Extracting train -> {IMG_DIR / 'train'}")
    extract_split(split_dir, "train")
    print("Done. Images at:", IMG_DIR)


if __name__ == "__main__":
    main()
