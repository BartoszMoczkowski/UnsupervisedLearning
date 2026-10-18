#!/usr/bin/env python3
"""Extract raw JPEGs from the downloaded wikiart parquet shards to disk.

Reads parquet from TRAINING_DATA_DIR (default ./wikiart-parquet),
writes images to <TRAINING_DATA_DIR>-images/<split>/NNNNNNN.jpg
Resumable: skips images that already exist.
"""

import os
import glob
import polars as pl
from pathlib import Path

OUT_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
IMG_DIR = Path(os.environ.get("IMAGES_DIR", OUT_DIR + "-images"))
BATCH = 512  # rows decoded per step; lower if RAM is tight


def extract_split(split_dir: str):
    split = os.path.basename(split_dir)
    out_split = IMG_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{split_dir}/*.parquet"))
    # resume: continue from the number of images already extracted
    idx = len(list(out_split.glob("*.jpg")))
    if idx:
        print(f"  resuming {split} at image {idx:,}")

    for f in files:
        lf = pl.scan_parquet(f)
        n = lf.select(pl.len()).collect().item()
        for offset in range(0, n, BATCH):
            if offset + BATCH <= idx:      # whole batch already extracted
                continue
            chunk = lf.slice(offset, BATCH).collect()
            for b in chunk["image"].struct.field("bytes"):
                if idx and offset < idx:   # skip already-written rows in partial batch
                    offset += 1
                    continue
                (out_split / f"{offset:07d}.jpg").write_bytes(b)
                offset += 1
        print(f"  {split}: {os.path.basename(f)} done ({offset:,} total)")


def main():
    splits = sorted(glob.glob(f"{OUT_DIR}/default/*"))
    if not splits:
        raise SystemExit(f"No splits found under {OUT_DIR}/default/ — run the downloader first")
    for split_dir in splits:
        print(f"Extracting {os.path.basename(split_dir)} -> {IMG_DIR / os.path.basename(split_dir)}")
        extract_split(split_dir)
    print("Done. Images at:", IMG_DIR)


if __name__ == "__main__":
    main()
