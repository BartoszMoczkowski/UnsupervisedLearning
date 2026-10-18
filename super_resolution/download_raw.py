#!/usr/bin/env python3
"""Download the full huggan/wikiart dataset (all splits) to disk as parquet."""

import os
from huggingface_hub import snapshot_download, HfApi

REPO_ID = "huggan/wikiart"
OUT_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # pip install hf_transfer


def main():
    # 1. Show what splits exist on the parquet conversion branch
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset",
                                revision="refs/convert/parquet")
    splits = sorted({f.split("/")[1] for f in files if f.endswith(".parquet")})
    print("Available splits:", splits)
    print(f"Total parquet files: {sum(f.endswith('.parquet') for f in files)}")

    # 2. Download everything (resumable — safe to Ctrl-C and re-run)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="refs/convert/parquet",
        local_dir=OUT_DIR,
        max_workers=8,
    )
    print("Downloaded to:", local_dir)

    # 3. Verify
    import polars as pl
    import glob
    for split in sorted(glob.glob(f"{local_dir}/default/*")):
        name = os.path.basename(split)
        lf = pl.scan_parquet(f"{split}/*.parquet")
        n = lf.select(pl.len()).collect().item()
        print(f"  {name}: {n:,} rows")


if __name__ == "__main__":
    main()
