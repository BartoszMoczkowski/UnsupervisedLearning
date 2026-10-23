#!/usr/bin/env python3
"""Download the full huggan/wikiart train split (81,444 images) to disk as parquet."""

import os
from huggingface_hub import snapshot_download, HfApi

REPO_ID = "huggan/wikiart"
OUT_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # pip install hf_transfer


def main():
    # 1. Show the train shards on the main branch
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"Train shards: {len(shards)}")
    print(shards[0], "...", shards[-1])

    # 2. Download everything (resumable — safe to Ctrl-C and re-run)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=OUT_DIR,
        max_workers=8,
    )
    print("Downloaded to:", local_dir)

    # 3. Verify
    import polars as pl
    lf = pl.scan_parquet(f"{local_dir}/data/*.parquet")
    n = lf.select(pl.len()).collect().item()
    print(f"  train: {n:,} rows")


if __name__ == "__main__":
    main()
