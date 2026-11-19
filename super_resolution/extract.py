#!/usr/bin/env python3
"""Extract pre-split, pre-scaled JPEGs from downloaded Wikiart parquet shards.

By default, images are written to:
    <TRAINING_DATA_DIR>-images/<resolution>/train_98/
    <TRAINING_DATA_DIR>-images/<resolution>/test_1/
    <TRAINING_DATA_DIR>-images/<resolution>/eval_1/

The split is deterministic for a given seed and set of parquet shards. Filenames
are based on source-row IDs, so extraction is resumable and extracting only one
set produces exactly the same files as extracting all sets.

Examples:
    python -m super_resolution.extract
    python -m super_resolution.extract --sets eval --resolution 1024
    python -m super_resolution.extract --sets train test --train-percent 95 --test-percent 4 --eval-percent 1
"""

import argparse
import json
import os
import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import polars as pl
from PIL import Image

OUT_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
IMG_DIR = Path(os.environ.get("IMAGES_DIR", OUT_DIR + "-images"))
BATCH = 512              # rows decoded per step; lower if RAM is tight
TARGET_MIN_SIDE = 300    # shorter side after scaling
JPEG_QUALITY = 90
SPLITS = ("train", "test", "eval")


def downscale(b: bytes, target_min_side: int = TARGET_MIN_SIDE) -> bytes:
    with Image.open(BytesIO(b)) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = target_min_side / min(w, h)
        if scale < 1.0:  # only shrink; tiny images pass through unchanged
            img = img.resize(
                (round(w * scale), round(h * scale)),
                Image.LANCZOS,
            )
        buf = BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return buf.getvalue()


def _percentage_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate_percentages(percentages: dict[str, float]) -> None:
    if any(percentages[name] < 0 for name in SPLITS):
        raise ValueError("Split percentages must be non-negative")
    if abs(sum(percentages.values()) - 100.0) > 1e-8:
        raise ValueError(
            f"Split percentages must sum to 100, got {sum(percentages.values()):g}"
        )


def _split_assignments(
    n_total: int, percentages: dict[str, float], seed: int
) -> tuple[bytearray, dict[str, int]]:
    """Return a split index per source row and exact split sizes."""
    n_train = int(n_total * percentages["train"] / 100)
    n_test = int(n_total * percentages["test"] / 100)
    counts = {"train": n_train, "test": n_test, "eval": n_total - n_train - n_test}

    shuffled_indices = list(range(n_total))
    random.Random(seed).shuffle(shuffled_indices)
    assignments = bytearray(n_total)  # zero initialized: train
    for split_index, split in enumerate(("test", "eval"), start=1):
        start = n_train if split == "test" else n_train + n_test
        stop = start + counts[split]
        for source_index in shuffled_indices[start:stop]:
            assignments[source_index] = split_index
    return assignments, counts


def _check_manifest(
    resolution_dir: Path, manifest: dict, output_dirs: dict[str, Path]
) -> None:
    manifest_path = resolution_dir / "split_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != manifest:
            raise ValueError(
                f"Existing split settings differ at {manifest_path}. Use another "
                "output directory or remove the old extracted split before rerunning."
            )
        return

    if any(any(path.glob("*.jpg")) for path in output_dirs.values()):
        raise ValueError(
            f"Images already exist under {resolution_dir} but {manifest_path} is "
            "missing. Refusing to resume an unverified split."
        )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def extract_dataset(
    data_dir: Path,
    output_dir: Path,
    resolution: int = TARGET_MIN_SIDE,
    percentages: dict[str, float] | None = None,
    seed: int = 43,
    sets: Iterable[str] = SPLITS,
) -> dict[str, Path]:
    """Extract selected sets and return their output directories."""
    if resolution <= 0:
        raise ValueError("Resolution must be greater than zero")
    percentages = percentages or {"train": 98.0, "test": 1.0, "eval": 1.0}
    _validate_percentages(percentages)
    requested = {name for name in sets}
    unknown = requested.difference(SPLITS)
    if unknown:
        raise ValueError(f"Unknown split sets: {sorted(unknown)}")
    if not requested:
        raise ValueError("Select at least one set to extract")

    split_dir = data_dir / "data"
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet shards found under {split_dir}/ — run the downloader first"
        )

    lazy_shards = [pl.scan_parquet(str(path)) for path in files]
    shard_sizes = [lf.select(pl.len()).collect().item() for lf in lazy_shards]
    n_total = sum(shard_sizes)
    if n_total == 0:
        raise ValueError("The parquet shards contain no images")

    assignments, counts = _split_assignments(n_total, percentages, seed)
    resolution_dir = output_dir / str(resolution)
    output_dirs = {
        split: resolution_dir / f"{split}_{_percentage_name(percentages[split])}"
        for split in SPLITS
        if split in requested
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "seed": seed,
        "resolution": resolution,
        "percentages": percentages,
        "counts": counts,
        "n_source_rows": n_total,
        "filename_scheme": "zero-padded global source-row index",
    }
    _check_manifest(resolution_dir, manifest, output_dirs)

    existing = {
        split: len(list(path.glob("*.jpg"))) for split, path in output_dirs.items()
    }
    for split, count in existing.items():
        if count:
            print(f"  resuming {split}: {count:,}/{counts[split]:,} images present")

    written = dict.fromkeys(requested, 0)
    shard_start = 0
    for source_path, lf, shard_size in zip(files, lazy_shards, shard_sizes):
        shard_written = 0
        for offset in range(0, shard_size, BATCH):
            stop = min(offset + BATCH, shard_size)
            selected_rows = []
            for local_index in range(offset, stop):
                source_index = shard_start + local_index
                split = SPLITS[assignments[source_index]]
                if split not in requested:
                    continue
                if (output_dirs[split] / f"{source_index:07d}.jpg").exists():
                    continue
                selected_rows.append(local_index)

            if not selected_rows:
                continue

            # Filter before collecting so eval-only extraction does not decode the
            # train/test image bytes into Python objects.
            chunk = (
                lf.with_row_index("_row_index")
                .filter(pl.col("_row_index").is_in(selected_rows))
                .select(
                    pl.col("_row_index"),
                    pl.col("image").struct.field("bytes").alias("image_bytes"),
                )
                .collect()
            )
            for row in chunk.iter_rows(named=True):
                source_index = shard_start + row["_row_index"]
                split = SPLITS[assignments[source_index]]
                output_path = output_dirs[split] / f"{source_index:07d}.jpg"
                temporary_path = output_path.with_suffix(".jpg.tmp")
                temporary_path.write_bytes(downscale(row["image_bytes"], resolution))
                temporary_path.replace(output_path)
                written[split] += 1
                shard_written += 1

        print(
            f"  {source_path.name}: wrote {shard_written:,} selected images "
            f"({shard_size:,} source rows)"
        )
        shard_start += shard_size

    for split in sorted(requested):
        print(
            f"  {split}: {existing[split] + written[split]:,}/{counts[split]:,} "
            f"images -> {output_dirs[split]}"
        )
    return output_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(OUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=IMG_DIR)
    parser.add_argument(
        "--resolution",
        type=int,
        default=int(os.environ.get("FEATURE_RESOLUTION", TARGET_MIN_SIDE)),
        help="Target shorter side in pixels; also used in the output path",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train-percent", type=float, default=98.0)
    parser.add_argument("--test-percent", type=float, default=1.0)
    parser.add_argument("--eval-percent", type=float, default=1.0)
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Sets to extract; use '--sets eval' to avoid extracting train/test",
    )
    args = parser.parse_args(argv)

    percentages = {
        "train": args.train_percent,
        "test": args.test_percent,
        "eval": args.eval_percent,
    }
    try:
        output_dirs = extract_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            percentages=percentages,
            seed=args.seed,
            sets=args.sets,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("Done. Extracted image directories:")
    for path in output_dirs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
