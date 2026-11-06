"""Run a training experiment from the configs/ architecture.

Usage:
    python -m super_resolution.run_experiment <experiment_module> <dataset_root> [n_seen_images]

Examples:
    python -m super_resolution.run_experiment e0_baseline /data/wikiart-parquet 100000
    python -m super_resolution.run_experiment e3_dct_architectures /data/wikiart-extracted 50000
"""

import sys
from pathlib import Path

import comet_ml

from .configs.experiments.e0_baseline import build_experiment as e0_build
from .parametrized_train import train_experiment

DATASET_ROOT = Path("/data/wikiart-parquet-images")
N_SEEN_IMAGES = 100_000


def main() -> None:

    experiment = e0_build(DATASET_ROOT, N_SEEN_IMAGES)

    train_experiment(experiment)


if __name__ == "__main__":
    main()
