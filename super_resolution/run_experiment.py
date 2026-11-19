"""Run a training experiment from the configs/ architecture.

Usage:
    python -m super_resolution.run_experiment <experiment_module> <dataset_root> [n_seen_images]

Examples:
    python -m super_resolution.run_experiment e0_baseline /data/wikiart-parquet 100000
    python -m super_resolution.run_experiment e3_dct_architectures /data/wikiart-extracted 50000
"""

import os
import sys
from pathlib import Path

import comet_ml

from .configs.experiments.e0_baseline import build_experiment as e0_build
from .configs.experiments.e1_frequency_losses import build_experiment as e1_build
from .configs.experiments.e2_color import build_experiment as e2_build
from .configs.experiments.e3_dct_architectures import build_experiment as e3_build
from .configs.experiments.e4_perceptual import build_experiment as e4_build
from .parametrized_train import train_experiment

_image_dirs = os.environ.get(
    "IMAGES_DIRS", os.environ.get("IMAGES_DIR", "/data/wikiart-parquet-images")
)
DATASET_ROOT = [Path(root) for root in _image_dirs.split(os.pathsep) if root]
N_SEEN_IMAGES = 1_000_000


def main() -> None:

    experiment = e4_build(DATASET_ROOT, N_SEEN_IMAGES)

    train_experiment(experiment)


if __name__ == "__main__":
    main()
