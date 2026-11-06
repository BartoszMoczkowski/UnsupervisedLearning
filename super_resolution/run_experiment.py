"""Run a training experiment from the configs/ architecture.

Usage:
    python -m super_resolution.run_experiment <experiment_module> <dataset_root> [n_seen_images]

Examples:
    python -m super_resolution.run_experiment e0_baseline /data/wikiart-parquet 100000
    python -m super_resolution.run_experiment e3_dct_architectures /data/wikiart-extracted 50000
"""

import sys
from importlib import import_module
from pathlib import Path

AVAILABLE_EXPERIMENTS = [
    "e0_baseline",
    "e1_frequency_losses",
    "e2_color",
    "e3_dct_architectures",
    "e4_perceptual",
]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        print(f"Available experiments: {', '.join(AVAILABLE_EXPERIMENTS)}")
        sys.exit(1)
    experiment_module = sys.argv[1]
    dataset_root = Path(sys.argv[2])
    n_seen_images = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000

    if experiment_module not in AVAILABLE_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment_module}'; choose from {AVAILABLE_EXPERIMENTS}"
        )

    module = import_module(f"super_resolution.configs.experiments.{experiment_module}")
    experiment = module.build_experiment(dataset_root, n_seen_images)

    from super_resolution.parametrized_train import train_experiment

    train_experiment(experiment)


if __name__ == "__main__":
    main()
