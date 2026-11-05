"""E0: paper and cheap spatial baselines (plan section 7, E0)."""

from super_resolution.configs.experiments import (
    Experiment,
    GeneratorL1LossConfig,
    base_generator_model,
    charbonnier_base,
    make_stage,
)
from super_resolution.configs.elements.loss import ModuleGeneratorLoss
from super_resolution.losses.pixel import MSELoss


def build_experiment(dataset_root, n_seen_images: int = 100_000) -> Experiment:
    """E0-MSE / E0-L1 / E0-CHAR: A0 RGB/RGB with pixel losses only."""
    stages = [
        make_stage(
            "E0-MSE",
            dataset_root,
            base_generator_model(),
            ModuleGeneratorLoss(MSELoss()),
            n_seen_images,
        ),
        make_stage(
            "E0-L1",
            dataset_root,
            base_generator_model(),
            GeneratorL1LossConfig(),
            n_seen_images,
        ),
        make_stage(
            "E0-CHAR",
            dataset_root,
            base_generator_model(),
            charbonnier_base(),
            n_seen_images,
        ),
    ]
    return Experiment(name="E0-spatial-baselines", stages=stages)
