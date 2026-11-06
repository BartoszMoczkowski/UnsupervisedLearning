"""E3: frequency-domain architectures (plan section 7, E3).

The key architectural comparison: A2 (packed control) vs A3 (pure DCT) vs A4 (DCT + spatial
refinement), matched on trunk width/depth as closely as practical.
"""

from super_resolution.configs.elements.model import AdamOptimizerConfig, GeneratorModel
from super_resolution.configs.experiments import (
    DCTGeneratorModelConfig,
    DCTHybridModelConfig,
    Experiment,
    PackedBlockModelConfig,
    GeneratorL1LossConfig,
    make_stage,
)


def build_experiment(dataset_root, n_seen_images: int = 100_000) -> Experiment:
    """E3-PACK / E3-DCT / E3-DCT-REFINE at r = 2 with identical 16-block trunks."""
    l1 = GeneratorL1LossConfig()
    stages = [
        make_stage(
            "E3-PACK",
            dataset_root,
            GeneratorModel(PackedBlockModelConfig(width=64, num_blocks=16, scale=2), AdamOptimizerConfig(1e-4, (0.9, 0.999))),
            l1,
            n_seen_images,
        ),
        make_stage(
            "E3-DCT",
            dataset_root,
            GeneratorModel(DCTGeneratorModelConfig(width=64, num_blocks=16, scale=2), AdamOptimizerConfig(1e-4, (0.9, 0.999))),
            GeneratorL1LossConfig(),
            n_seen_images,
        ),
        make_stage(
            "E3-DCT-REFINE",
            dataset_root,
            GeneratorModel(
                DCTHybridModelConfig(dct_width=64, dct_num_blocks=16, spatial_width=32, n_spatial_blocks=2, scale=2),
                AdamOptimizerConfig(1e-4, (0.9, 0.999)),
            ),
            GeneratorL1LossConfig(),
            n_seen_images,
        ),
    ]
    return Experiment(name="E3-dct-architectures", stages=stages)
