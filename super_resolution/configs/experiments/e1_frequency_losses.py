"""E1: frequency-aware losses on one fixed A0 generator (plan section 7, E1)."""

from super_resolution.configs.experiments import (
    Experiment,
    FrequencyAwareGeneratorLoss,
    GeneratorL1LossConfig,
    base_generator_model,
    make_stage,
)


def _l1() -> GeneratorL1LossConfig:
    return GeneratorL1LossConfig()


def build_experiment(dataset_root, n_seen_images: int = 100_000) -> Experiment:
    """E1-PY-RGB / E1-PY-Y / E1-PY-Y-HI / E1-DCT-Y.

    Fixed A0 (paper SRResNet); only the objective changes.
    """
    stages = [
        make_stage(
            "E1-PY-RGB",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=_l1(),
                lambda_pyr=1.0,
                pyramid_levels=3,
                luma_pyramid=False,
            ),
            n_seen_images,
        ),
        make_stage(
            "E1-PY-Y",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=_l1(),
                lambda_pyr=1.0,
                pyramid_levels=3,
                luma_pyramid=True,
                lambda_chroma=0.1,
            ),
            n_seen_images,
        ),
        make_stage(
            "E1-PY-Y-HI",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=_l1(),
                lambda_pyr=1.0,
                pyramid_levels=3,
                pyramid_band_weights=[3.0, 1.0, 0.5],
                luma_pyramid=True,
                lambda_chroma=0.1,
            ),
            n_seen_images,
        ),
        make_stage(
            "E1-DCT-Y",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=_l1(),
                lambda_dct=1.0,
                dct_group_weights={"dc": 1.0, "low_ac": 2.0, "mid_ac": 1.0, "high_ac": 0.5},
                lambda_chroma=0.1,
            ),
            n_seen_images,
        ),
    ]
    return Experiment(name="E1-frequency-losses", stages=stages)
