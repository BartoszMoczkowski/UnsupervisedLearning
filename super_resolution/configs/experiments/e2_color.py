"""E2: color representation comparison (plan section 7, E2)."""

from super_resolution.configs.experiments import (
    Experiment,
    FrequencyAwareGeneratorLoss,
    GeneratorL1LossConfig,
    base_generator_model,
    make_stage,
)


def build_experiment(dataset_root, n_seen_images: int = 100_000) -> Experiment:
    """E2-RGB / E2-RGB-YLOSS / E2-YCC / E2-Y-ONLY with a shared base objective."""
    l1 = GeneratorL1LossConfig()
    stages = [
        make_stage(
            "E2-RGB",
            dataset_root,
            base_generator_model("rgb"),
            l1,
            n_seen_images,
        ),
        make_stage(
            "E2-RGB-YLOSS",
            dataset_root,
            base_generator_model("rgb"),
            FrequencyAwareGeneratorLoss(
                base_loss=GeneratorL1LossConfig(),
                lambda_pyr=1.0,
                luma_pyramid=True,
                lambda_chroma=0.1,
            ),
            n_seen_images,
        ),
        make_stage(
            "E2-YCC",
            dataset_root,
            base_generator_model("ycbcr"),
            FrequencyAwareGeneratorLoss(
                base_loss=None,
                base_in_ycbcr=True,
                lambda_y=4.0,
                lambda_cb=1.0,
                lambda_cr=1.0,
            ),
            n_seen_images,
        ),
        make_stage(
            "E2-Y-ONLY",
            dataset_root,
            base_generator_model("y"),
            FrequencyAwareGeneratorLoss(
                base_loss=None,
                base_in_ycbcr=True,
                lambda_y=6.0,
                lambda_cb=1.0,
                lambda_cr=1.0,
            ),
            n_seen_images,
        ),
    ]
    return Experiment(name="E2-color-representation", stages=stages)
