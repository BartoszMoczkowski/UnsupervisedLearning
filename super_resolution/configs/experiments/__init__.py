from ..elements.dataset import WikiartDatasetConfig
from ..elements.loss import (
    FrequencyAwareGeneratorLoss,
    GeneratorCharbonnierLossConfig,
    GeneratorL1LossConfig,
)
from ..elements.model import (
    AdamOptimizerConfig,
    DCTGeneratorModelConfig,
    DCTHybridModelConfig,
    DiscriminatorModel,
    GeneratorModel,
    PackedBlockModelConfig,
    SRGANPaperDiscriminatorConfig,
    SRResNetModelConfig,
)
from ..elements.preprocessing import (
    BicubicDownsample,
    Int8To01Float,
    RandomCropAugment,
)
from ..elements.stage import Experiment, Stage

PAPER_HR_CROP = 96
PAPER_SCALE = 4


def paper_dataset_config(
    dataset_root, batch_size: int = 16, hr_size: int = PAPER_HR_CROP
) -> WikiartDatasetConfig:
    """Paper-faithful data pipeline: HR crop -> spatial augment -> bicubic LR."""
    return WikiartDatasetConfig(
        root=dataset_root,
        hr_size=hr_size,
        batch_size=batch_size,
    )


def base_generator_model(
    input_output: str = "rgb",
    lr: float = 1e-4,
    width: int = 64,
    num_blocks: int = 16,
    scale: int = PAPER_SCALE,
) -> GeneratorModel:
    return GeneratorModel(
        SRResNetModelConfig(width, num_blocks, scale, True, input_output),
        AdamOptimizerConfig(lr=lr, betas=(0.9, 0.999)),
    )


def standard_preprocessing(hr_size: int = PAPER_HR_CROP, scale: int = PAPER_SCALE):
    """Returns (general, hr, lr) preprocessing for a spatial-model stage."""
    return (
        RandomCropAugment(hr_size),
        Int8To01Float(),
        BicubicDownsample(scale),
    )


def make_stage(
    name: str,
    dataset_root,
    generator_model: GeneratorModel,
    generator_loss,
    n_seen_images: int,
    discriminator_model: DiscriminatorModel | None = None,
    discriminator_loss=None,
    hr_size: int = PAPER_HR_CROP,
    batch_size: int = 256,
) -> Stage:
    general, hr_pre, lr_pre = standard_preprocessing(hr_size)
    return Stage(
        human_readable_name=name,
        dataset=paper_dataset_config(dataset_root, batch_size, hr_size),
        general_preprocessing=general,
        hr_preprocessing=hr_pre,
        lr_preprocessing=lr_pre,
        generator_model=generator_model,
        generator_loss=generator_loss,
        n_seen_images=n_seen_images,
        discriminator_model=discriminator_model,
        discriminator_loss=discriminator_loss,
    )


def charbonnier_base() -> GeneratorCharbonnierLossConfig:
    return GeneratorCharbonnierLossConfig(eps=1e-3)


def gan_discriminator_model(lr: float = 1e-4) -> DiscriminatorModel:
    return DiscriminatorModel(
        SRGANPaperDiscriminatorConfig(img_size=(PAPER_HR_CROP, PAPER_HR_CROP)),
        AdamOptimizerConfig(lr=lr, betas=(0.9, 0.999)),
    )


__all__ = [
    "Experiment",
    "Stage",
    "WikiartDatasetConfig",
    "FrequencyAwareGeneratorLoss",
    "GeneratorCharbonnierLossConfig",
    "GeneratorL1LossConfig",
    "AdamOptimizerConfig",
    "DCTGeneratorModelConfig",
    "DCTHybridModelConfig",
    "DiscriminatorModel",
    "GeneratorModel",
    "PackedBlockModelConfig",
    "SRGANPaperDiscriminatorConfig",
    "SRResNetModelConfig",
    "BicubicDownsample",
    "Int8To01Float",
    "RandomCropAugment",
    "paper_dataset_config",
    "base_generator_model",
    "standard_preprocessing",
    "make_stage",
    "charbonnier_base",
    "gan_discriminator_model",
]
