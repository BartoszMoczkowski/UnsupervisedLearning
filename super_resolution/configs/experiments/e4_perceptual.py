"""E4: expensive perceptual references and SRGAN (plan section 7, E4)."""

from super_resolution.configs.elements.loss import DiscriminatorBCELoss
from super_resolution.configs.experiments import (
    Experiment,
    FrequencyAwareGeneratorLoss,
    GeneratorL1LossConfig,
    base_generator_model,
    gan_discriminator_model,
    make_stage,
)


def build_experiment(dataset_root, n_seen_images: int = 100_000) -> Experiment:
    """E4-VGG22 / E4-VGG54 / E4-SRGAN-MSE / E4-SRGAN-VGG54.

    Keep as separate later stages per plan implementation order: only run after the best
    non-adversarial configuration is stable.
    """
    discriminator = gan_discriminator_model()
    d_loss = DiscriminatorBCELoss()
    stages = [
        make_stage(
            "E4-VGG22",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=GeneratorL1LossConfig(), lambda_vgg=1.0, vgg_layer="vgg22"
            ),
            n_seen_images,
        ),
        make_stage(
            "E4-VGG54",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=GeneratorL1LossConfig(), lambda_vgg=1.0, vgg_layer="vgg54"
            ),
            n_seen_images,
        ),
        make_stage(
            "E4-SRGAN-MSE",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=GeneratorL1LossConfig(), lambda_adv=1e-3
            ),
            n_seen_images,
            discriminator_model=discriminator,
            discriminator_loss=d_loss,
        ),
        make_stage(
            "E4-SRGAN-VGG54",
            dataset_root,
            base_generator_model(),
            FrequencyAwareGeneratorLoss(
                base_loss=GeneratorL1LossConfig(),
                lambda_vgg=1.0,
                vgg_layer="vgg54",
                lambda_adv=1e-3,
            ),
            n_seen_images,
            discriminator_model=discriminator,
            discriminator_loss=d_loss,
        ),
    ]
    return Experiment(name="E4-perceptual-references", stages=stages)
