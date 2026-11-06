import math
from io import BytesIO
from pathlib import Path

import comet_ml
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from super_resolution.configs.elements.loss import DetailedGeneratorLoss
from super_resolution.configs.elements.stage import (
    Experiment,
    Stage,
    StageState,
    serializable_config,
)


def _serializable_config(value):
    return serializable_config(value)


def _apply_preprocessing(step, images: torch.Tensor) -> torch.Tensor:
    return step.apply(images) if step is not None else images


def _save_state(
    stage: Stage,
    generator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator: torch.nn.Module | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    n_epochs_completed: int,
    n_seen_images: int,
) -> StageState:
    state: StageState = {
        "completed": n_seen_images >= stage.n_seen_images,
        "n_epochs_completed": n_epochs_completed,
        "n_seen_images": n_seen_images,
        "generator": generator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
    }
    if discriminator is not None and discriminator_optimizer is not None:
        state["discriminator"] = discriminator.state_dict()
        state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    stage.save_stage_results(state)
    return state


def _log_stage_visuals(
    comet_experiment,
    stage: Stage,
    generator: torch.nn.Module,
    target_device: str,
    n_images: int = 5,
) -> None:
    """Pick n_images from the eval split and log LR|SR|HR strips to Comet."""
    if comet_experiment is None:
        return
    _, _, eval_dataset = stage.dataset.load_datasets()
    if len(eval_dataset) == 0:
        return
    generator.eval()

    n_take = min(n_images, len(eval_dataset))
    indices = torch.randperm(len(eval_dataset), generator=torch.Generator().manual_seed(43))[:n_take]

    hr_all = torch.stack([eval_dataset[int(i)] for i in indices])
    if hr_all.ndim == 5:
        hr_all = rearrange(hr_all, "b n c h w -> (b n) c h w")
    hr_all = hr_all.to(target_device)
    hr_all = _apply_preprocessing(stage.hr_preprocessing, hr_all)
    lr_all = _apply_preprocessing(stage.lr_preprocessing, hr_all)

    with torch.no_grad():
        sr_all = generator(lr_all)
    sr_all = sr_all.clamp(0, 1)

    # LR shown upscaled to HR size (nearest) so the strip is visually comparable.
    lr_up = torch.nn.functional.interpolate(
        lr_all, size=hr_all.shape[-2:], mode="nearest"
    )
    strip = torch.cat([lr_up, sr_all, hr_all], dim=3)  # concat along width
    for i in range(strip.shape[0]):
        pil_image = Image.fromarray(
            (strip[i].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        )
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        comet_experiment.log_image(
            buffer,
            name=f"{stage.human_readable_name}/lr_sr_hr_{i}",
        )
    generator.train()


def _train_stage(
    stage: Stage,
    previous_generator_state: dict | None,
    previous_discriminator_state: dict | None,
    target_device: str,
    comet_experiment=None,
) -> StageState:
    generator = stage.generator_model.model_config.build().to(target_device)
    generator_optimizer = stage.generator_model.optimizer_config.build(generator)
    discriminator = None
    discriminator_optimizer = None
    if stage.discriminator_model is not None:
        discriminator = stage.discriminator_model.model_config.build().to(target_device)
        discriminator_optimizer = stage.discriminator_model.optimizer_config.build(discriminator)

    if previous_generator_state is not None:
        generator.load_state_dict(previous_generator_state)
    if discriminator is not None and previous_discriminator_state is not None:
        discriminator.load_state_dict(previous_discriminator_state)

    n_epochs_completed = 0
    n_seen_images = 0
    if stage.state_data_path().exists():
        state = stage.load_state()
        generator.load_state_dict(state["generator"])
        generator_optimizer.load_state_dict(state["generator_optimizer"])
        if discriminator is not None and discriminator_optimizer is not None:
            discriminator.load_state_dict(state["discriminator"])
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        n_epochs_completed = state["n_epochs_completed"]
        n_seen_images = state["n_seen_images"]
        if state["completed"] and n_seen_images >= stage.n_seen_images:
            return state

    stage.save_stage_config()
    train_loader = stage.dataset.load_train_dataloader()
    n_remaining_images = stage.n_seen_images - n_seen_images
    n_epochs = math.ceil(n_remaining_images / len(train_loader.dataset))
    is_cuda = target_device == "cuda"
    generator.train()
    if discriminator is not None:
        discriminator.train()

    for _ in range(n_epochs):
        steps_in_epoch = math.ceil(len(train_loader.dataset) / stage.dataset.batch_size)
        progress = tqdm(
            train_loader,
            desc=f"{stage.human_readable_name} (epoch {n_epochs_completed + 1}/{n_epochs})",
            total=steps_in_epoch,
            unit="step",
        )
        for hr_img in progress:
            if hr_img.ndim == 5:
                hr_img = rearrange(hr_img, "b n c h w -> (b n) c h w")
            hr_img = hr_img.to(target_device, non_blocking=is_cuda)
            hr_img = _apply_preprocessing(stage.general_preprocessing, hr_img)
            lr_img = _apply_preprocessing(stage.lr_preprocessing, hr_img)
            hr_img = _apply_preprocessing(stage.hr_preprocessing, hr_img)

            if discriminator is not None:
                if discriminator_optimizer is None or stage.discriminator_loss is None:
                    raise ValueError("Discriminator model and loss must be configured together")
                discriminator.requires_grad_(True)
                with torch.no_grad():
                    sr_img = generator(lr_img)
                discriminator_optimizer.zero_grad(set_to_none=True)
                d_fake = discriminator(sr_img)
                d_real = discriminator(hr_img)
                d_loss = stage.discriminator_loss.calculate_loss(d_fake, d_real)
                d_loss.backward()
                discriminator_optimizer.step()
                discriminator.requires_grad_(False)

            generator_optimizer.zero_grad(set_to_none=True)
            sr_img = generator(lr_img)
            if isinstance(stage.generator_loss, DetailedGeneratorLoss):
                loss_terms = stage.generator_loss.calculate_loss_terms(sr_img, hr_img, discriminator)
                loss = loss_terms["total"]
            else:
                loss = stage.generator_loss.calculate_loss(sr_img, hr_img, discriminator)
            loss.backward()
            generator_optimizer.step()
            n_seen_images += hr_img.shape[0]

            progress.set_postfix(
                seen=f"{n_seen_images}/{stage.n_seen_images}",
                loss=f"{loss.item():.4f}",
                imgs_per_s=f"{progress.format_dict['rate'] * stage.dataset.batch_size:.0f}" if progress.format_dict.get("rate") else "?",
            )
            # Comet round-trips are expensive: log every 50 steps, not every step.
            if (
                comet_experiment is not None
                and isinstance(stage.generator_loss, DetailedGeneratorLoss)
                and (progress.n % 50 == 0)
            ):
                comet_experiment.log_metrics(
                    {f"train_{key}": value.item() for key, value in loss_terms.items()}
                )

        n_epochs_completed += 1
        state = _save_state(
            stage,
            generator,
            generator_optimizer,
            discriminator,
            discriminator_optimizer,
            n_epochs_completed,
            n_seen_images,
        )

    return state


def train_experiment(experiment: Experiment):
    comet_experiment = comet_ml.Experiment(project_name="unsupervisedlearning")
    comet_experiment.set_name(experiment.name)
    target_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    previous_generator_state = None
    previous_discriminator_state = None

    try:
        for stage in experiment.stages:
            comet_experiment.log_parameters(
                _serializable_config(stage),
                prefix=stage.compute_id(),
            )
            state = _train_stage(
                stage,
                previous_generator_state,
                previous_discriminator_state,
                target_device,
                comet_experiment,
            )
            previous_generator_state = state["generator"]
            previous_discriminator_state = state.get("discriminator")
            final_generator = stage.generator_model.model_config.build().to(target_device)
            final_generator.load_state_dict(previous_generator_state)
            _log_stage_visuals(comet_experiment, stage, final_generator, target_device)
    finally:
        comet_experiment.end()
