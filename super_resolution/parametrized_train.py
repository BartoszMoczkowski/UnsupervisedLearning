import math
from pathlib import Path

import comet_ml
import torch
from tqdm import tqdm

from super_resolution.configs.elements.stage import Experiment, Stage, StageState


def _serializable_config(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serializable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_config(item) for item in value]
    if isinstance(value, torch.nn.Module):
        return value.__class__.__name__
    if hasattr(value, "__dict__"):
        config = {
            key: _serializable_config(item)
            for key, item in value.__dict__.items()
        }
        config["type"] = value.__class__.__name__
        return config
    return str(value)


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


def _train_stage(
    stage: Stage,
    previous_generator_state: dict | None,
    previous_discriminator_state: dict | None,
    target_device: str,
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

    for _ in tqdm(range(n_epochs), desc=stage.human_readable_name):
        for hr_img in train_loader:
            if hr_img.ndim == 5:
                hr_img = hr_img.flatten(0, 1)
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
            loss = stage.generator_loss.calculate_loss(sr_img, hr_img, discriminator)
            loss.backward()
            generator_optimizer.step()
            n_seen_images += hr_img.shape[0]

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
    comet_experiment = comet_ml.Experiment()
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
            )
            previous_generator_state = state["generator"]
            previous_discriminator_state = state.get("discriminator")
    finally:
        comet_experiment.end()
