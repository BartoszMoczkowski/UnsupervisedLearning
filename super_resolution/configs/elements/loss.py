from abc import ABC
from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


class GeneratorLoss(ABC):
    def calculate_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        pass


class GeneratorMseLoss(GeneratorLoss):
    def calculate_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        return F.mse_loss(sr_images, hr_images)


class GeneratorDiscriminatorLoss(GeneratorLoss):
    def __init__(self) -> None:
        self.bce_loss = BCEWithLogitsLoss()

    def calculate_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if discriminator is None:
            raise ValueError("Discriminator loss requires a discriminator")
        d_fake_adv = discriminator(sr_images)
        return self.bce_loss(d_fake_adv, torch.ones_like(d_fake_adv))


class CompositeGeneratorLoss(GeneratorLoss):
    def __init__(self, losses: list[GeneratorLoss], weights: list[float]) -> None:
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights

    def calculate_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss.calculate_loss(sr_images, hr_images, discriminator)
        return total_loss


class GeneratorLossList(Enum):
    MSE_LOSS = GeneratorMseLoss
    DISCRIMINATOR_LOSS = GeneratorDiscriminatorLoss
    COMPOSITE_LOSS = CompositeGeneratorLoss


class DiscriminatorLoss(ABC):
    def calculate_loss(self, scores_fake: torch.Tensor, scores_real: torch.Tensor) -> torch.Tensor:
        pass


class DiscriminatorBCELoss(DiscriminatorLoss):
    def __init__(self) -> None:
        self.bce_loss = BCEWithLogitsLoss()

    def calculate_loss(self, scores_fake: torch.Tensor, scores_real: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(scores_fake, torch.zeros_like(scores_fake)) + self.bce_loss(
            scores_real, torch.ones_like(scores_real)
        )


class DiscriminatorLossList(Enum):
    BCE_LOSS = DiscriminatorBCELoss
