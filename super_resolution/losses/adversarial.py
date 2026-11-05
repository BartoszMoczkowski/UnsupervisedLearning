"""Adversarial losses with logits (plan section 5.7)."""

import torch
import torch.nn as nn

from super_resolution.losses.pixel import LossOutput


class DiscriminatorAdversarialLoss(nn.Module):
    """BCEWithLogits(D(hr), 1) + BCEWithLogits(D(sr.detach()), 0)."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, scores_fake: torch.Tensor, scores_real: torch.Tensor
    ) -> LossOutput:
        fake = self.bce(scores_fake, torch.zeros_like(scores_fake))
        real = self.bce(scores_real, torch.ones_like(scores_real))
        return LossOutput(total=fake + real, terms={"d_fake": fake, "d_real": real})


class GeneratorAdversarialLoss(nn.Module):
    """BCEWithLogits(D(sr), 1) — expects the discriminator output on sr as input."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, discriminator_scores_on_sr: torch.Tensor) -> LossOutput:
        term = self.bce(discriminator_scores_on_sr, torch.ones_like(discriminator_scores_on_sr))
        return LossOutput(total=term, terms={"adv": term})
