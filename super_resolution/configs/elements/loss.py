from abc import ABC
from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from ...losses.adversarial import GeneratorAdversarialLoss
from ...losses.dct import DCTCoefficientLoss
from ...losses.pixel import CharbonnierLoss, L1Loss, MSELoss
from ...losses.pyramid import LaplacianPyramidLoss
from ...losses.vgg import VGGFeatureLoss
from ...losses.ycbcr import (
    ChromaL1Loss,
    LumaPyramidLoss,
    YCbCrWeightedLoss,
)


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
            total_loss = total_loss + weight * loss.calculate_loss(
                sr_images, hr_images, discriminator
            )
        return total_loss


class GeneratorLossList(Enum):
    MSE_LOSS = GeneratorMseLoss
    DISCRIMINATOR_LOSS = GeneratorDiscriminatorLoss
    COMPOSITE_LOSS = CompositeGeneratorLoss


class DiscriminatorLoss(ABC):
    def calculate_loss(
        self, scores_fake: torch.Tensor, scores_real: torch.Tensor
    ) -> torch.Tensor:
        pass


class DiscriminatorBCELoss(DiscriminatorLoss):
    def __init__(self) -> None:
        self.bce_loss = BCEWithLogitsLoss()

    def calculate_loss(
        self, scores_fake: torch.Tensor, scores_real: torch.Tensor
    ) -> torch.Tensor:
        return self.bce_loss(
            scores_fake, torch.zeros_like(scores_fake)
        ) + self.bce_loss(scores_real, torch.ones_like(scores_real))


class DiscriminatorLossList(Enum):
    BCE_LOSS = DiscriminatorBCELoss


# ---------------------------------------------------------------------------
# Loss configs for the frequency/perceptual experiments (plan section 5).
# All keep the existing GeneratorLoss interface so they plug into Stage unchanged;
# DetailedGeneratorLoss additionally exposes unweighted per-term values for logging.
# ---------------------------------------------------------------------------


class DetailedGeneratorLoss(GeneratorLoss):
    """Base class returning a dict of total + unweighted terms (plan section 5 intro)."""

    def calculate_loss_terms(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        pass

    def calculate_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        discriminator: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        return self.calculate_loss_terms(sr_images, hr_images, discriminator)["total"]


class ModuleGeneratorLoss(DetailedGeneratorLoss):
    """Wraps a parameter-free loss module such as MSE/L1/Charbonnier."""

    def __init__(self, module) -> None:
        self.module = module

    def calculate_loss_terms(
        self, sr_images, hr_images, discriminator=None
    ) -> dict[str, torch.Tensor]:
        output = self.module(sr_images, hr_images)
        return {
            "total": output["total"],
            **{f"loss_{k}": v for k, v in output["terms"].items()},
        }


class GeneratorL1LossConfig(DetailedGeneratorLoss):
    def calculate_loss_terms(
        self, sr_images, hr_images, discriminator=None
    ) -> dict[str, torch.Tensor]:
        return ModuleGeneratorLoss(L1Loss()).calculate_loss_terms(
            sr_images, hr_images, discriminator
        )


class GeneratorCharbonnierLossConfig(DetailedGeneratorLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def calculate_loss_terms(
        self, sr_images, hr_images, discriminator=None
    ) -> dict[str, torch.Tensor]:
        return ModuleGeneratorLoss(CharbonnierLoss(self.eps)).calculate_loss_terms(
            sr_images, hr_images, discriminator
        )


class FrequencyAwareGeneratorLoss(DetailedGeneratorLoss):
    """Plan 5.8 recommended inexpensive composition:

    L = L_base(RGB or YCbCr)
      + lambda_pyr  * L_pyr(Y)
      + lambda_chroma * (L1(Cb) + L1(Cr))
      + optional lambda_dct * L_dct(Y)
      + optional lambda_vgg * L_vgg(RGB)
      + optional lambda_adv * L_adv   (requires discriminator)
    """

    def __init__(
        self,
        base_loss: DetailedGeneratorLoss | None = None,
        lambda_pyr: float = 0.0,
        pyramid_levels: int = 3,
        pyramid_band_weights: list[float] | None = None,
        luma_pyramid: bool = True,
        lambda_chroma: float = 0.0,
        lambda_dct: float = 0.0,
        dct_group_weights: dict[str, float] | None = None,
        lambda_vgg: float = 0.0,
        vgg_layer: str = "vgg54",
        lambda_adv: float = 0.0,
        base_in_ycbcr: bool = False,
        lambda_y: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_cr: float = 1.0,
    ) -> None:
        self.base_loss = base_loss
        self.lambda_pyr = lambda_pyr
        self.pyramid_levels = pyramid_levels
        self.pyramid_band_weights = pyramid_band_weights
        self.luma_pyramid = luma_pyramid
        self.lambda_chroma = lambda_chroma
        self.lambda_dct = lambda_dct
        self.dct_group_weights = dct_group_weights
        self.lambda_vgg = lambda_vgg
        self.vgg_layer = vgg_layer
        self.lambda_adv = lambda_adv
        self.base_in_ycbcr = base_in_ycbcr
        self.lambda_y, self.lambda_cb, self.lambda_cr = lambda_y, lambda_cb, lambda_cr

    def _modules(self):
        """Build (and cache on first use) the underlying loss modules."""
        if not hasattr(self, "_module_cache"):
            self._module_cache = {
                "base": self.base_loss,
                "pyr": LumaPyramidLoss(self.pyramid_levels, self.pyramid_band_weights)
                if self.luma_pyramid
                else LaplacianPyramidLoss(
                    self.pyramid_levels, self.pyramid_band_weights
                ),
                "chroma": ChromaL1Loss(),
                "dct": DCTCoefficientLoss(self.dct_group_weights),
                "ycbcr": YCbCrWeightedLoss(
                    self.lambda_y, self.lambda_cb, self.lambda_cr
                ),
            }
        return self._module_cache

    def calculate_loss_terms(
        self, sr_images, hr_images, discriminator=None
    ) -> dict[str, torch.Tensor]:
        modules = self._modules()
        total = sr_images.new_zeros(())
        terms: dict[str, torch.Tensor] = {}

        if self.base_in_ycbcr:
            base_out = modules["ycbcr"](sr_images, hr_images)
            base_total, base_terms = base_out["total"], base_out["terms"]
        elif modules["base"] is not None:
            base_out = modules["base"].calculate_loss_terms(sr_images, hr_images)
            base_total, base_terms = (
                base_out["total"],
                {key: value for key, value in base_out.items() if key != "total"},
            )
        else:
            base_total, base_terms = None, {}
        if base_total is not None:
            total = total + base_total
            for key, value in base_terms.items():
                terms[f"base_{key}"] = value

        if self.lambda_pyr:
            pyr_out = modules["pyr"](sr_images, hr_images)
            total = total + self.lambda_pyr * pyr_out["total"]
            terms["weighted_pyramid"] = self.lambda_pyr * pyr_out["total"]
            for key, value in pyr_out["terms"].items():
                terms[f"pyr_{key}"] = value

        if self.lambda_chroma:
            chroma_out = modules["chroma"](sr_images, hr_images)
            total = total + self.lambda_chroma * chroma_out["total"]
            terms["weighted_chroma"] = self.lambda_chroma * chroma_out["total"]

        if self.lambda_dct:
            dct_out = modules["dct"](sr_images, hr_images)
            total = total + self.lambda_dct * dct_out["total"]
            terms["weighted_dct"] = self.lambda_dct * dct_out["total"]
            for key, value in dct_out["terms"].items():
                terms[f"dct_{key}"] = value

        if self.lambda_vgg:
            vgg_out = _get_vgg_loss(self.vgg_layer)(sr_images, hr_images)
            total = total + self.lambda_vgg * vgg_out["total"]
            terms["weighted_vgg"] = self.lambda_vgg * vgg_out["total"]
            for key, value in vgg_out["terms"].items():
                terms[f"vgg_{key}"] = value

        if self.lambda_adv:
            if discriminator is None:
                raise ValueError("lambda_adv requires a discriminator in the stage")
            d_fake_adv = discriminator(sr_images)
            adv_out = GeneratorAdversarialLoss()(d_fake_adv)
            total = total + self.lambda_adv * adv_out["total"]
            terms["weighted_adv"] = self.lambda_adv * adv_out["total"]

        terms["total"] = total
        return terms

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_module_cache", None)
        return state


_VGG_LOSS_CACHE: dict[str, VGGFeatureLoss] = {}


def _get_vgg_loss(layer: str) -> VGGFeatureLoss:
    """Module-level cache: keeps pretrained VGG19 out of pickled Stage configs."""
    if layer not in _VGG_LOSS_CACHE:
        _VGG_LOSS_CACHE[layer] = VGGFeatureLoss(layer)
    return _VGG_LOSS_CACHE[layer]


class GeneratorLossListNew(Enum):
    """Registry of the new-style losses (kept separate to preserve old configs)."""

    L1 = GeneratorL1LossConfig
    CHARBONNIER = GeneratorCharbonnierLossConfig
    FREQUENCY_AWARE = FrequencyAwareGeneratorLoss
