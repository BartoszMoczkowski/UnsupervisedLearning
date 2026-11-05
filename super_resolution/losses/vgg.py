"""Frozen VGG19 feature loss (plan section 5.6): VGG22 and VGG54 variants."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights

from super_resolution.losses.pixel import LossOutput

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Feature map after activation at conv 2.2 / 5.4 (indexing conv-relu blocks 0-based).
_VGG19_RELU_INDICES = {"vgg22": 7, "vgg54": 34}


class VGGFeatureLoss(nn.Module):
    """L_vgg = mean((phi(sr) - phi(hr))^2) on frozen, eval-mode VGG19 features."""

    def __init__(
        self,
        layer: str = "vgg54",
        weights: VGG19_Weights | None = None,
    ) -> None:
        super().__init__()
        if layer not in _VGG19_RELU_INDICES:
            raise ValueError(f"layer must be one of {sorted(_VGG19_RELU_INDICES)}, got {layer}")
        if weights is None:
            weights = VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features[: _VGG19_RELU_INDICES[layer] + 1]
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.layer = layer
        self.network = vgg.eval()
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> LossOutput:
        if sr.shape != hr.shape:
            raise ValueError(f"sr and hr shapes must match, got {tuple(sr.shape)} vs {tuple(hr.shape)}")
        mean = self.mean.to(dtype=sr.dtype)
        std = self.std.to(dtype=sr.dtype)
        sr_features = self.network((sr - mean) / std)
        with torch.no_grad():
            hr_features = self.network((hr - mean) / std)
        term = torch.mean((sr_features - hr_features) ** 2)
        return LossOutput(total=term, terms={self.layer: term})
