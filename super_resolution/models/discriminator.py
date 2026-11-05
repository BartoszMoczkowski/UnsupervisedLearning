"""A5: paper-style SRGAN discriminator for 96x96 RGB inputs (plan section 4.6)."""

import torch
import torch.nn as nn


class SRGANDiscriminator(nn.Module):
    """8-conv VGG-style discriminator returning a single real/fake logit.

    Produces the paper's 512x6x6 feature map for 96x96 input, then two dense layers.
    """

    def __init__(
        self,
        channels_input: int = 3,
        img_size: tuple[int, int] = (96, 96),
        feature_width: int = 64,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        w = feature_width

        def conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(leaky_slope, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(channels_input, w, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            conv_block(w, w, 2),
            conv_block(w, w * 2, 1),
            conv_block(w * 2, w * 2, 2),
            conv_block(w * 2, w * 4, 1),
            conv_block(w * 4, w * 4, 2),
            conv_block(w * 4, w * 8, 1),
            conv_block(w * 8, w * 8, 2),
        )
        final_h = (img_size[0] - 1) // 16 + 1
        final_w = (img_size[1] - 1) // 16 + 1
        flattened = w * 8 * final_h * final_w
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 1024),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
