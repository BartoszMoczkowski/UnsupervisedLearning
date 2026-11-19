from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from image_destroyer.filters import BinaryClosing, MaskFilter
from image_destroyer.strategies import (
    MaskStrategy,
    PerlinNoiseConfig,
    PerlinNoiseStrategy,
)


class ImageDestroyer:
    def __init__(
        self,
        image: Image.Image,
        strategy: MaskStrategy | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._image = image.copy()
        self._strategy = (
            strategy
            if strategy is not None
            else PerlinNoiseStrategy(
                PerlinNoiseConfig(scale=0.01, threshold=0.91)
            )
        )
        self._seed = seed

    def destroy(
        self,
        *,
        strategy: MaskStrategy | None = None,
        filters: Iterable[MaskFilter] | None = None,
    ) -> Image.Image:
        rgb_image = self._image.convert("RGB")
        pixels = np.asarray(rgb_image).copy()
        active_strategy = self._strategy if strategy is None else strategy
        mask = active_strategy.generate_mask(
            pixels.shape[:2],
            np.random.default_rng(self._seed),
        )
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != pixels.shape[:2]:
            raise ValueError(
                "A strategy must return a mask matching the image dimensions"
            )

        active_filters = (BinaryClosing(size=7),) if filters is None else filters
        for mask_filter in active_filters:
            mask = np.asarray(mask_filter.apply(mask), dtype=bool)
            if mask.shape != pixels.shape[:2]:
                raise ValueError(
                    "A mask filter must preserve the mask dimensions"
                )

        pixels[mask] = 0
        return Image.fromarray(pixels)
