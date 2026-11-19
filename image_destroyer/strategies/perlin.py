from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_destroyer.strategies._common import destruction_limit, select_with_limit


@dataclass(frozen=True)
class PerlinNoiseConfig:
    scale: float = 0.03
    threshold: float = 0.65
    max_destroyed_fraction: float = 1 / 16

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be greater than zero")
        if not 0 <= self.threshold < 1:
            raise ValueError("threshold must be in the range [0, 1)")
        if not 0 <= self.max_destroyed_fraction <= 1:
            raise ValueError("max_destroyed_fraction must be between 0 and 1")


@dataclass(frozen=True)
class PerlinNoiseStrategy:
    config: PerlinNoiseConfig = PerlinNoiseConfig()

    def generate_mask(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        noise = self._generate_noise(shape, rng)
        probabilities = np.clip(
            (noise - self.config.threshold) / (1.0 - self.config.threshold),
            0.0,
            1.0,
        )
        candidates = rng.random(shape) < probabilities
        limit = destruction_limit(shape, self.config.max_destroyed_fraction)
        return select_with_limit(candidates, limit, rng)

    def _generate_noise(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        height, width = shape
        scale = self.config.scale
        grid_height = int(np.ceil(height * scale)) + 2
        grid_width = int(np.ceil(width * scale)) + 2

        angles = rng.uniform(0.0, 2.0 * np.pi, size=(grid_height, grid_width))
        gradients_x = np.cos(angles)
        gradients_y = np.sin(angles)
        y = np.arange(height) * scale
        x = np.arange(width) * scale
        yy, xx = np.meshgrid(y, x, indexing="ij")
        x0 = np.floor(xx).astype(int)
        y0 = np.floor(yy).astype(int)
        x1, y1 = x0 + 1, y0 + 1
        xf, yf = xx - x0, yy - y0

        def fade(value: np.ndarray) -> np.ndarray:
            return value**3 * (value * (value * 6 - 15) + 10)

        u, v = fade(xf), fade(yf)
        n00 = gradients_x[y0, x0] * xf + gradients_y[y0, x0] * yf
        n10 = gradients_x[y0, x1] * (xf - 1) + gradients_y[y0, x1] * yf
        n01 = gradients_x[y1, x0] * xf + gradients_y[y1, x0] * (yf - 1)
        n11 = gradients_x[y1, x1] * (xf - 1) + gradients_y[y1, x1] * (yf - 1)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        noise = nx0 + v * (nx1 - nx0)
        minimum, maximum = noise.min(), noise.max()
        if maximum == minimum:
            return np.zeros(shape, dtype=float)
        return (noise - minimum) / (maximum - minimum)
