from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_destroyer.strategies._common import (
    destruction_limit,
    neighbor_indices,
    source_mask,
)


@dataclass(frozen=True)
class CellularAutomataConfig:
    min_sources: int = 2
    max_sources: int = 5
    max_destroyed_fraction: float = 1 / 16
    max_generations: int = 80
    base_probability: float = 0.12
    neighbor_probability: float = 0.16

    def __post_init__(self) -> None:
        if self.min_sources < 0 or self.max_sources < self.min_sources:
            raise ValueError("source counts must satisfy 0 <= min_sources <= max_sources")
        if not 0 <= self.max_destroyed_fraction <= 1:
            raise ValueError("max_destroyed_fraction must be between 0 and 1")
        if self.max_generations < 0:
            raise ValueError("max_generations cannot be negative")
        if not 0 <= self.base_probability <= 1:
            raise ValueError("base_probability must be between 0 and 1")
        if not 0 <= self.neighbor_probability <= 1:
            raise ValueError("neighbor_probability must be between 0 and 1")


@dataclass(frozen=True)
class CellularAutomataStrategy:
    config: CellularAutomataConfig = CellularAutomataConfig()

    def generate_mask(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        limit = destruction_limit(shape, self.config.max_destroyed_fraction)
        mask = source_mask(
            shape,
            self.config.min_sources,
            self.config.max_sources,
            limit,
            rng,
        )
        height, width = shape
        mask_flat = mask.ravel()
        destroyed_count = int(mask.sum())
        candidates = np.unique(
            neighbor_indices(np.flatnonzero(mask_flat), shape)
        )
        candidates = candidates[~mask_flat[candidates]]

        for _ in range(self.config.max_generations):
            remaining = limit - destroyed_count
            if remaining <= 0 or not len(candidates):
                break

            neighbors = self._count_neighbors(mask_flat, candidates, shape)
            probability = np.clip(
                self.config.base_probability
                + neighbors * self.config.neighbor_probability,
                0.0,
                1.0,
            )
            successful = (neighbors > 0) & (
                rng.random(len(candidates)) < probability
            )
            if not successful.any():
                break

            transition = candidates[successful]
            if len(transition) > remaining:
                transition = rng.choice(transition, size=remaining, replace=False)
            mask_flat[transition] = True
            destroyed_count += len(transition)
            if destroyed_count >= limit:
                break

            candidates = np.unique(
                np.concatenate(
                    (
                        candidates[~successful],
                        neighbor_indices(transition, shape),
                    )
                )
            )
            candidates = candidates[~mask_flat[candidates]]

        return mask

    @staticmethod
    def _count_neighbors(
        mask: np.ndarray,
        candidates: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        y, x = np.divmod(candidates, width)
        neighbors = np.zeros(len(candidates), dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_y, neighbor_x = y + dy, x + dx
                valid = (
                    (neighbor_y >= 0)
                    & (neighbor_y < height)
                    & (neighbor_x >= 0)
                    & (neighbor_x < width)
                )
                neighbors[valid] += mask[
                    neighbor_y[valid] * width + neighbor_x[valid]
                ]
        return neighbors
