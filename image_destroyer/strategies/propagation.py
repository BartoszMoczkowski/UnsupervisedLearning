from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_destroyer.strategies._common import (
    destruction_limit,
    neighbor_indices,
    source_mask,
)


@dataclass(frozen=True)
class PropagationConfig:
    min_sources: int = 3
    max_sources: int = 10
    max_destroyed_fraction: float = 1 / 16
    start_probability: float = 0.97
    decay: float = 0.9
    max_generations: int = 100

    def __post_init__(self) -> None:
        if self.min_sources < 0 or self.max_sources < self.min_sources:
            raise ValueError("source counts must satisfy 0 <= min_sources <= max_sources")
        if not 0 <= self.max_destroyed_fraction <= 1:
            raise ValueError("max_destroyed_fraction must be between 0 and 1")
        if not 0 <= self.start_probability <= 1:
            raise ValueError("start_probability must be between 0 and 1")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay must be between 0 and 1")
        if self.max_generations < 0:
            raise ValueError("max_generations cannot be negative")


@dataclass(frozen=True)
class PropagationStrategy:
    config: PropagationConfig = PropagationConfig()

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
        frontier = np.flatnonzero(mask)
        destroyed_count = len(frontier)

        for generation in range(self.config.max_generations):
            remaining = limit - destroyed_count
            if remaining <= 0:
                break

            probability = self.config.start_probability * self.config.decay**generation
            neighbors, neighbor_counts = np.unique(
                neighbor_indices(frontier, shape),
                return_counts=True,
            )
            available = ~mask.flat[neighbors]
            candidates = neighbors[available]
            if not len(candidates):
                break
            chances = 1.0 - np.power(1.0 - probability, neighbor_counts[available])
            frontier = candidates[rng.random(len(candidates)) < chances]
            if not len(frontier):
                break
            if len(frontier) > remaining:
                frontier = rng.choice(frontier, size=remaining, replace=False)

            mask.flat[frontier] = True
            destroyed_count += len(frontier)

        return mask
