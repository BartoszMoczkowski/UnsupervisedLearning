from __future__ import annotations

import numpy as np


def destruction_limit(shape: tuple[int, int], fraction: float) -> int:
    return int(shape[0] * shape[1] * fraction)


def neighbor_indices(indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not len(indices):
        return np.empty(0, dtype=np.intp)

    height, width = shape
    y, x = np.divmod(indices, width)
    adjacent_indices = []
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
            adjacent_indices.append(
                neighbor_y[valid] * width + neighbor_x[valid]
            )
    return np.concatenate(adjacent_indices)


def select_with_limit(
    candidates: np.ndarray,
    remaining: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if remaining <= 0:
        return np.zeros(candidates.shape, dtype=bool)

    indices = np.flatnonzero(candidates)
    if len(indices) <= remaining:
        return candidates.astype(bool, copy=True)

    selected = rng.choice(indices, size=remaining, replace=False)
    result = np.zeros(candidates.size, dtype=bool)
    result[selected] = True
    return result.reshape(candidates.shape)


def source_mask(
    shape: tuple[int, int],
    minimum: int,
    maximum: int,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = min(int(rng.integers(minimum, maximum + 1)), limit)
    count = min(count, shape[0] * shape[1])
    mask = np.zeros(shape, dtype=bool)
    if count:
        indices = rng.choice(mask.size, size=count, replace=False)
        mask.flat[indices] = True
    return mask
