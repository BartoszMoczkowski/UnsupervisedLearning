from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


class MaskFilter(Protocol):
    def apply(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Return a transformed mask with unchanged dimensions."""


@dataclass(frozen=True)
class BinaryClosing:
    size: int = 3

    def __post_init__(self) -> None:
        if self.size < 3 or self.size % 2 == 0:
            raise ValueError("size must be an odd integer of at least 3")

    def apply(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        closed = mask_image.filter(ImageFilter.MaxFilter(self.size))
        closed = closed.filter(ImageFilter.MinFilter(self.size))
        return np.asarray(closed, dtype=np.uint8) > 0
