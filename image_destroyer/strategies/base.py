from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class MaskStrategy(Protocol):
    def generate_mask(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Generate a 2D boolean mask; True marks pixels to destroy."""
