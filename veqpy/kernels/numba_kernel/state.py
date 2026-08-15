"""Private packed-state coercion shared by the backend implementations."""

from __future__ import annotations

from typing import Any

import numpy as np


def coerce_initial_state(value: Any, x_size: int) -> np.ndarray:
    """Return one owned finite packed initial state for a fixed topology."""

    packed = np.asarray(value, dtype=np.float64)
    if packed.ndim != 1 or packed.shape != (int(x_size),):
        raise ValueError(f"x0 must have shape ({int(x_size)},), got {packed.shape}")
    if not np.all(np.isfinite(packed)):
        raise ValueError("x0 must contain only finite values")
    return np.ascontiguousarray(packed, dtype=np.float64).copy()
