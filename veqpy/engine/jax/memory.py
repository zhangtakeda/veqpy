"""Private host/device conversion helpers for the optional JAX backend."""

from __future__ import annotations

from typing import Any

import numpy as np


def device_put_array(jax_module: Any, value: np.ndarray) -> Any:
    """Copy a NumPy array to the active JAX device."""

    return jax_module.device_put(np.asarray(value))


def copy_device_array_to_numpy(value: Any) -> np.ndarray:
    """Copy a JAX value back to host NumPy.

    The host conversion itself establishes readiness for public NumPy results,
    so callers should avoid an additional ``block_until_ready`` unless they are
    measuring device-only timing.
    """

    return np.asarray(value)


def copy_device_array_into(value: Any, out: np.ndarray) -> None:
    """Synchronize and copy a JAX value into a caller-owned NumPy array."""

    host_value = copy_device_array_to_numpy(value)
    if host_value.shape != out.shape:
        raise ValueError(f"Expected device value with shape {out.shape}, got {host_value.shape}")
    np.copyto(out, host_value)
