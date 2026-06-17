"""
Module: engine.backend

Role:
- Define backend vocabulary and backend-related errors.
- Keep optional backend configuration objects free of concrete backend imports.

Notes:
- This module must not import JAX. Optional backend import policy lives behind
  backend-specific runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["numba", "jax"]

SUPPORTED_BACKENDS: tuple[BackendName, ...] = ("numba", "jax")
DEFAULT_BACKEND: BackendName = "numba"


class BackendError(RuntimeError):
    """Base class for backend configuration and capability failures."""


class BackendFeatureError(BackendError):
    """Base class for backend feature/capability errors."""


class UnsupportedBackendFeature(BackendFeatureError):
    """Raised when a valid backend lacks support for a requested feature."""


class MissingOptionalBackendError(BackendError):
    """Raised when an optional backend dependency is required but unavailable."""


class InvalidBackendError(BackendError, ValueError):
    """Raised when a backend name is not recognized."""


@dataclass(frozen=True, slots=True)
class JaxBackendOptions:
    """Configuration intended for the optional JAX backend.

    The dataclass is backend-vocabulary only: it intentionally stores plain
    Python values and does not import or reference JAX objects.
    """

    platform: str | None = None
    enable_x64: bool = True
    preallocate: bool | None = None
    mem_fraction: float | None = None
    allocator: str | None = None
    donate_x: bool = False
    profile_memory: bool = False


def normalize_backend(backend: str | None) -> BackendName:
    """Normalize a backend spelling to a canonical backend name."""

    if backend is None:
        return DEFAULT_BACKEND
    name = str(backend).strip().lower()
    if name in SUPPORTED_BACKENDS:
        return name  # type: ignore[return-value]
    supported = ", ".join(repr(item) for item in SUPPORTED_BACKENDS)
    raise InvalidBackendError(f"Unsupported backend {backend!r}; supported backends: {supported}.")
