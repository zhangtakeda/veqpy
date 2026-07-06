"""Public Kernel wrapper and typed Kernel contract."""

from __future__ import annotations

from .errors import TopologyError
from .kernel import Kernel
from .types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    config_with_overrides,
)

__all__ = [
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelPrepareResult",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "SolveResult",
    "TopologyError",
    "config_with_overrides",
]
