"""Compatibility import surface for the pure ``veqlib.kernel`` API."""

from __future__ import annotations

from veqlib.kernel import (
    Kernel,
    KernelBoundary,
    KernelBuild,
    KernelInput,
    KernelResult,
    KernelSolve,
    KernelTopology,
    TopologyError,
    build,
)

__all__ = [
    "Kernel",
    "KernelBoundary",
    "KernelBuild",
    "KernelInput",
    "KernelResult",
    "KernelSolve",
    "KernelTopology",
    "TopologyError",
    "build",
]
