"""Python-facing VEQlib facade APIs.

This package owns typed runtime arguments, artifact lifecycle, and Python handle
helpers for the C++/CMake core under :mod:`veqlib.core`.
"""

from __future__ import annotations

from .builder import (
    KernelArtifact,
    KernelBuildError,
    KernelCleanResult,
    build_artifact,
    clean,
)
from .kernel import Kernel, build, solve
from .registry import KernelLoadError, KernelRegistry, LoadedKernel, SolverThreadError
from .solver import VEQlibSolver
from .types import (
    KernelBoundary,
    KernelBuildOptions,
    KernelConfig,
    KernelInput,
    KernelResult,
    KernelTopology,
    TopologyError,
)

__all__ = [
    "Kernel",
    "KernelArtifact",
    "KernelBoundary",
    "KernelBuildOptions",
    "KernelBuildError",
    "KernelCleanResult",
    "KernelConfig",
    "KernelInput",
    "KernelLoadError",
    "KernelRegistry",
    "KernelResult",
    "KernelTopology",
    "LoadedKernel",
    "SolverThreadError",
    "TopologyError",
    "VEQlibSolver",
    "build",
    "build_artifact",
    "clean",
    "solve",
]
