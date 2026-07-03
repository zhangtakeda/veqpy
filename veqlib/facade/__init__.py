"""Python-facing VEQlib facade APIs.

This package owns typed runtime arguments, artifact lifecycle, and Python handle
helpers for the C++/CMake core under :mod:`veqlib.core`.
"""

from __future__ import annotations

from .builder import (
    CleanResult,
    PrepareError,
    PrepareResult,
    clean,
    prepare,
)
from .kernel import Kernel, build, solve
from .registry import KernelLoadError, KernelRegistry, LoadedKernel, SolverThreadError
from .solver import VEQlibSolver
from .source_semantics import materialize_kernel_source
from .types import (
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    TopologyError,
)

__all__ = [
    "Kernel",
    "PrepareResult",
    "KernelBoundary",
    "KernelRecipe",
    "PrepareError",
    "CleanResult",
    "KernelConfig",
    "KernelSource",
    "KernelLoadError",
    "KernelRegistry",
    "SolveResult",
    "KernelTopology",
    "LoadedKernel",
    "SolverThreadError",
    "TopologyError",
    "VEQlibSolver",
    "build",
    "prepare",
    "clean",
    "solve",
    "materialize_kernel_source",
]
