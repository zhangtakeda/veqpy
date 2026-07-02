"""Python-facing VEQlib facade APIs.

This package owns typed runtime arguments, artifact lifecycle, and Python handle
helpers for the C++/CMake core under :mod:`veqlib.core`.
"""

from __future__ import annotations

from .builder import (
    CleanResult,
    CompileError,
    CompileResult,
    clean,
    compile,
)
from .kernel import Kernel, build, solve
from .registry import KernelLoadError, KernelRegistry, LoadedKernel, SolverThreadError
from .solver import VEQlibSolver
from .types import (
    KernelBoundary,
    KernelConfig,
    KernelInput,
    KernelRecipe,
    KernelTopology,
    SolveResult,
    TopologyError,
)

__all__ = [
    "Kernel",
    "CompileResult",
    "KernelBoundary",
    "KernelRecipe",
    "CompileError",
    "CleanResult",
    "KernelConfig",
    "KernelInput",
    "KernelLoadError",
    "KernelRegistry",
    "SolveResult",
    "KernelTopology",
    "LoadedKernel",
    "SolverThreadError",
    "TopologyError",
    "VEQlibSolver",
    "build",
    "compile",
    "clean",
    "solve",
]
