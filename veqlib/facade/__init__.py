"""Python-facing VEQlib facade APIs.

This package owns typed runtime arguments, artifact lifecycle, and Python handle
helpers for the C++/CMake backend under :mod:`veqlib.cxx_core`.
"""

from __future__ import annotations

from importlib import import_module

from .types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    TopologyError,
)

_LAZY_EXPORTS = {
    "Kernel": ("veqlib.facade.kernel", "Kernel"),
    "PrepareResult": ("veqlib.cxx_core.builder", "PrepareResult"),
    "PrepareError": ("veqlib.cxx_core.builder", "PrepareError"),
    "CleanResult": ("veqlib.cxx_core.builder", "CleanResult"),
    "KernelLoadError": ("veqlib.cxx_core.registry", "KernelLoadError"),
    "KernelRegistry": ("veqlib.cxx_core.registry", "KernelRegistry"),
    "LoadedKernel": ("veqlib.cxx_core.registry", "LoadedKernel"),
    "SolverThreadError": ("veqlib.cxx_core.registry", "SolverThreadError"),
    "VEQlibSolver": ("veqlib.cxx_core.solver", "VEQlibSolver"),
    "build": ("veqlib.facade.kernel", "build"),
    "prepare": ("veqlib.cxx_core.builder", "prepare"),
    "clean": ("veqlib.cxx_core.builder", "clean"),
    "solve": ("veqlib.facade.kernel", "solve"),
}

__all__ = [
    "Kernel",
    "PrepareResult",
    "KernelBoundary",
    "KernelPrepareResult",
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
]


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
