"""Python-facing VEQlib facade APIs.

This package owns typed Kernel arguments, backend-neutral handle construction,
source materialization, and public solve helpers. Backend-specific build and
runtime details stay in private VEQlib implementation packages.
"""

from __future__ import annotations

from importlib import import_module

from veqpy.types import (
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
    "PrepareResult": ("veqpy.kernels.cxx_kernel.builder", "PrepareResult"),
    "PrepareError": ("veqpy.kernels.cxx_kernel.builder", "PrepareError"),
    "CleanResult": ("veqpy.kernels.cxx_kernel.builder", "CleanResult"),
    "KernelLoadError": ("veqpy.kernels.cxx_kernel.registry", "KernelLoadError"),
    "KernelRegistry": ("veqpy.kernels.cxx_kernel.registry", "KernelRegistry"),
    "LoadedKernel": ("veqpy.kernels.cxx_kernel.registry", "LoadedKernel"),
    "SolverThreadError": ("veqpy.kernels.cxx_kernel.registry", "SolverThreadError"),
    "VEQlibSolver": ("veqpy.kernels.cxx_kernel.solver", "VEQlibSolver"),
    "build": ("veqlib.facade.kernel", "build"),
    "prepare": ("veqpy.kernels.cxx_kernel.builder", "prepare"),
    "clean": ("veqpy.kernels.cxx_kernel.builder", "clean"),
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
