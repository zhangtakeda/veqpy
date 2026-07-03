"""Python-facing VEQPy Numba facade APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Kernel",
    "NumbaKernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "SolveResult",
    "build",
]

_EXPORTS = {
    "Kernel": ("veqpy.facade.kernel", "Kernel"),
    "NumbaKernel": ("veqpy.facade.kernel", "NumbaKernel"),
    "build": ("veqpy.facade.kernel", "build"),
    "KernelBoundary": ("veqlib.facade", "KernelBoundary"),
    "KernelConfig": ("veqlib.facade", "KernelConfig"),
    "KernelRecipe": ("veqlib.facade", "KernelRecipe"),
    "KernelSource": ("veqlib.facade", "KernelSource"),
    "KernelTopology": ("veqlib.facade", "KernelTopology"),
    "SolveResult": ("veqlib.facade", "SolveResult"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported Numba facade symbols lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
