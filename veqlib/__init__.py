"""VEQlib package root."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelPrepareResult",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "SolveResult",
    "build",
    "solve",
]

_EXPORTS = {name: ("veqlib.facade", name) for name in __all__}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
