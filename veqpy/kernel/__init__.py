"""Temporary VEQPy import bridge for the VEQlib-owned Numba backend."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NumbaKernel",
]

_EXPORTS = {"NumbaKernel": ("veqlib.numba_core.kernel", "_NumbaKernelImpl")}


def __getattr__(name: str) -> Any:
    """Resolve exported kernel-facade symbols lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
