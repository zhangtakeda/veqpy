"""
Module: layout.__init__

Role:
- Mark ``veqpy.layout`` as the executable layout package.

Public API:
- KernelLayout
- build_kernel_layout

Notes:
- Package-root exports are the cross-submodule layout contract.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "KernelLayout",
    "build_kernel_layout",
]

_EXPORTS = {
    "KernelLayout": ("veqpy.layout.runtime", "KernelLayout"),
    "build_kernel_layout": ("veqpy.layout.binding", "build_kernel_layout"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported layout symbols lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
