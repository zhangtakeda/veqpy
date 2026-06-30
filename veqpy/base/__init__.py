"""
Module: base

Role:
- Expose shared base utilities for serialization, registries, and reactive caching.

Public API:
- Reactive
- depends_on
- Registry
- Serial
- SERIAL_TYPE_REGISTRY
- read_serializer
- write_serializer
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Reactive",
    "depends_on",
    "Registry",
    "Serial",
    "SERIAL_TYPE_REGISTRY",
    "read_serializer",
    "write_serializer",
]

_EXPORTS = {
    "Reactive": ("veqpy.base.reactive", "Reactive"),
    "depends_on": ("veqpy.base.reactive", "depends_on"),
    "Registry": ("veqpy.base.registry", "Registry"),
    "Serial": ("veqpy.base.serial", "Serial"),
    "SERIAL_TYPE_REGISTRY": ("veqpy.base.serial", "SERIAL_TYPE_REGISTRY"),
    "read_serializer": ("veqpy.base.serial", "read_serializer"),
    "write_serializer": ("veqpy.base.serial", "write_serializer"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported base utilities lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
