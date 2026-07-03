"""
Module: model.__init__

Role:
- Export public model-layer types and package-level entrypoints.

Public API:
- Boundary
- Grid
- Profile
- Geqdsk
- Equilibrium

Notes:
- This module only provides package-level exports.
- Does not own packed runtime state, solver policy, or backend selection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Equilibrium",
    "Grid",
    "Geqdsk",
    "Boundary",
    "Profile",
]

_EXPORTS = {
    "Boundary": ("veqpy.model.boundary", "Boundary"),
    "Equilibrium": ("veqpy.model.equilibrium", "Equilibrium"),
    "Geqdsk": ("veqpy.model.geqdsk", "Geqdsk"),
    "Grid": ("veqpy.model.grid", "Grid"),
    "Profile": ("veqpy.model.profile", "Profile"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported model-layer types lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
