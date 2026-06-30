"""
Module: solver.__init__

Role:
- Export public solver-layer types and package-level entrypoints.

Public API:
- Solver
- SolverConfig
- SolverRecord
- SolverResult

Notes:
- This module only provides package-level exports.
- Does not own packed layout definitions, engine backend selection, or benchmark organization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Solver",
    "SolverConfig",
    "SolverRecord",
    "SolverResult",
]

_EXPORTS = {
    "Solver": ("veqpy.solver.solver", "Solver"),
    "SolverConfig": ("veqpy.solver.solver_config", "SolverConfig"),
    "SolverRecord": ("veqpy.solver.solver_record", "SolverRecord"),
    "SolverResult": ("veqpy.solver.solver_result", "SolverResult"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported solver-layer types lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
