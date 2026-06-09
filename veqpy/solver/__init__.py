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

from veqpy.solver.solver import Solver
from veqpy.solver.solver_config import SolverConfig
from veqpy.solver.solver_record import SolverRecord
from veqpy.solver.solver_result import SolverResult

__all__ = [
    "Solver",
    "SolverConfig",
    "SolverRecord",
    "SolverResult",
]
