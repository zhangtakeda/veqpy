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

from .boundary import Boundary
from .equilibrium import Equilibrium
from .geqdsk import Geqdsk
from .grid import Grid
from .profile import Profile

__all__ = [
    "Equilibrium",
    "Grid",
    "Geqdsk",
    "Boundary",
    "Profile",
]
