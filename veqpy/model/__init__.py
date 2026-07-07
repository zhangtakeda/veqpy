"""
Package: veqpy.model

Role:
- Define serializable model-layer objects for grids, profiles, GEQDSK data,
  and equilibria.
- Provide the public model import surface.

Public API:
- Grid, Profile, Geqdsk, and Equilibrium.

Dependencies:
- veqpy.base for Serial and Reactive infrastructure.
- veqpy.numerics for model-side quadrature, calculus, interpolation, and projection helpers.

Downstream:
- veqpy.kernels consumes model objects when building runtime cases and snapshots.
- Examples, docs, tests, and user code import model objects from this package root.

Design notes:
- Model objects describe physical state and diagnostics, not solver execution.
- Concrete files inside this package may import each other directly.
"""

from __future__ import annotations

from .equilibrium import Equilibrium
from .geqdsk import Geqdsk
from .grid import Grid
from .profile import Profile

__all__ = [
    "Equilibrium",
    "Grid",
    "Geqdsk",
    "Profile",
]
