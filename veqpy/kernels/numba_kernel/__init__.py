"""
Package: veqpy.kernels.numba_kernel

Role:
- Implement the private Numba backend selected by _BuildPolicy.backend.
- Own packed layouts, runtime workspaces, source routes, initialization,
  residual assembly, solve adapters, and equilibrium snapshot construction.

Public API:
- No package-root public symbols.

Dependencies:
- veqpy.kernels for Kernel dataclasses and ABI helpers.
- veqpy.numerics for Grid tables and fusionprime-base for Equilibrium
  materialization.
- veqpy.numerics for interpolation and numerical helper routines.

Downstream:
- veqpy.kernels.dispatch instantiates this backend for backend="numba".
- Backend tests and benchmarks may import concrete implementation modules directly.

Design notes:
- The package root documents backend ownership; implementation symbols live in
  concrete modules.
- Runtime packed ABI details stay inside this package.
"""

from __future__ import annotations

__all__: list[str] = []
