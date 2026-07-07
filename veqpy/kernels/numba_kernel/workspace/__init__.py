"""
Package: veqpy.kernels.numba_kernel.workspace

Role:
- Allocate and expose packed Numba runtime workspaces.
- Keep grid, profile, geometry, source, and residual memory ownership explicit.

Public API:
- Concrete workspace objects are imported from their concrete modules.

Dependencies:
- veqpy.kernels.numba_kernel field-row ABI and stage helpers.

Downstream:
- veqpy.kernels.numba_kernel.runtime owns workspace allocation for Kernel cases.
- Binding and stage modules consume workspace objects by direct module import.

Design notes:
- Scratch layout details stay in the concrete workspace modules.
- Workspaces are backend runtime state, not model-layer objects.
"""

from __future__ import annotations

__all__: list[str] = []
