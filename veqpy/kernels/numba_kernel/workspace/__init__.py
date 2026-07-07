"""
Package: veqpy.kernels.numba_kernel.workspace

Role:
- Allocate and expose packed Numba runtime workspaces.
- Keep grid, profile, geometry, source, and residual memory ownership explicit.

Public API:
- GridWorkspace, GeometryWorkspace, ProfileWorkspace, ResidualWorkspace, and SourceWorkspace.
- allocate_runtime_state.

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

from veqpy.kernels.numba_kernel.workspace.grid_workspace import GridWorkspace

from .allocation import allocate_runtime_state
from .geometry_workspace import GeometryWorkspace
from .profile_workspace import ProfileWorkspace
from .residual_workspace import ResidualWorkspace
from .source_workspace import SourceWorkspace

__all__ = [
    "GeometryWorkspace",
    "GridWorkspace",
    "ProfileWorkspace",
    "ResidualWorkspace",
    "SourceWorkspace",
    "allocate_runtime_state",
]
