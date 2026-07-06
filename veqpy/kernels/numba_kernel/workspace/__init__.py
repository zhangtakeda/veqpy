"""
Module: workspace.__init__

Role:
- Expose the workspace package-root contract used outside ``veqpy.kernels.numba_kernel.workspace``.

Public API:
- GridWorkspace and allocate_runtime_state for operator construction
- Workspace classes consumed by engine/layout/operator

Notes:
- Scratch semantics and allocation details stay in their owning modules.
- Package roots are the only modules that declare ``__all__``.
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
