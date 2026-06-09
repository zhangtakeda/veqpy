"""
Module: workspace.__init__

Role:
- Export public workspace interfaces used outside ``veqpy.workspace``.

Public API:
- GridWorkspace
- allocate_runtime_state

Notes:
- Stage-local workspace internals stay in their owning modules.
- Package roots are the only modules that declare ``__all__``.
"""

from __future__ import annotations

from veqpy.workspace.allocation import allocate_runtime_state
from veqpy.workspace.grid_workspace import GridWorkspace

__all__ = [
    "GridWorkspace",
    "allocate_runtime_state",
]
