"""
Module: model.numerics

Role:
- Host model-facing numerical helpers used by grid and equilibrium objects.
"""

from __future__ import annotations

from .geometry import update_geometry_hot_auto
from .grid_workspace import GridWorkspace
from .interpolate import (
    barycentric_log_weights,
    interpolation_matrix,
)
from .profile_eval import update_profile

__all__ = [
    "GridWorkspace",
    "barycentric_log_weights",
    "interpolation_matrix",
    "update_geometry_hot_auto",
    "update_profile",
]
