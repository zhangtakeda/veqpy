"""
Module: workspace.geometry_workspace

Role:
- Own geometry-stage runtime memory.

Public API:
- GeometryWorkspace

Notes:
- Geometry kernels consume the arrays allocated here.
- This module does not bind executable layout callables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(init=False, slots=True)
class GeometryWorkspace:
    """Geometry stage memory owner.

    ``surface_fields`` shape: ``(9, Nr, Nt)`` with rows:
    sin_tb, R, R_t, Z_t, J, JdivR, grtdivJR_t, gttdivJR, gttdivJR_r.

    ``radial_fields`` shape: ``(5, Nr)`` with rows:
    S_r, V_r, Kn, Kn_r, Ln_r.
    """

    surface_fields: np.ndarray
    radial_fields: np.ndarray

    def __init__(self, *, nr: int, nt: int) -> None:
        """Allocate geometry-stage runtime memory."""

        self.surface_fields = np.empty((9, nr, nt), dtype=np.float64)
        self.radial_fields = np.empty((5, nr), dtype=np.float64)
