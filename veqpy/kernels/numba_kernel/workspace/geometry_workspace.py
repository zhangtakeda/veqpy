"""
Module: veqpy.kernels.numba_kernel.workspace.geometry_workspace

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

from veqpy.kernels.numba_kernel.workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_KN_R,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_S_R,
    GEOMETRY_RADIAL_V_R,
    GEOMETRY_SURFACE_GRTDIVJR_T,
    GEOMETRY_SURFACE_GTTDIVJR,
    GEOMETRY_SURFACE_GTTDIVJR_R,
    GEOMETRY_SURFACE_J,
    GEOMETRY_SURFACE_JDIVR,
    GEOMETRY_SURFACE_R,
    GEOMETRY_SURFACE_R_T,
    GEOMETRY_SURFACE_SIN_TB,
    GEOMETRY_SURFACE_Z_T,
)


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

    @property
    def sin_tb_surface(self) -> np.ndarray:
        """Return ``sin(theta_bar)`` samples on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_SIN_TB]

    @property
    def R_surface(self) -> np.ndarray:
        """Return major-radius samples on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_R]

    @property
    def R_t_surface(self) -> np.ndarray:
        """Return poloidal derivatives of ``R`` on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_R_T]

    @property
    def Z_t_surface(self) -> np.ndarray:
        """Return poloidal derivatives of ``Z`` on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_Z_T]

    @property
    def J_surface(self) -> np.ndarray:
        """Return Jacobian samples on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_J]

    @property
    def JdivR_surface(self) -> np.ndarray:
        """Return ``J / R`` samples on ``(rho, theta)`` nodes."""
        return self.surface_fields[GEOMETRY_SURFACE_JDIVR]

    @property
    def grtdivJR_t_surface(self) -> np.ndarray:
        """Return theta derivatives of ``g_rt / (J R)``."""
        return self.surface_fields[GEOMETRY_SURFACE_GRTDIVJR_T]

    @property
    def gttdivJR_surface(self) -> np.ndarray:
        """Return ``g_tt / (J R)`` samples."""
        return self.surface_fields[GEOMETRY_SURFACE_GTTDIVJR]

    @property
    def gttdivJR_r_surface(self) -> np.ndarray:
        """Return radial derivatives of ``g_tt / (J R)``."""
        return self.surface_fields[GEOMETRY_SURFACE_GTTDIVJR_R]

    @property
    def S_r(self) -> np.ndarray:
        """Return radial derivative of enclosed surface area."""
        return self.radial_fields[GEOMETRY_RADIAL_S_R]

    @property
    def V_r(self) -> np.ndarray:
        """Return radial derivative of enclosed volume."""
        return self.radial_fields[GEOMETRY_RADIAL_V_R]

    @property
    def Kn(self) -> np.ndarray:
        """Return geometry factor ``K_n``."""
        return self.radial_fields[GEOMETRY_RADIAL_KN]

    @property
    def Kn_r(self) -> np.ndarray:
        """Return radial derivative of ``K_n``."""
        return self.radial_fields[GEOMETRY_RADIAL_KN_R]

    @property
    def Ln_r(self) -> np.ndarray:
        """Return geometry factor ``L_n,r``."""
        return self.radial_fields[GEOMETRY_RADIAL_LN_R]
