"""
Module: workspace.grid_workspace

Role:
- Hold static grid memory snapshots for operator hot paths.

Public API:
- GridWorkspace

Notes:
- ``GridWorkspace`` lowers ``Grid`` data into packed array views.
- Grid construction remains outside hot runtime kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.grid import Grid


@dataclass(frozen=True, slots=True)
class GridWorkspace:
    """Operator-level Grid snapshot: arrays and metadata required by the engine ABI.

    K_max is normalized: Grid.K_max=None maps to M_max.
    """

    Nr: int
    Nt: int
    M_max: int
    L_max: int
    K_max: int
    quadrature_scheme: str
    calculus_scheme: str
    K_values: np.ndarray
    weights: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray

    # rho           (Nr,)
    # x             (Nr,)
    # y             (Nr,)
    # rho_powers    (K_max+2, Nr)
    # T             (L_max+1, Nr)
    # T_r           (L_max+1, Nr)
    # T_rr          (L_max+1, Nr)
    radial_fields: np.ndarray  # (8+K_max+3*L_max, Nr)

    # theta         (Nt,)
    # cos_mtheta    (M_max+1, Nt)
    # sin_mtheta    (M_max+1, Nt)
    # m_cos_mtheta  (M_max+1, Nt)
    # m_sin_mtheta  (M_max+1, Nt)
    # m2_cos_mtheta (M_max+1, Nt)
    # m2_sin_mtheta (M_max+1, Nt)
    poloidal_fields: np.ndarray  # (7+6*M_max, Nt)

    @property
    def rho(self) -> np.ndarray:
        """Radial nodes packed for runtime kernels."""
        return self.radial_fields[0]

    @property
    def x(self) -> np.ndarray:
        """Chebyshev coordinate packed for runtime kernels."""
        return self.radial_fields[1]

    @property
    def y(self) -> np.ndarray:
        """Envelope coordinate packed for runtime kernels."""
        return self.radial_fields[2]

    @property
    def rho_powers(self) -> np.ndarray:
        """Packed powers of ``rho`` used by profile envelopes."""
        return self.radial_fields[3 : 5 + self.K_max]

    @property
    def T(self) -> np.ndarray:
        """Packed Chebyshev basis values."""
        return self.radial_fields[5 + self.K_max : 6 + self.K_max + self.L_max]

    @property
    def T_r(self) -> np.ndarray:
        """Packed first derivatives of the Chebyshev basis."""
        return self.radial_fields[6 + self.K_max + self.L_max : 7 + self.K_max + 2 * self.L_max]

    @property
    def T_rr(self) -> np.ndarray:
        """Packed second derivatives of the Chebyshev basis."""
        return self.radial_fields[7 + self.K_max + 2 * self.L_max : 8 + self.K_max + 3 * self.L_max]

    @property
    def theta(self) -> np.ndarray:
        """Poloidal angle nodes packed for runtime kernels."""
        return self.poloidal_fields[0]

    @property
    def cos_mtheta(self) -> np.ndarray:
        """Packed cosine Fourier table."""
        return self.poloidal_fields[1 : 2 + self.M_max]

    @property
    def sin_mtheta(self) -> np.ndarray:
        """Packed sine Fourier table."""
        return self.poloidal_fields[2 + self.M_max : 3 + 2 * self.M_max]

    @property
    def m_cos_mtheta(self) -> np.ndarray:
        """Packed first-derivative cosine table."""
        return self.poloidal_fields[3 + 2 * self.M_max : 4 + 3 * self.M_max]

    @property
    def m_sin_mtheta(self) -> np.ndarray:
        """Packed first-derivative sine table."""
        return self.poloidal_fields[4 + 3 * self.M_max : 5 + 4 * self.M_max]

    @property
    def m2_cos_mtheta(self) -> np.ndarray:
        """Packed second-derivative cosine table."""
        return self.poloidal_fields[5 + 4 * self.M_max : 6 + 5 * self.M_max]

    @property
    def m2_sin_mtheta(self) -> np.ndarray:
        """Packed second-derivative sine table."""
        return self.poloidal_fields[6 + 5 * self.M_max : 7 + 6 * self.M_max]

    @classmethod
    def from_grid(cls, grid: Grid) -> Self:
        """Lower ``Grid`` into static arrays consumed by runtime binding."""

        return cls(
            Nr=int(grid.Nr),
            Nt=int(grid.Nt),
            M_max=int(grid.M_max),
            L_max=int(grid.L_max),
            K_max=grid.K_max or grid.M_max,
            quadrature_scheme=grid.quadrature_scheme,
            calculus_scheme=grid.calculus_scheme,
            K_values=grid.K_values.copy(),
            weights=grid.weights.copy(),
            differentiator=grid.differentiator.copy(),
            accumulator=grid.accumulator.copy(),
            radial_fields=_pack_radial_fields(
                grid.rho, grid.x, grid.y, grid.rho_powers, grid.T, grid.T_r, grid.T_rr
            ),
            poloidal_fields=_pack_poloidal_fields(
                grid.theta,
                grid.cos_mtheta,
                grid.sin_mtheta,
                grid.m_cos_mtheta,
                grid.m_sin_mtheta,
                grid.m2_cos_mtheta,
                grid.m2_sin_mtheta,
            ),
        )

    def to_grid(self) -> Grid:
        """Rebuild a full Grid from the snapshot for Equilibrium materialization and
        other callers that need a real Grid."""
        from veqpy.model.grid import Grid

        return Grid(
            Nr=self.Nr,
            Nt=self.Nt,
            L_max=self.L_max,
            M_max=self.M_max,
            K_max=self.K_max,
            quadrature_scheme=self.quadrature_scheme,
            calculus_scheme=self.calculus_scheme,
        )


def _pack_radial_fields(
    rho: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rho_powers: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
) -> np.ndarray:
    """Pack radial fields into a read-only (R, Nr) 2D array according to the layout contract."""
    Nr = rho.shape[0]
    K_max = rho_powers.shape[0] - 2
    L_max = T.shape[0] - 1

    fields = np.empty((8 + K_max + 3 * L_max, Nr), dtype=np.float64)
    fields[0] = rho
    fields[1] = x
    fields[2] = y
    fields[3 : 5 + K_max] = rho_powers
    fields[5 + K_max : 6 + K_max + L_max] = T
    fields[6 + K_max + L_max : 7 + K_max + 2 * L_max] = T_r
    fields[7 + K_max + 2 * L_max : 8 + K_max + 3 * L_max] = T_rr
    fields.flags.writeable = False
    return fields


def _pack_poloidal_fields(
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
) -> np.ndarray:
    """Pack poloidal fields into a read-only (P, Nt) 2D array according to the layout contract."""
    Nt = theta.shape[0]
    M_max = cos_mtheta.shape[0] - 1

    fields = np.empty((7 + 6 * M_max, Nt), dtype=np.float64)
    fields[0] = theta
    fields[1 : 2 + M_max] = cos_mtheta
    fields[2 + M_max : 3 + 2 * M_max] = sin_mtheta
    fields[3 + 2 * M_max : 4 + 3 * M_max] = m_cos_mtheta
    fields[4 + 3 * M_max : 5 + 4 * M_max] = m_sin_mtheta
    fields[5 + 4 * M_max : 6 + 5 * M_max] = m2_cos_mtheta
    fields[6 + 5 * M_max : 7 + 6 * M_max] = m2_sin_mtheta
    fields.flags.writeable = False
    return fields
