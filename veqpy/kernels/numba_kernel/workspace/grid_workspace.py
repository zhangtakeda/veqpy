"""
Module: veqpy.kernels.numba_kernel.workspace.grid_workspace

Role:
- Hold static grid memory snapshots for kernel hot paths.

Public API:
- GridWorkspace

Notes:
- ``GridWorkspace`` lowers ``Grid`` data into packed array views.
- Grid construction remains outside hot runtime kernels.
- Hot-path bindings consume ``radial_fields`` / ``poloidal_fields`` plus grid
  metadata; row properties are semantic views for debugging and row contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from veqpy.kernels.numba_kernel.workspace.field_rows import (
    GRID_POLOIDAL_COS_MTHETA_START,
    GRID_POLOIDAL_THETA,
    GRID_RADIAL_R,
    GRID_RADIAL_R_POWERS_START,
    GRID_RADIAL_X,
    GRID_RADIAL_Y,
)
from veqpy.numerics import interpolation_matrix

if TYPE_CHECKING:
    from veqpy.model.grid import Grid


@dataclass(frozen=True, slots=True)
class GridWorkspace:
    """Kernel Grid snapshot: arrays and metadata required by the runtime ABI.

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

    # r           (Nr,)
    # x             (Nr,)
    # y             (Nr,)
    # r_powers    (K_max+2, Nr)
    # T             (L_max+1, Nr)
    # T_r           (L_max+1, Nr)
    # T_rr          (L_max+1, Nr)
    # axis_weights  (Nr,): interpolation from radial nodes to r=0
    # edge_weights  (Nr,): interpolation from radial nodes to r=1
    radial_fields: np.ndarray  # (10+K_max+3*L_max, Nr)

    # theta         (Nt,)
    # cos_mtheta    (M_max+1, Nt)
    # sin_mtheta    (M_max+1, Nt)
    # m_cos_mtheta  (M_max+1, Nt)
    # m_sin_mtheta  (M_max+1, Nt)
    # m2_cos_mtheta (M_max+1, Nt)
    # m2_sin_mtheta (M_max+1, Nt)
    poloidal_fields: np.ndarray  # (7+6*M_max, Nt)

    @property
    def r(self) -> np.ndarray:
        """Radial nodes packed for runtime kernels."""
        return self.radial_fields[GRID_RADIAL_R]

    @property
    def x(self) -> np.ndarray:
        """Chebyshev coordinate packed for runtime kernels."""
        return self.radial_fields[GRID_RADIAL_X]

    @property
    def y(self) -> np.ndarray:
        """Envelope coordinate packed for runtime kernels."""
        return self.radial_fields[GRID_RADIAL_Y]

    @property
    def r_powers(self) -> np.ndarray:
        """Packed powers of ``r`` used by profile envelopes."""
        start = GRID_RADIAL_R_POWERS_START
        return self.radial_fields[start : start + self.K_max + 2]

    @property
    def T(self) -> np.ndarray:
        """Packed Chebyshev basis values."""
        start = GRID_RADIAL_R_POWERS_START + self.K_max + 2
        return self.radial_fields[start : start + self.L_max + 1]

    @property
    def T_r(self) -> np.ndarray:
        """Packed first derivatives of the Chebyshev basis."""
        start = GRID_RADIAL_R_POWERS_START + self.K_max + self.L_max + 3
        return self.radial_fields[start : start + self.L_max + 1]

    @property
    def T_rr(self) -> np.ndarray:
        """Packed second derivatives of the Chebyshev basis."""
        start = GRID_RADIAL_R_POWERS_START + self.K_max + 2 * self.L_max + 4
        return self.radial_fields[start : start + self.L_max + 1]

    @property
    def edge_interpolation_weights(self) -> np.ndarray:
        """Weights that evaluate a radial nodal field at the LCFS, ``r=1``."""
        return self.radial_fields[-1]

    @property
    def axis_interpolation_weights(self) -> np.ndarray:
        """Weights that evaluate a radial nodal field at the axis, ``r=0``."""
        return self.radial_fields[-2]

    @property
    def theta(self) -> np.ndarray:
        """Poloidal angle nodes packed for runtime kernels."""
        return self.poloidal_fields[GRID_POLOIDAL_THETA]

    @property
    def cos_mtheta(self) -> np.ndarray:
        """Packed cosine Fourier table."""
        start = GRID_POLOIDAL_COS_MTHETA_START
        return self.poloidal_fields[start : start + self.M_max + 1]

    @property
    def sin_mtheta(self) -> np.ndarray:
        """Packed sine Fourier table."""
        start = GRID_POLOIDAL_COS_MTHETA_START + self.M_max + 1
        return self.poloidal_fields[start : start + self.M_max + 1]

    @property
    def m_cos_mtheta(self) -> np.ndarray:
        """Packed first-derivative cosine table."""
        start = GRID_POLOIDAL_COS_MTHETA_START + 2 * (self.M_max + 1)
        return self.poloidal_fields[start : start + self.M_max + 1]

    @property
    def m_sin_mtheta(self) -> np.ndarray:
        """Packed first-derivative sine table."""
        start = GRID_POLOIDAL_COS_MTHETA_START + 3 * (self.M_max + 1)
        return self.poloidal_fields[start : start + self.M_max + 1]

    @property
    def m2_cos_mtheta(self) -> np.ndarray:
        """Packed second-derivative cosine table."""
        start = GRID_POLOIDAL_COS_MTHETA_START + 4 * (self.M_max + 1)
        return self.poloidal_fields[start : start + self.M_max + 1]

    @property
    def m2_sin_mtheta(self) -> np.ndarray:
        """Packed second-derivative sine table."""
        start = GRID_POLOIDAL_COS_MTHETA_START + 5 * (self.M_max + 1)
        return self.poloidal_fields[start : start + self.M_max + 1]

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
                grid.r,
                grid.x,
                grid.y,
                grid.r_powers,
                grid.T,
                grid.T_r,
                grid.T_rr,
                grid.K_max or grid.M_max,
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
    r: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r_powers: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    K_max: int,
) -> np.ndarray:
    """Pack radial fields into a read-only (R, Nr) 2D array according to the layout contract."""
    Nr = r.shape[0]
    L_max = T.shape[0] - 1
    K_max = int(K_max)

    fields = np.empty((10 + K_max + 3 * L_max, Nr), dtype=np.float64)
    fields[GRID_RADIAL_R] = r
    fields[GRID_RADIAL_X] = x
    fields[GRID_RADIAL_Y] = y

    r_start = GRID_RADIAL_R_POWERS_START
    r_stop = r_start + K_max + 2
    T_start = r_stop
    T_stop = T_start + L_max + 1
    T_r_stop = T_stop + L_max + 1

    _copy_or_extend_r_powers(fields[r_start:r_stop], r, r_powers)
    fields[T_start:T_stop] = T
    fields[T_stop:T_r_stop] = T_r
    fields[T_r_stop : T_r_stop + L_max + 1] = T_rr
    endpoint_weights = interpolation_matrix(
        r,
        np.array([0.0, 1.0], dtype=np.float64),
    )
    fields[-2] = endpoint_weights[0]
    fields[-1] = endpoint_weights[1]
    fields.flags.writeable = False
    return fields


def _copy_or_extend_r_powers(out: np.ndarray, r: np.ndarray, r_powers: np.ndarray) -> None:
    """Fill the workspace ABI's fixed r-power block, padding powers if needed."""

    copied = min(out.shape[0], r_powers.shape[0])
    out[:copied] = r_powers[:copied]
    for power in range(copied, out.shape[0]):
        out[power] = r**power


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
    fields[GRID_POLOIDAL_THETA] = theta

    block = M_max + 1
    start = GRID_POLOIDAL_COS_MTHETA_START
    fields[start : start + block] = cos_mtheta
    start += block
    fields[start : start + block] = sin_mtheta
    start += block
    fields[start : start + block] = m_cos_mtheta
    start += block
    fields[start : start + block] = m_sin_mtheta
    start += block
    fields[start : start + block] = m2_cos_mtheta
    start += block
    fields[start : start + block] = m2_sin_mtheta
    fields.flags.writeable = False
    return fields
