"""
Module: workspace.residual_workspace

Role:
- Own residual/root-stage runtime memory.

Public API:
- ResidualWorkspace

Notes:
- Packed residual semantics remain owned by the Kernel runtime.
- This module allocates reusable residual buffers and scratch arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from veqpy.kernels.numba_kernel.workspace.field_rows import (
    RESIDUAL_ROOT_FFN_PSIN,
    RESIDUAL_ROOT_PN_PSIN,
    RESIDUAL_ROOT_PSIN,
    RESIDUAL_ROOT_PSIN_R,
    RESIDUAL_ROOT_PSIN_RR,
    RESIDUAL_SURFACE_G,
    RESIDUAL_SURFACE_GPSIN_R,
    RESIDUAL_SURFACE_GPSIN_R_SIN_TB,
    RESIDUAL_SURFACE_GPSIN_Z,
)


@dataclass(init=False, slots=True)
class ResidualWorkspace:
    """Residual/root stage memory owner.

    ``root_fields`` shape: ``(5, Nr)`` with rows psin, psin_r, psin_rr,
    FFn_psin, and Pn_psin.

    ``surface_fields`` shape: ``(4, Nr, Nt)`` with rows G, G*psin_R,
    G*psin_Z, and G*psin_R*sin_tb.

    ``pack_scratch`` is the one-dimensional temporary buffer used by projection
    kernels. ``pack_scratch_rows`` stores reusable row reductions for the
    high-block packer. Neither scratch area has a persistent value after a pack
    call.
    """

    root_fields: np.ndarray
    packed_residual: np.ndarray
    surface_fields: np.ndarray
    pack_scratch: np.ndarray
    pack_scratch_rows: np.ndarray
    collocation_sqrt_weights: np.ndarray

    def __init__(
        self,
        *,
        nr: int,
        nt: int,
        x_size: int,
        radial_weights: np.ndarray,
        active_residual_block_count: int = 0,
    ) -> None:
        """Allocate residual/root-stage runtime memory."""

        if radial_weights.ndim != 1 or radial_weights.size != nr:
            raise ValueError(f"Invalid radial weights shape {radial_weights.shape}")

        self.root_fields = np.empty((5, nr), dtype=np.float64)
        self.packed_residual = np.empty(x_size, dtype=np.float64)
        self.surface_fields = np.empty((4, nr, nt), dtype=np.float64)
        self.pack_scratch = np.empty(nr, dtype=np.float64)
        self.pack_scratch_rows = np.empty(
            (max(1, int(active_residual_block_count) + 5), nr),
            dtype=np.float64,
        )
        poloidal_quadrature_weight = 2.0 * np.pi / max(nt, 1)
        self.collocation_sqrt_weights = np.sqrt(poloidal_quadrature_weight * radial_weights)

    @property
    def psin(self) -> np.ndarray:
        """Return normalized poloidal flux samples."""
        return self.root_fields[RESIDUAL_ROOT_PSIN]

    @property
    def psin_r(self) -> np.ndarray:
        """Return radial derivative of normalized poloidal flux."""
        return self.root_fields[RESIDUAL_ROOT_PSIN_R]

    @property
    def psin_rr(self) -> np.ndarray:
        """Return second radial derivative of normalized poloidal flux."""
        return self.root_fields[RESIDUAL_ROOT_PSIN_RR]

    @property
    def FFn_psin(self) -> np.ndarray:
        """Return normalized ``F F'`` source profile."""
        return self.root_fields[RESIDUAL_ROOT_FFN_PSIN]

    @property
    def Pn_psin(self) -> np.ndarray:
        """Return normalized pressure-gradient source profile."""
        return self.root_fields[RESIDUAL_ROOT_PN_PSIN]

    @property
    def G(self) -> np.ndarray:
        """Return compact Grad-Shafranov residual samples."""
        return self.surface_fields[RESIDUAL_SURFACE_G]

    @property
    def Gpsin_R(self) -> np.ndarray:
        """Return cached ``G * psin_R`` samples."""
        return self.surface_fields[RESIDUAL_SURFACE_GPSIN_R]

    @property
    def Gpsin_Z(self) -> np.ndarray:
        """Return cached ``G * psin_Z`` samples."""
        return self.surface_fields[RESIDUAL_SURFACE_GPSIN_Z]

    @property
    def Gpsin_R_sin_tb(self) -> np.ndarray:
        """Return cached ``G * psin_R * sin(theta_bar)`` samples."""
        return self.surface_fields[RESIDUAL_SURFACE_GPSIN_R_SIN_TB]
