"""
Module: workspace.residual_workspace

Role:
- Own residual/root-stage runtime memory.

Public API:
- ResidualWorkspace

Notes:
- Packed residual semantics remain owned by ``veqpy.operator``.
- This module allocates reusable residual buffers and scratch arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
