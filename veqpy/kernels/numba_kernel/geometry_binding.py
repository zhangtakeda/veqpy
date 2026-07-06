"""
Module: layout.geometry_binding

Role:
- Bind geometry stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from geometry runtime memory ownership.

Notes:
- This module binds preallocated arrays and backend callables; it does not allocate memory.
- Numerical kernels remain private to this backend package.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veqpy.kernels.numba_kernel.geometry_stage import update_geometry_hot_auto

from .numba_source import update_fourier_family_fields


def build_geometry_stage_runner(
    *,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    profile_fields: np.ndarray,
    c_family_source_profile_ids: np.ndarray,
    s_family_source_profile_ids: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    surface_fields: np.ndarray,
    radial_fields: np.ndarray,
    grid_radial_fields: np.ndarray,
    grid_poloidal_fields: np.ndarray,
) -> Callable[[], None]:
    """Bind a geometry stage closure against preallocated runtime arrays."""
    c_effective_order = int(c_effective_order)
    s_effective_order = int(s_effective_order)
    a = float(a)
    R0 = float(R0)
    Z0 = float(Z0)

    def runner() -> None:
        # Refresh Fourier family fields first: geometry consumes compact c/s
        # arrays, while the owning profile fields may have just changed in Stage A.
        update_fourier_family_fields(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            profile_fields,
            c_family_source_profile_ids,
            s_family_source_profile_ids,
            c_effective_order,
            s_effective_order,
        )
        # update_geometry_hot_auto overwrites all surface/radial rows, so callers can
        # reuse the same workspace without clearing it between residual calls.
        update_geometry_hot_auto(
            surface_fields,
            radial_fields,
            a,
            R0,
            Z0,
            grid_radial_fields,
            grid_poloidal_fields,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_effective_order,
            s_effective_order,
        )

    return runner
