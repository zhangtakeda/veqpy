"""
Module: layout.profile_binding

Role:
- Bind executable profile-stage callables from preallocated workspace arrays.

Notes:
- Profile object construction and refresh semantics live in ``veqpy.operator.profile_runtime``.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def build_profile_stage_runner(
    *,
    active_profile_ids: np.ndarray,
    profile_fields: np.ndarray,
    profile_rp_fields: np.ndarray,
    profile_env_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_coeff_index_rows: np.ndarray,
    active_lengths: np.ndarray,
    update_profiles_packed_bulk: Callable,
) -> Callable[[np.ndarray], None]:
    """Bind the profile stage runner against workspace arrays and backend kernel."""

    if active_profile_ids.size == 0:
        return lambda x: None

    def runner(x: np.ndarray) -> None:
        update_profiles_packed_bulk(
            profile_fields,
            profile_rp_fields,
            profile_env_fields,
            active_profile_ids,
            T,
            T_r,
            T_rr,
            active_offsets,
            active_scales,
            x,
            active_coeff_index_rows,
            active_lengths,
        )

    return runner
