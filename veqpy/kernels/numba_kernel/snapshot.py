"""
Module: veqpy.kernels.numba_kernel.snapshot

Role:
- Build ``Equilibrium`` snapshots from Numba backend runtime state.
"""

from __future__ import annotations

import numpy as np

from veqpy.model import Equilibrium, Grid, Profile
from veqpy.model.equilibrium import _resample_equilibrium_root_fields

from .packed_layout import decode_packed_blocks


def snapshot_equilibrium_from_kernel_runtime(
    *,
    x: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    grid: Grid,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    profile_names: tuple[str, ...],
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_offsets: np.ndarray,
    profile_scales: np.ndarray,
    profile_powers: np.ndarray,
    profile_envelope_powers: np.ndarray,
    profile_amplitude_powers: np.ndarray,
    psin: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    p0: float,
    alpha1: float,
    alpha2: float,
    output_grid: Grid | None = None,
) -> Equilibrium:
    """Materialize an ``Equilibrium`` directly from Kernel runtime state."""

    coeff_values = decode_packed_blocks(
        x,
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    shape_profiles = _snapshot_equilibrium_profiles(
        coeff_values,
        shape_profile_names=shape_profile_names,
        profile_index=profile_index,
        profile_offsets=profile_offsets,
        profile_scales=profile_scales,
        profile_powers=profile_powers,
        profile_envelope_powers=profile_envelope_powers,
        profile_amplitude_powers=profile_amplitude_powers,
    )
    if output_grid is not None:
        psin, psin_r, psin_rr, FFn_psin, Pn_psin = _resample_equilibrium_root_fields(
            source_grid=grid,
            target_grid=output_grid,
            psin=psin,
            psin_r=psin_r,
            FFn_psin=FFn_psin,
            Pn_psin=Pn_psin,
        )
        grid = output_grid
    return Equilibrium(
        R0=float(R0),
        Z0=float(Z0),
        B0=float(B0),
        a=float(a),
        grid=grid,
        shape_profiles=shape_profiles,
        psin=psin.copy(),
        FFn_psin=np.asarray(FFn_psin, dtype=np.float64).copy(),
        Pn_psin=Pn_psin.copy(),
        psin_r=psin_r.copy(),
        psin_rr=psin_rr.copy(),
        p0=float(p0),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
    )


def _snapshot_equilibrium_profiles(
    coeff_values: tuple[np.ndarray | None, ...],
    *,
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_offsets: np.ndarray,
    profile_scales: np.ndarray,
    profile_powers: np.ndarray,
    profile_envelope_powers: np.ndarray,
    profile_amplitude_powers: np.ndarray,
) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    for name in shape_profile_names:
        profile_id = profile_index[name]
        coeff = coeff_values[profile_id]
        profiles[name] = Profile(
            scale=float(profile_scales[profile_id]),
            power=int(profile_powers[profile_id]),
            envelope_power=int(profile_envelope_powers[profile_id]),
            amplitude_power=float(profile_amplitude_powers[profile_id]),
            offset=float(profile_offsets[profile_id]),
            coeff=None if coeff is None else np.asarray(coeff, dtype=np.float64).copy(),
        )
    return profiles
