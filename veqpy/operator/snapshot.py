"""
Module: operator.snapshot

Role:
- Materialize immutable model snapshots from refreshed operator runtime arrays.

Notes:
- Snapshot helpers copy runtime-owned arrays before returning model objects.
- They do not run stages, allocate runtime memory, or bind executable layouts.
"""

from __future__ import annotations

import numpy as np

from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.model.problem import Problem
from veqpy.model.profile import Profile
from veqpy.operator.packed_layout import decode_packed_blocks


def snapshot_equilibrium_from_runtime(
    *,
    x: np.ndarray,
    problem: Problem,
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
    alpha1: float,
    alpha2: float,
) -> Equilibrium:
    """Materialize an Equilibrium snapshot from current Operator runtime arrays."""

    coeff_values = decode_packed_blocks(
        x,
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    shape_profiles = snapshot_equilibrium_profiles(
        coeff_values,
        shape_profile_names=shape_profile_names,
        profile_index=profile_index,
        profile_offsets=profile_offsets,
        profile_scales=profile_scales,
        profile_powers=profile_powers,
        profile_envelope_powers=profile_envelope_powers,
        profile_amplitude_powers=profile_amplitude_powers,
    )
    return Equilibrium(
        R0=problem.R0,
        Z0=problem.Z0,
        B0=problem.B0,
        a=problem.a,
        grid=grid,
        shape_profiles=shape_profiles,
        psin=psin.copy(),
        FFn_psin=np.asarray(FFn_psin, dtype=np.float64).copy(),
        Pn_psin=Pn_psin.copy(),
        psin_r=psin_r.copy(),
        psin_rr=psin_rr.copy(),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
    )


def snapshot_equilibrium_profiles(
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
    """Snapshot passive shape-profile specs using the supplied packed state."""

    profiles: dict[str, Profile] = {}
    for name in shape_profile_names:
        p = profile_index[name]
        # Shape profiles are the only model profiles needed for geometry
        # reconstruction in Equilibrium; source/F/psin profiles are represented
        # by root fields and source derivatives instead.
        profiles[name] = snapshot_profile(
            coeff_values[p],
            offset=float(profile_offsets[p]),
            scale=float(profile_scales[p]),
            power=int(profile_powers[p]),
            envelope_power=int(profile_envelope_powers[p]),
            amplitude_power=float(profile_amplitude_powers[p]),
        )
    return profiles


def snapshot_profile(
    coeff_values: np.ndarray | None,
    *,
    offset: float,
    scale: float,
    power: int,
    envelope_power: int,
    amplitude_power: float,
) -> Profile:
    """Copy one passive profile spec and replace its active coefficients."""

    # Active shape profiles receive the solved coefficient block; passive ones
    # keep coeff=None and therefore remain pure offset/static profiles.
    coeff = None if coeff_values is None else np.asarray(coeff_values, dtype=np.float64).copy()
    return Profile(
        scale=scale,
        power=power,
        envelope_power=envelope_power,
        amplitude_power=amplitude_power,
        offset=offset,
        coeff=coeff,
    )
