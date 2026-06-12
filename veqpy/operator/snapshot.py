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
from veqpy.model.profile import Profile
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import decode_packed_blocks


def snapshot_equilibrium_from_runtime(
    *,
    x: np.ndarray,
    case: OperatorCase,
    grid: Grid,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    profile_names: tuple[str, ...],
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
    psin: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> Equilibrium:
    """Materialize an Equilibrium snapshot from current Operator runtime arrays."""

    # The packed x supplies only active coefficients.  Passive profile specs come
    # from profiles_by_name and are copied below so the Equilibrium is detached
    # from future Operator.refresh_case mutations.
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
        profiles_by_name=profiles_by_name,
    )
    return Equilibrium(
        R0=case.R0,
        Z0=case.Z0,
        B0=case.B0,
        a=case.a,
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
    profiles_by_name: dict[str, Profile],
) -> dict[str, Profile]:
    """Snapshot passive shape-profile specs using the supplied packed state."""

    return {
        # Shape profiles are the only model profiles needed for geometry
        # reconstruction in Equilibrium; source/F/psin profiles are represented
        # by root fields and source derivatives instead.
        name: snapshot_profile(profiles_by_name[name], coeff_values[profile_index[name]])
        for name in shape_profile_names
    }


def snapshot_profile(profile: Profile, coeff_values: np.ndarray | None) -> Profile:
    """Copy one passive profile spec and replace its active coefficients."""

    copied = profile.copy()
    # Active shape profiles receive the solved coefficient block; passive ones
    # keep coeff=None and therefore remain pure offset/static profiles.
    copied.coeff = (
        None if coeff_values is None else np.asarray(coeff_values, dtype=np.float64).copy()
    )
    return copied
