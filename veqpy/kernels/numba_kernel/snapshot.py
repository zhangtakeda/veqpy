"""
Module: veqpy.kernels.numba_kernel.snapshot

Role:
- Build ``Equilibrium`` snapshots from Numba backend runtime state.
"""

from __future__ import annotations

import numpy as np
from fusionprime_base import MU0, Equilibrium, Geometry

from veqpy.model.equilibrium import _resample_equilibrium_root_fields
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

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
    """Materialize a new frozen base ``Equilibrium`` from runtime state."""

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
    geometry = _base_geometry_from_profiles(
        grid=grid,
        R0=R0,
        Z0=Z0,
        a=a,
        B0=B0,
        shape_profiles=shape_profiles,
    )
    # The direct runtime stores normalized source fields.  alpha1 and alpha2
    # are copied into the physical base roots here; no backend-owned array is
    # retained by the frozen State.
    alpha1 = float(alpha1)
    alpha2 = float(alpha2)
    return Equilibrium(
        geometry=geometry,
        FF_psi=np.array(FFn_psin, dtype=np.float64, copy=True) * alpha1,
        P_psi=np.array(Pn_psin, dtype=np.float64, copy=True) * alpha1 / MU0,
        psi_r=np.array(psin_r, dtype=np.float64, copy=True) * alpha2,
        psi_rr=np.array(psin_rr, dtype=np.float64, copy=True) * alpha2,
        B0=float(B0),
        P0=float(p0),
    ).freeze()


def _base_geometry_from_profiles(
    *,
    grid: Grid,
    R0: float,
    Z0: float,
    a: float,
    B0: float,
    shape_profiles: dict[str, Profile],
) -> Geometry:
    """Translate the numerical profile family to base Geometry roots."""

    del B0
    m_max = int(grid.M_max)
    l_max = int(grid.L_max)

    def coefficients(name: str) -> np.ndarray:
        profile = shape_profiles.get(name)
        result = np.zeros(l_max + 1, dtype=np.float64)
        if profile is not None and profile.coeff is not None:
            count = min(result.size, profile.coeff.size)
            result[:count] = np.asarray(profile.coeff[:count], dtype=np.float64)
        return result

    c_lcfs = np.zeros(m_max + 1, dtype=np.float64)
    s_lcfs = np.zeros(m_max, dtype=np.float64)
    c_coeffs = np.zeros((m_max + 1, l_max + 1), dtype=np.float64)
    s_coeffs = np.zeros((m_max, l_max + 1), dtype=np.float64)
    for harmonic in range(m_max + 1):
        profile = shape_profiles.get(f"c{harmonic}")
        if profile is not None:
            c_lcfs[harmonic] = float(profile.offset)
        c_coeffs[harmonic] = coefficients(f"c{harmonic}")
    for harmonic in range(1, m_max + 1):
        profile = shape_profiles.get(f"s{harmonic}")
        if profile is not None:
            s_lcfs[harmonic - 1] = float(profile.offset)
        s_coeffs[harmonic - 1] = coefficients(f"s{harmonic}")

    kappa_profile = shape_profiles.get("k")
    kappa_lcfs = 1.0 if kappa_profile is None else float(kappa_profile.offset)
    return Geometry(
        Nr=int(grid.Nr),
        Nt=int(grid.Nt),
        radial_rule=str(grid.quadrature_scheme),
        radial_calculus=str(grid.calculus_scheme),
        K_max=None if grid.K_max is None else int(grid.K_max),
        R0=float(R0),
        Z0=float(Z0),
        a=float(a),
        kappa_lcfs=kappa_lcfs,
        c_lcfs=c_lcfs,
        s_lcfs=s_lcfs,
        h_coeffs=coefficients("h"),
        v_coeffs=coefficients("v"),
        kappa_coeffs=coefficients("k"),
        c_coeffs=c_coeffs,
        s_coeffs=s_coeffs,
    ).freeze()


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
