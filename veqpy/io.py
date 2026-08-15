"""Boundary formats for FusionPRIME state snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
from fusionprime_base import MU0
from fusionprime_base import Equilibrium as BaseEquilibrium

from .model.equilibrium import Equilibrium as _RasterEquilibrium
from .model.geqdsk import Geqdsk
from .model.grid import Grid as _RasterGrid
from .model.profile import Profile as _RasterProfile


def export_geqdsk(
    equilibrium: BaseEquilibrium,
    *,
    R_range: tuple[float, float],
    Z_range: tuple[float, float],
    NR: int,
    NZ: int | None = None,
    header: str = "",
    limiter: np.ndarray | None = None,
    psi_axis: float = 0.0,
    psi_outside: float | None = None,
) -> Geqdsk:
    """Export a frozen FusionPRIME ``Equilibrium`` as a GEQDSK payload.

    The exporter is deliberately an I/O adapter.  It lowers the frozen base
    state to the rectangular GEQDSK rasterizer and never exposes the
    VEQPy numerical model classes as part of the public state contract.
    """

    if not isinstance(equilibrium, BaseEquilibrium):
        raise TypeError("equilibrium must be fusionprime_base.Equilibrium")
    raster_state = _lower_to_geqdsk_snapshot(equilibrium)
    return raster_state.to_geqdsk(
        R_range=R_range,
        Z_range=Z_range,
        NR=NR,
        NZ=NZ,
        header=header,
        limiter=limiter,
        psi_axis=psi_axis,
        psi_outside=psi_outside,
    )


def _lower_to_geqdsk_snapshot(equilibrium: BaseEquilibrium) -> _RasterEquilibrium:
    geometry = equilibrium.geometry
    grid = _RasterGrid(
        Nr=int(geometry.Nr),
        Nt=int(geometry.Nt),
        L_max=int(geometry.L_max),
        M_max=int(geometry.M_max),
        K_max=None if geometry.K_max is None else int(geometry.K_max),
        quadrature_scheme=str(geometry.radial_rule),
        calculus_scheme=str(geometry.radial_calculus),
    )
    shape_profiles: dict[str, _RasterProfile] = {
        "h": _basic_profile(geometry.h_coeffs),
        "v": _basic_profile(geometry.v_coeffs),
        "k": _basic_profile(geometry.kappa_coeffs, offset=float(geometry.kappa_lcfs)),
    }
    powers = np.asarray(geometry.K_values, dtype=np.int64)
    for harmonic in range(int(geometry.M_max) + 1):
        shape_profiles[f"c{harmonic}"] = _shape_profile(
            geometry.c_coeffs[harmonic],
            offset=float(geometry.c_lcfs[harmonic]),
            power=int(powers[harmonic]),
        )
    for harmonic in range(1, int(geometry.M_max) + 1):
        shape_profiles[f"s{harmonic}"] = _shape_profile(
            geometry.s_coeffs[harmonic - 1],
            offset=float(geometry.s_lcfs[harmonic - 1]),
            power=int(powers[harmonic]),
        )

    psi_span = float(geometry.integrate_radial(np.asarray(equilibrium.psi_r, dtype=np.float64)))
    if not np.isfinite(psi_span) or abs(psi_span) <= 1.0e-14:
        raise ValueError("equilibrium has no finite nonzero axis-to-LCFS flux span")
    return _RasterEquilibrium(
        R0=float(geometry.R0),
        Z0=float(geometry.Z0),
        B0=float(equilibrium.B0),
        a=float(geometry.a),
        grid=grid,
        shape_profiles=shape_profiles,
        FFn_psin=np.asarray(equilibrium.FF_psi, dtype=np.float64),
        Pn_psin=np.asarray(equilibrium.P_psi, dtype=np.float64) * MU0,
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        psin_r=np.asarray(equilibrium.psin_r, dtype=np.float64),
        psin_rr=np.asarray(equilibrium.psin_rr, dtype=np.float64),
        p0=float(equilibrium.P0),
        alpha1=1.0,
        alpha2=psi_span,
    )


def _basic_profile(coefficients: Any, *, offset: float = 0.0) -> _RasterProfile:
    return _RasterProfile(
        offset=offset,
        power=0,
        envelope_power=1,
        coeff=np.asarray(coefficients, dtype=np.float64),
    )


def _shape_profile(coefficients: Any, *, offset: float, power: int) -> _RasterProfile:
    return _RasterProfile(
        offset=offset,
        power=power,
        envelope_power=1,
        coeff=np.asarray(coefficients, dtype=np.float64),
    )


__all__ = ["export_geqdsk"]
