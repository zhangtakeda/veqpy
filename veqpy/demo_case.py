"""Small dependency-free Plasma fixture used by the VEQPy CLI smoke path."""

from __future__ import annotations

import numpy as np
from fusionprime_base import (
    MU0,
    Current,
    Equilibrium,
    Flux,
    Geometry,
    Kinetic,
    Plasma,
    Source,
)


def make_demo_plasma() -> Plasma:
    """Return a valid frozen five-State Plasma for a minimal VEQ solve."""

    geometry = Geometry(
        Nr=8,
        Nt=12,
        radial_rule="uniform",
        radial_calculus="spectral",
        K_max=None,
        R0=3.0,
        Z0=0.0,
        a=0.9,
        kappa_lcfs=1.5,
        c_lcfs=np.zeros(3),
        s_lcfs=np.zeros(2),
        h_coeffs=np.zeros(3),
        v_coeffs=np.zeros(3),
        kappa_coeffs=np.zeros(3),
        c_coeffs=np.zeros((3, 3)),
        s_coeffs=np.zeros((2, 3)),
    )
    equilibrium = Equilibrium(
        geometry=geometry,
        FF_psi=np.full(geometry.Nr, -0.06),
        P_psi=np.full(geometry.Nr, -0.36 / MU0),
        psi_r=1.4 * geometry.r,
        B0=5.0,
        P0=1_000.0,
    )
    rho = np.linspace(0.0, 1.0, geometry.Nr)
    kinetic = Kinetic(
        rho=rho,
        ion_names=("D",),
        Aion=np.array([2.014]),
        Znuc=np.array([1]),
        Zion=np.ones((1, geometry.Nr)),
        Z2ion=np.ones((1, geometry.Nr)),
        ni=np.full((1, geometry.Nr), 2.0e19),
        Ti=np.full((1, geometry.Nr), 5.0e3),
        Te=np.full(geometry.Nr, 5.0e3),
        omega=np.zeros((1, geometry.Nr)),
    )
    current = Current(
        rho=rho,
        q=1.0 + rho,
        q_rho=np.ones_like(rho),
        Itor=np.linspace(0.0, 1.0e6, geometry.Nr),
        jtor=np.full(geometry.Nr, 1.0e5),
        jtotal=np.full(geometry.Nr, 1.0e5),
        ellpara=np.full(geometry.Nr, 20.0),
        etapara=np.full(geometry.Nr, 1.0e-8),
        jbootstrap=np.zeros_like(rho),
        jdriven=np.zeros_like(rho),
    )
    flux = Flux(rho=rho, ion_names=("D",))
    source = Source(rho=rho, ion_names=("D",))
    return Plasma(
        equilibrium=equilibrium,
        kinetic=kinetic,
        current=current,
        flux=flux,
        source=source,
    ).freeze()


__all__ = ["make_demo_plasma"]
