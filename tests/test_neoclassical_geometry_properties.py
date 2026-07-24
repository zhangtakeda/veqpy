from __future__ import annotations

import numpy as np

import veqpy as veq
from veqpy.model import Grid


def _profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    pressure = (
        beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / (1.0 + exp_ap * (alpha_p - 1.0))
    )
    current = (
        (1.0 - beta0)
        * alpha_f
        * (np.exp(alpha_f * psin) - exp_af)
        / (1.0 + exp_af * (alpha_f - 1.0))
    )
    return pressure, current


def _uniform_demo_equilibrium():
    topology = veq.KernelTopology(
        h_count=3,
        v_count=0,
        kappa_count=6,
        psin_count=6,
        F_count=0,
        c_counts=(),
        s_counts=(3,),
        Nr=16,
        Nt=16,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=51,
    )
    kernel = veq.build(
        topology=topology,
        recipe=veq.KernelRecipe(backend="numba"),
        config=veq.KernelConfig(initial="cold"),
    )
    boundary = veq.KernelBoundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=(float(np.arcsin(0.5)),),
    )
    source_axis = np.linspace(0.0, 1.0, topology.sample_count)
    pprime, ffprime = _profiles(source_axis)
    result = kernel.solve(
        boundary=boundary,
        source=veq.KernelSource(
            pprime=pprime,
            ffprime=ffprime,
            Ip=3.0e6,
        ),
    )
    assert result.success
    grid = Grid(
        Nr=65,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=topology.L_max,
        M_max=topology.M_max,
        K_max=topology.K_max,
    )
    return kernel.build_equilibrium().resample(grid)


def test_uniform_equilibrium_exposes_neoclassical_geometry_profiles() -> None:
    equilibrium = _uniform_demo_equilibrium()

    np.testing.assert_allclose(equilibrium.Rc, equilibrium.R0 + equilibrium.a * equilibrium.h)
    np.testing.assert_allclose(
        equilibrium.epsilon,
        equilibrium.a * equilibrium.rho / equilibrium.Rc,
    )
    assert equilibrium.h.shape == equilibrium.rho.shape
    assert equilibrium.v.shape == equilibrium.rho.shape
    assert equilibrium.kappa.shape == equilibrium.rho.shape
    assert np.all(np.isfinite(equilibrium.ftrap))
    assert np.all((equilibrium.ftrap >= 0.0) & (equilibrium.ftrap <= 1.0))
    assert np.isclose(equilibrium.ftrap[0], 0.07780086351634649, rtol=2e-13)
    assert np.isclose(equilibrium.ftrap[-1], 0.80962984152523, rtol=2e-13)
    assert not equilibrium.ftrap.flags.writeable
