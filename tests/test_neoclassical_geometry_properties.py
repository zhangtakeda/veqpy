from __future__ import annotations

import numpy as np
import pytest

import veqpy as veq
from veqpy.kernels.abi.source_semantics import MU0
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


def _demo_equilibrium(*, b0: float = 3.0, ip: float = 3.0e6):
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
        B0=b0,
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
            Ip=ip,
        ),
    )
    assert result.success
    equilibrium = kernel.build_equilibrium()
    kernel.close()
    return equilibrium


def _uniform_demo_equilibrium():
    equilibrium = _demo_equilibrium()
    topology = equilibrium.grid
    grid = Grid(
        Nr=65,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=topology.L_max,
        M_max=topology.M_max,
        K_max=topology.K_max,
    )
    return equilibrium.resample(grid)


def _axis_even_rho2_limit(values: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return a copy with the removable even axis value reconstructed."""

    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape[0] >= 3 and abs(rho[0]) < 1.0e-10:
        x0, x1, x2 = rho[:3] ** 2
        result[0] = result[1] + (result[2] - result[1]) * (x0 - x1) / (x2 - x1)
    return result


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
    assert equilibrium.ftrap[0] == 0.0
    assert np.isclose(equilibrium.ftrap[-1], 0.80962984152523, rtol=2e-13)
    assert not equilibrium.ftrap.flags.writeable


@pytest.mark.parametrize(
    ("b0", "ip"),
    [(-3.0, -3.0e6), (-3.0, 3.0e6), (3.0, -3.0e6), (3.0, 3.0e6)],
)
def test_signed_f_and_q_follow_the_fixed_cocos_contract(b0: float, ip: float) -> None:
    equilibrium = _demo_equilibrium(b0=b0, ip=ip)

    assert equilibrium.F[-1] == pytest.approx(equilibrium.R0 * b0, rel=2.0e-15)
    assert np.all(np.sign(equilibrium.F) == np.sign(b0))
    assert np.all(np.sign(equilibrium.q) == np.sign(b0 * ip))
    assert equilibrium.Ip == pytest.approx(ip, rel=2.0e-13)

    geqdsk = equilibrium.to_geqdsk(
        R_range=(
            float(np.min(equilibrium.R)) - 0.1,
            float(np.max(equilibrium.R)) + 0.1,
        ),
        Z_range=(
            float(np.min(equilibrium.Z)) - 0.1,
            float(np.max(equilibrium.Z)) + 0.1,
        ),
        NR=17,
        NZ=19,
    )
    assert geqdsk.Bt0 == b0
    assert np.all(np.sign(geqdsk.F) == np.sign(b0))
    assert np.all(np.sign(geqdsk.q) == np.sign(b0 * ip))


@pytest.mark.parametrize("b0", [-3.0, 3.0])
def test_pj2_jpara_reconstructs_imas_jtotal_with_gm1(b0: float) -> None:
    equilibrium = _demo_equilibrium(b0=b0)
    two_pi = 2.0 * np.pi

    # VEQ's Ln_r is the poloidal mean of J/R, while V_r is 2*pi times
    # the integral of J*R. Their ratio therefore gives IMAS gm1=<R^-2>.
    with np.errstate(divide="ignore", invalid="ignore"):
        gm1 = two_pi**2 * equilibrium.Ln_r / equilibrium.V_r
    gm1 = _axis_even_rho2_limit(gm1, equilibrium.rho)
    surface_weights = equilibrium.J * equilibrium.R
    with np.errstate(divide="ignore", invalid="ignore"):
        gm1_direct = np.sum(surface_weights / equilibrium.R**2, axis=1) / np.sum(
            surface_weights,
            axis=1,
        )
    gm1_direct = _axis_even_rho2_limit(gm1_direct, equilibrium.rho)
    np.testing.assert_allclose(gm1, gm1_direct, rtol=2.0e-15, atol=2.0e-15)

    bphi2 = (equilibrium.F[:, None] / equilibrium.R) ** 2
    bp2 = (
        (equilibrium.alpha2 * equilibrium.psin_r[:, None]) ** 2
        * equilibrium.gttdivJR
        / (equilibrium.J * equilibrium.R)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gm5 = np.sum(surface_weights * (bphi2 + bp2), axis=1) / np.sum(
            surface_weights,
            axis=1,
        )
        gm9 = two_pi * equilibrium.S_r / equilibrium.V_r
    gm5 = _axis_even_rho2_limit(gm5, equilibrium.rho)
    gm9 = _axis_even_rho2_limit(gm9, equilibrium.rho)
    dpressure_dpsi = equilibrium.alpha1 * equilibrium.Pn_psin / (two_pi * MU0)

    jtotal_from_pj2 = equilibrium.jpara * equilibrium.F * gm1 / equilibrium.B0
    np.testing.assert_allclose(
        equilibrium.jtotal,
        jtotal_from_pj2,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert not equilibrium.jtotal.flags.writeable
    jtor_over_r = equilibrium.jtor * gm9
    jtor_over_r += two_pi * dpressure_dpsi * (1.0 - equilibrium.F**2 * gm1 / gm5)
    jtotal_from_jtor = gm5 * jtor_over_r / (equilibrium.F * gm1 * equilibrium.B0)

    relative_l2 = np.linalg.norm(jtotal_from_pj2 - jtotal_from_jtor) / np.linalg.norm(
        jtotal_from_jtor
    )
    interior_relative_l2 = np.linalg.norm(
        jtotal_from_pj2[:-1] - jtotal_from_jtor[:-1]
    ) / np.linalg.norm(jtotal_from_jtor[:-1])
    assert relative_l2 < 8.0e-3
    assert interior_relative_l2 < 2.0e-3
