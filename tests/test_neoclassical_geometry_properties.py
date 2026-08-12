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
    assert np.isclose(equilibrium.ftrap[-1], 0.8098271965033752, rtol=2e-13)
    assert not equilibrium.ftrap.flags.writeable


def test_equilibrium_exposes_imas_toroidal_flux_coordinates() -> None:
    equilibrium = _uniform_demo_equilibrium()

    np.testing.assert_allclose(
        equilibrium.Phi_r,
        2.0 * np.pi * equilibrium.F * equilibrium.Ln_r,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        equilibrium.Phi,
        equilibrium.grid.accumulate(equilibrium.Phi_r),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    expected = np.sqrt(equilibrium.Phi / (np.pi * equilibrium.B0))
    np.testing.assert_allclose(equilibrium.rho_tor, expected, rtol=2.0e-15, atol=2.0e-15)
    rho_tor_edge = np.sqrt(
        equilibrium.grid.full_integral(equilibrium.Phi_r) / (np.pi * equilibrium.B0)
    )
    np.testing.assert_allclose(
        equilibrium.rho_tor_norm,
        expected / rho_tor_edge,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = equilibrium.F * equilibrium.Ln_r / (equilibrium.B0 * equilibrium.rho_tor)
    derivative = _axis_even_rho2_limit(derivative, equilibrium.rho)
    np.testing.assert_allclose(
        equilibrium.rho_tor_r,
        derivative,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        equilibrium.rho_tor_norm_r,
        derivative / rho_tor_edge,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert equilibrium.rho_tor_norm[0] == 0.0
    assert equilibrium.rho_tor_norm[-1] == pytest.approx(1.0, abs=3.0e-16)
    assert not equilibrium.rho_tor.flags.writeable


def test_equilibrium_gm_series_matches_imas_flux_surface_averages() -> None:
    equilibrium = _uniform_demo_equilibrium()
    weights = equilibrium.J * equilibrium.R

    def surface_average(values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
        return _axis_even_rho2_limit(result, equilibrium.rho)

    with np.errstate(divide="ignore", invalid="ignore"):
        bp2 = (
            (equilibrium.alpha2 * equilibrium.psin_r[:, None]) ** 2
            * equilibrium.gttdivJR
            / (equilibrium.J * equilibrium.R)
        )
        grad_rho_tor2 = (
            equilibrium.rho_tor_r[:, None] ** 2
            * equilibrium.gttdivJR
            * equilibrium.R
            / equilibrium.J
        )
    bp2[0] = np.nan
    grad_rho_tor2[0] = np.nan
    b2 = (equilibrium.F[:, None] / equilibrium.R) ** 2 + bp2

    expected = (
        surface_average(1.0 / equilibrium.R**2),
        surface_average(grad_rho_tor2 / equilibrium.R**2),
        surface_average(grad_rho_tor2),
        surface_average(1.0 / b2),
        surface_average(b2),
        surface_average(grad_rho_tor2 / b2),
        surface_average(np.sqrt(grad_rho_tor2)),
        surface_average(equilibrium.R),
        surface_average(1.0 / equilibrium.R),
    )
    for index, reference in enumerate(expected, start=1):
        values = getattr(equilibrium, f"gm{index}")
        np.testing.assert_allclose(values, reference, rtol=3.0e-14, atol=3.0e-14)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        assert not values.flags.writeable
        assert getattr(equilibrium, f"gm{index}") is values


@pytest.mark.parametrize(
    ("b0", "ip"),
    [(-3.0, -3.0e6), (-3.0, 3.0e6), (3.0, -3.0e6), (3.0, 3.0e6)],
)
def test_signed_f_and_q_follow_the_fixed_cocos_contract(b0: float, ip: float) -> None:
    equilibrium = _demo_equilibrium(b0=b0, ip=ip)

    edge_f2 = np.dot(equilibrium.grid.edge_interpolation_weights, equilibrium.F2)
    assert equilibrium.grid.rho[-1] < 1.0
    # Endpoint extrapolation from an open Gauss grid loses a few digits even
    # though the edge-conditioned integral identity is exact on the grid.
    assert edge_f2 == pytest.approx((equilibrium.R0 * b0) ** 2, rel=2.0e-8)
    assert np.all(np.sign(equilibrium.F) == np.sign(b0))
    assert np.all(np.sign(equilibrium.q) == np.sign(b0 * ip))
    assert equilibrium.Ip == pytest.approx(ip, rel=2.0e-13)
    np.testing.assert_allclose(
        equilibrium.rho_tor**2,
        equilibrium.Phi / (np.pi * b0),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    rho_tor_edge = np.sqrt(equilibrium.grid.full_integral(equilibrium.Phi_r) / (np.pi * b0))
    np.testing.assert_allclose(
        equilibrium.rho_tor_norm,
        equilibrium.rho_tor / rho_tor_edge,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert 0.0 < equilibrium.rho_tor_norm[0] < equilibrium.rho_tor_norm[-1] < 1.0
    for index in range(1, 10):
        assert np.all(getattr(equilibrium, f"gm{index}") > 0.0)

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
    endpoint_grid = Grid(
        Nr=equilibrium.grid.Nr + 2,
        Nt=max(equilibrium.grid.Nt, 64),
        quadrature_scheme="uniform",
        L_max=equilibrium.grid.L_max,
        M_max=equilibrium.grid.M_max,
        K_max=equilibrium.grid.K_max,
    )
    endpoint_equilibrium = equilibrium.resample(endpoint_grid)
    endpoint_boundary = np.column_stack(
        (endpoint_equilibrium.R[-1], endpoint_equilibrium.Z[-1])
    )
    np.testing.assert_allclose(
        geqdsk.boundary[:-1],
        endpoint_boundary,
    )
    np.testing.assert_array_equal(geqdsk.boundary[-1], geqdsk.boundary[0])
    assert geqdsk.boundary.shape[0] >= 65


def test_geqdsk_roundtrip_preserves_an_asymmetric_vertical_grid(tmp_path) -> None:
    equilibrium = _demo_equilibrium()
    equilibrium.Z0 = 0.25
    z_range = (-1.7, 1.1)

    geqdsk = equilibrium.to_geqdsk(
        R_range=(0.2, 2.0),
        Z_range=z_range,
        NR=17,
        NZ=19,
    )

    # GEQDSK Z0 is zmid for the rectangular psi grid, not the equilibrium's
    # fitted plasma-boundary reference height.
    expected_zmid = 0.5 * sum(z_range)
    assert geqdsk.Z0 == pytest.approx(expected_zmid)
    assert geqdsk.Zaxis == pytest.approx(float(equilibrium.Z[0, 0]))

    output = tmp_path / "asymmetric-z-grid.geqdsk"
    geqdsk.write(output)
    restored = veq.Geqdsk(output)

    assert restored.Z0 == pytest.approx(expected_zmid)
    assert restored.Zmin == pytest.approx(z_range[0])
    assert restored.Zmax == pytest.approx(z_range[1])
    np.testing.assert_allclose(restored.psi, geqdsk.psi, rtol=5.0e-10, atol=5.0e-11)
    np.testing.assert_allclose(restored.boundary, geqdsk.boundary)
    np.testing.assert_array_equal(restored.boundary[-1], restored.boundary[0])
    assert restored.boundary.shape[0] >= 65


@pytest.mark.parametrize("b0", [-3.0, 3.0])
def test_pj2_jpara_reconstructs_imas_jtotal_with_gm1(b0: float) -> None:
    equilibrium = _demo_equilibrium(b0=b0)
    two_pi = 2.0 * np.pi

    # VEQ's Ln_r is the poloidal mean of J/R, while V_r is 2*pi times
    # the integral of J*R. Their ratio therefore gives IMAS gm1=<R^-2>.
    gm1 = equilibrium.gm1
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
        gm5_direct = np.sum(surface_weights * (bphi2 + bp2), axis=1) / np.sum(
            surface_weights,
            axis=1,
        )
        gm9_direct = two_pi * equilibrium.S_r / equilibrium.V_r
    gm5_direct = _axis_even_rho2_limit(gm5_direct, equilibrium.rho)
    gm9_direct = _axis_even_rho2_limit(gm9_direct, equilibrium.rho)
    np.testing.assert_allclose(equilibrium.gm5, gm5_direct, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(equilibrium.gm9, gm9_direct, rtol=2.0e-15, atol=2.0e-15)
    gm5 = equilibrium.gm5
    gm9 = equilibrium.gm9
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


def test_jtor_axis_is_the_conservative_itor_over_area_limit() -> None:
    equilibrium = _uniform_demo_equilibrium()
    rho2_1 = equilibrium.rho[1] ** 2
    rho2_2 = equilibrium.rho[2] ** 2
    ratio_1 = equilibrium.Itor[1] / equilibrium.S[1]
    ratio_2 = equilibrium.Itor[2] / equilibrium.S[2]
    expected = (rho2_2 * ratio_1 - rho2_1 * ratio_2) / (rho2_2 - rho2_1)

    assert equilibrium.rho[0] == 0.0
    assert np.isfinite(equilibrium.jtor[0])
    assert equilibrium.jtor[0] == pytest.approx(expected, rel=2.0e-15)
