from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from veqpy import (
    Kernel,
    KernelBoundary,
    KernelRecipe,
    KernelSource,
    KernelTopology,
)
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE, source_driver_for
from veqpy.kernels.abi.source_semantics import MU0, materialize_kernel_source
from veqpy.numerics import make_quadrature

P_ROUTE_CASES = (
    ("PF", "psin", "uniform"),
    ("PP", "psin", "uniform"),
    ("PI", "r", "uniform"),
    ("PJ1", "psin", "uniform"),
    ("PJ2", "psin", "uniform"),
    ("PJ3", "r", "grid"),
    ("PQ", "r", "grid"),
)


def _topology(
    *,
    route: str = "PQ",
    coordinate: str = "r",
    nodes: str = "uniform",
    constraint: str = "none",
    sample_count: int = 9,
    nr: int = 8,
) -> KernelTopology:
    if nodes == "grid":
        sample_count = nr
    return KernelTopology(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=(
            2
            if coordinate == "psin" and nodes == "uniform" and route not in {"PJ2", "PJ3"}
            else 0
        ),
        F_count=2 if route in {"PJ2", "PJ3"} else 0,
        c_counts=(),
        s_counts=(1,),
        Nr=nr,
        Nt=8,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint=constraint,
        sample_count=sample_count,
    )


def _driver_kwargs(
    route: str,
    parameter_nodes: np.ndarray,
    *,
    coordinate: str,
) -> dict[str, np.ndarray]:
    r = np.sqrt(parameter_nodes) if coordinate == "psin" else parameter_nodes
    if route == "PF":
        driver = -(1.0 + 0.2 * r * r) if coordinate == "psin" else 1.0 + 0.2 * r * r
    elif route == "PP":
        driver = r * (1.0 + 0.2 * r * r)
    elif route == "PI":
        driver = r * r * (1.0e6 + 0.2e6 * r * r)
    elif route in {"PJ1", "PJ2", "PJ3"}:
        driver = 1.0e6 + 0.2e6 * r * r
    else:
        driver = 1.7 + 0.1 * r * r
    return {source_driver_for(route, coordinate): np.asarray(driver, dtype=np.float64)}


def test_kernel_source_requires_exactly_one_pressure_representation() -> None:
    pressure = np.linspace(2.0, 1.0, 5, dtype=np.float64)
    derivative = np.gradient(pressure)
    q = np.linspace(1.7, 1.8, 5, dtype=np.float64)

    from_p = KernelSource(p=pressure, q=q)
    assert from_p.pressure_name == "p"
    assert from_p.pressure_profile is from_p.p
    assert from_p.P_r is None
    assert from_p.p0 is None
    assert from_p.p is not None
    assert not from_p.p.flags.writeable

    from_pprime = KernelSource(P_r=derivative, q=q)
    assert from_pprime.pressure_name == "P_r"
    assert from_pprime.pressure_profile is from_pprime.P_r
    assert from_pprime.p is None
    assert from_pprime.p0 == 0.0

    with pytest.raises(ValueError, match="requires exactly one pressure input"):
        KernelSource(q=q)
    with pytest.raises(ValueError, match="got p, P_r"):
        KernelSource(p=pressure, P_r=derivative, q=q)
    with pytest.raises(ValueError, match="p0 is derived from p"):
        KernelSource(p=pressure, p0=1.0, q=q)
    with pytest.raises(ValueError, match="p and q must share the same shape"):
        KernelSource(p=pressure[:-1], q=q)


@pytest.mark.parametrize(
    ("coordinate", "wrong_keyword", "required_keyword"),
    [
        ("r", "P_psin", "P_r"),
        ("rho", "P_r", "P_rho"),
        ("psin", "P_rho", "P_psin"),
    ],
)
def test_pressure_derivative_keyword_must_match_source_coordinate(
    coordinate: str,
    wrong_keyword: str,
    required_keyword: str,
) -> None:
    topology = _topology(route="PQ", coordinate=coordinate)
    values = np.linspace(-2.0e3, -1.0e3, topology.sample_count, dtype=np.float64)
    q = np.linspace(1.7, 1.8, topology.sample_count, dtype=np.float64)
    source = KernelSource(**{wrong_keyword: values, "q": q})

    with pytest.raises(
        ValueError,
        match=rf"coordinate='{coordinate}' requires pressure input {required_keyword}",
    ):
        materialize_kernel_source(topology, source)


@pytest.mark.parametrize(
    ("coordinate", "wrong_keyword", "required_keyword"),
    [
        ("r", "FF_psin", "FF_r"),
        ("rho", "FF_r", "FF_rho"),
        ("psin", "FF_rho", "FF_psin"),
    ],
)
def test_pf_derivative_keyword_must_match_source_coordinate(
    coordinate: str,
    wrong_keyword: str,
    required_keyword: str,
) -> None:
    topology = _topology(route="PF", coordinate=coordinate)
    values = np.linspace(-2.0e3, -1.0e3, topology.sample_count, dtype=np.float64)
    source = KernelSource(
        **{
            PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: values,
            wrong_keyword: np.linspace(-2.0, -1.0, topology.sample_count),
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"route PF with coordinate='{coordinate}' requires driver "
            rf"'{required_keyword}'"
        ),
    ):
        materialize_kernel_source(topology, source)


@pytest.mark.parametrize(
    ("route", "coordinate"),
    [
        ("PQ", "r"),
        ("PF", "psin"),
        ("PP", "psin"),
    ],
)
def test_uniform_p_is_lowered_to_coordinate_aware_pprime_and_edge_pressure(
    route: str,
    coordinate: str,
) -> None:
    topology = _topology(route=route, coordinate=coordinate)
    parameter = np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    physical_coordinate = parameter * parameter if route == "PP" else parameter
    edge_pressure = 6408.706536
    amplitude = 1800.0
    curvature = 250.0
    if coordinate == "r":
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate**2)
        expected_pprime = -2.0 * amplitude * physical_coordinate
    else:
        distance = 1.0 - physical_coordinate
        pressure = edge_pressure + amplitude * distance + curvature * distance * distance
        expected_pprime = -amplitude - 2.0 * curvature * distance
    driver_kwargs = _driver_kwargs(
        route,
        physical_coordinate,
        coordinate=coordinate,
    )

    from_p = materialize_kernel_source(
        topology,
        KernelSource(p=pressure, **driver_kwargs),
    )
    explicit = materialize_kernel_source(
        topology,
        KernelSource(
            **{PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: expected_pprime},
            p0=edge_pressure,
            **driver_kwargs,
        ),
    )

    assert from_p.scaled_p0 == pytest.approx(MU0 * edge_pressure)
    assert_allclose(from_p.scaled_pprime / MU0, expected_pprime, rtol=2.0e-13, atol=1.0e-7)
    assert_allclose(from_p.scaled_pprime, explicit.scaled_pprime, rtol=2.0e-13, atol=1.0e-13)
    assert from_p.scaled_p0 == pytest.approx(explicit.scaled_p0)


def test_pp_sqrt_psin_pressure_lowering_remains_stable_at_high_order() -> None:
    topology = _topology(
        route="PP",
        coordinate="psin",
        nodes="uniform",
        sample_count=51,
    )
    sqrt_psin = np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    psin = sqrt_psin * sqrt_psin
    edge_pressure = 6408.706536
    amplitude = 1800.0
    curvature = 250.0
    distance = 1.0 - psin
    pressure = edge_pressure + amplitude * distance + curvature * distance * distance
    expected_pprime = -amplitude - 2.0 * curvature * distance

    materialized = materialize_kernel_source(
        topology,
        KernelSource(
            p=pressure,
            **_driver_kwargs("PP", psin, coordinate="psin"),
        ),
    )

    assert materialized.scaled_p0 / MU0 == pytest.approx(edge_pressure)
    assert_allclose(
        materialized.scaled_pprime / MU0,
        expected_pprime,
        rtol=5.0e-11,
        atol=1.0e-6,
    )


def test_grid_p_uses_grid_differentiator_and_interpolates_the_lcfs_value() -> None:
    topology = _topology(nodes="grid")
    r, _ = make_quadrature(topology.Nr, scheme=topology.quadrature)
    edge_pressure = 6408.706536
    amplitude = 2200.0
    pressure = edge_pressure + amplitude * (1.0 - r * r)
    expected_pprime = -2.0 * amplitude * r
    source = KernelSource(
        p=pressure,
        q=1.71 + 0.16 * r * r,
    )

    materialized = materialize_kernel_source(topology, source)

    assert materialized.scaled_p0 / MU0 == pytest.approx(edge_pressure, abs=1.0e-8)
    assert_allclose(
        materialized.scaled_pprime / MU0,
        expected_pprime,
        rtol=2.0e-13,
        atol=1.0e-8,
    )


def test_p_lowering_does_not_impose_axis_parity() -> None:
    topology = _topology(nodes="uniform")
    r = np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    edge_pressure = 6408.706536
    amplitude = 2200.0
    pressure = edge_pressure + amplitude * (1.0 - r)

    materialized = materialize_kernel_source(
        topology,
        KernelSource(
            p=pressure,
            q=1.71 + 0.16 * r * r,
        ),
    )

    assert_allclose(
        materialized.scaled_pprime / MU0,
        -amplitude,
        rtol=2.0e-13,
        atol=1.0e-8,
    )
    assert materialized.scaled_p0 / MU0 == pytest.approx(edge_pressure)


@pytest.mark.parametrize("nodes", ["uniform", "grid"])
def test_constant_p_produces_an_exact_zero_derivative(nodes: str) -> None:
    topology = _topology(nodes=nodes)
    count = topology.sample_count
    pressure = np.full(count, 6408.706536, dtype=np.float64)
    parameter = (
        make_quadrature(count, scheme=topology.quadrature)[0]
        if nodes == "grid"
        else np.linspace(0.0, 1.0, count)
    )
    materialized = materialize_kernel_source(
        topology,
        KernelSource(
            p=pressure,
            q=1.71 + 0.16 * parameter * parameter,
        ),
    )

    assert_array_equal(materialized.scaled_pprime, np.zeros(count))
    assert materialized.scaled_p0 == pytest.approx(MU0 * pressure[0])


def test_constant_p_builds_an_equilibrium_from_its_absolute_pressure() -> None:
    topology = _topology(nodes="grid")
    r, _ = make_quadrature(topology.Nr, scheme=topology.quadrature)
    pressure = 6408.706536
    boundary = KernelBoundary(
        a=1.0,
        R0=10.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        s_offsets=(0.0,),
    )
    source = KernelSource(
        p=np.full(topology.Nr, pressure, dtype=np.float64),
        q=1.71 + 0.16 * r * r,
    )
    kernel = Kernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba"),
    )
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual = kernel.residual(x, boundary, source)
    equilibrium = kernel.build_equilibrium(x)

    assert np.all(np.isfinite(residual))
    assert_allclose(equilibrium.P_r, 0.0, rtol=0.0, atol=1.0e-12)
    assert_allclose(equilibrium.P, pressure, rtol=0.0, atol=1.0e-12)
    assert equilibrium.p0 == pytest.approx(pressure)
    assert equilibrium.alpha1 * equilibrium.alpha2 == pytest.approx(MU0 * pressure)


def test_p_rejects_zero_nonfinite_and_psin_grid_profiles() -> None:
    topology = _topology(nodes="grid")
    r, _ = make_quadrature(topology.Nr, scheme=topology.quadrature)
    q = 1.71 + 0.16 * r * r

    with pytest.raises(ValueError, match="p is all zero"):
        materialize_kernel_source(
            topology,
            KernelSource(p=np.zeros(topology.Nr), q=q),
        )

    nonfinite = np.ones(topology.Nr)
    nonfinite[-1] = np.nan
    with pytest.warns(RuntimeWarning, match="p must contain only finite values"):
        with pytest.raises(ValueError, match="p must contain only finite values"):
            materialize_kernel_source(
                topology,
                KernelSource(p=nonfinite, q=q),
            )

    psin_grid = _topology(coordinate="psin", nodes="grid")
    with pytest.raises(ValueError, match="psin\\(r\\) is solved at runtime"):
        materialize_kernel_source(
            psin_grid,
            KernelSource(p=np.ones(psin_grid.Nr), q=q),
        )


def test_beta_constraint_scales_p_mode_as_one_complete_profile() -> None:
    beta = 0.02
    nr = 16
    topology = _topology(
        nodes="grid",
        constraint="beta",
        nr=nr,
    )
    r, _ = make_quadrature(nr, scheme=topology.quadrature)
    boundary = KernelBoundary(
        a=1.0,
        R0=10.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        s_offsets=(0.0,),
    )
    edge_pressure = 6408.706536
    raw_pressure = edge_pressure + 900.0 * (1.0 - r * r)
    source = KernelSource(
        p=raw_pressure,
        q=1.71 + 0.16 * r * r,
        beta=beta,
    )
    kernel = Kernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba"),
    )
    x = np.zeros(kernel.x_size, dtype=np.float64)

    kernel.residual(x, boundary, source)
    equilibrium = kernel.build_equilibrium(x)
    common_scale = equilibrium.p0 / edge_pressure

    assert_allclose(
        equilibrium.P,
        raw_pressure * common_scale,
        rtol=2.0e-13,
        atol=1.0e-8,
    )
    assert equilibrium.beta_t == pytest.approx(beta)


@pytest.mark.parametrize(("route", "coordinate", "nodes"), P_ROUTE_CASES)
def test_p_and_explicit_pprime_p0_are_numba_route_equivalent(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = _topology(
        route=route,
        coordinate=coordinate,
        nodes=nodes,
    )
    parameter = (
        make_quadrature(topology.Nr, scheme=topology.quadrature)[0]
        if nodes == "grid"
        else np.linspace(0.0, 1.0, topology.sample_count)
    )
    physical_coordinate = parameter * parameter if route == "PP" else parameter
    edge_pressure = 6408.706536
    amplitude = 900.0
    if coordinate == "r":
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate**2)
        pprime = -2.0 * amplitude * physical_coordinate
    else:
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate)
        pprime = np.full(topology.sample_count, -amplitude)
    driver_kwargs = _driver_kwargs(
        route,
        physical_coordinate,
        coordinate=coordinate,
    )
    source_from_p = KernelSource(p=pressure, **driver_kwargs)
    source_from_pprime = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: pprime},
        p0=edge_pressure,
        **driver_kwargs,
    )
    boundary = KernelBoundary(
        a=0.5,
        R0=2.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        s_offsets=(0.0,),
    )
    kernel = Kernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba"),
    )
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual_from_p = kernel.residual(x, boundary, source_from_p)
    residual_from_pprime = kernel.residual(x, boundary, source_from_pprime)

    assert np.all(np.isfinite(residual_from_p))
    assert_allclose(
        residual_from_p,
        residual_from_pprime,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
