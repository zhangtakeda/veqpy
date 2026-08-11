from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy import (
    Kernel,
    KernelBoundary,
    KernelRecipe,
    KernelSource,
    KernelTopology,
)
from veqpy.kernels.abi.enums import SOURCE_DRIVER_BY_ROUTE
from veqpy.kernels.abi.source_semantics import MU0, materialize_kernel_source
from veqpy.numerics import make_quadrature

SOURCE_ROUTE_CASES = tuple(
    (route, coordinate, nodes)
    for route in ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ")
    for coordinate in ("rho", "psin")
    for nodes in ("uniform", "grid")
)


def _topology(route: str, coordinate: str, nodes: str, *, sample_count: int = 9) -> KernelTopology:
    nr = 8
    if nodes == "grid":
        sample_count = nr
    profile_counts: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 0,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
    }
    if route in {"PJ2", "PJ3"}:
        profile_counts["F_count"] = 2
    elif coordinate == "psin" and nodes == "uniform":
        profile_counts["psin_count"] = 2
    return KernelTopology(
        **profile_counts,
        Nr=nr,
        Nt=8,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint="ip",
        sample_count=sample_count,
    )


def _source_rho_axis(topology: KernelTopology) -> np.ndarray:
    if topology.nodes == "grid":
        rho, _ = make_quadrature(topology.Nr, scheme=topology.quadrature)
        return np.asarray(rho, dtype=np.float64)
    axis = np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    if topology.route == "PP" and topology.coordinate == "psin" and topology.nodes == "uniform":
        return axis
    if topology.coordinate == "psin":
        return np.sqrt(axis)
    return axis


def _route_source_profiles(topology: KernelTopology) -> tuple[np.ndarray, np.ndarray]:
    rho = _source_rho_axis(topology)
    pprime = (
        rho * (1.0e6 + 0.4e6 * rho * rho)
        if topology.coordinate == "rho"
        else 1.0e6 + 0.4e6 * rho * rho
    )
    if topology.route == "PI":
        driver = rho * rho * (1.0e6 + 0.8e6 * rho * rho)
    elif topology.route in {"PJ1", "PJ2", "PJ3"}:
        driver = 1.0e6 + 0.8e6 * rho * rho
    elif topology.route == "PP":
        driver = rho * (1.0e6 + 0.8e6 * rho * rho)
    elif topology.route == "PF":
        driver = (
            rho * (1.0e6 + 0.8e6 * rho * rho)
            if topology.coordinate == "rho"
            else 1.0e6 + 0.8e6 * rho * rho
        )
    else:
        driver = 1.0e6 + 0.8e6 * rho * rho
    return pprime.astype(np.float64), driver.astype(np.float64)


def _irregular_route_source_profiles(
    topology: KernelTopology,
) -> tuple[np.ndarray, np.ndarray]:
    pprime, driver = _route_source_profiles(topology)
    if topology.coordinate == "rho":
        pprime[0] = 0.2 * np.max(np.abs(pprime))
    else:
        pprime[0] = 1.2 * pprime[1]

    if topology.route in {"PF", "PP"} and topology.coordinate == "rho":
        driver[0] = 0.2 * np.max(np.abs(driver))
    elif topology.route == "PI":
        driver[0] = 0.2 * np.max(np.abs(driver))
    else:
        driver[0] = 1.2 * driver[1]
    return pprime, driver


def _boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=(float(np.arcsin(0.2)),),
    )


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
def test_route_source_lowering_preserves_raw_user_physics(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = _topology(route, coordinate, nodes)
    pprime, driver = _irregular_route_source_profiles(topology)
    source = KernelSource(
        pprime=pprime,
        **{SOURCE_DRIVER_BY_ROUTE[route]: driver},
        Ip=3.0e6,
        beta=0.02,
    )

    materialized = materialize_kernel_source(topology, source)

    assert materialized.scaled_pprime.flags.writeable is False
    assert materialized.scaled_driver.flags.writeable is False
    assert_allclose(materialized.scaled_pprime, pprime * MU0)
    assert materialized.scaled_Ip == pytest.approx(3.0e6 * MU0)
    assert materialized.beta == pytest.approx(0.02)
    if route in {"PI", "PJ1", "PJ2", "PJ3"}:
        assert_allclose(materialized.scaled_driver, driver * MU0)
    else:
        assert_allclose(materialized.scaled_driver, driver)


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
def test_numba_route_residual_accepts_finite_irregular_axis_samples(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = _topology(route, coordinate, nodes)
    pprime, driver = _irregular_route_source_profiles(topology)
    source = KernelSource(
        pprime=pprime,
        **{SOURCE_DRIVER_BY_ROUTE[route]: driver},
        Ip=3.0e6,
    )
    kernel = Kernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba"),
    )
    try:
        residual = kernel.residual(
            np.zeros(topology.x_size, dtype=np.float64),
            _boundary(),
            source,
        )
    finally:
        kernel.close()

    assert np.all(np.isfinite(residual))


def test_source_lowering_accepts_large_physical_profiles_when_constraints_are_valid() -> None:
    topology = _topology("PF", "psin", "uniform")
    pprime = np.full(topology.sample_count, 1.0e12, dtype=np.float64)
    ffprime = np.full(topology.sample_count, -1.0e12, dtype=np.float64)
    source = KernelSource(pprime=pprime, ffprime=ffprime, Ip=3.0e6)

    materialized = materialize_kernel_source(topology, source)

    assert_allclose(materialized.scaled_pprime, pprime * MU0)
    assert_allclose(materialized.scaled_driver, ffprime)


def test_source_lowering_rejects_nonfinite_route_profiles() -> None:
    topology = _topology("PI", "rho", "uniform")
    pprime, itor = _route_source_profiles(topology)
    itor[-1] = np.nan

    with pytest.raises(ValueError, match="itor must contain only finite values"):
        materialize_kernel_source(
            topology,
            KernelSource(pprime=pprime, itor=itor, Ip=3.0e6),
        )
