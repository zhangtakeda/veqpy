from __future__ import annotations

from pathlib import Path

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
from veqpy.kernels.abi.source_semantics import materialize_kernel_source
from veqpy.numerics import make_quadrature

pytestmark = pytest.mark.slow

SOURCE_ROUTE_CASES = (
    ("PF", "psin", "uniform"),
    ("PP", "psin", "uniform"),
    ("PI", "rho", "uniform"),
    ("PJ1", "psin", "uniform"),
    ("PJ2", "psin", "uniform"),
    ("PQ", "rho", "grid"),
)


def _parity_topology(route: str, coordinate: str, nodes: str) -> KernelTopology:
    nr = 8
    sample_count = nr if nodes == "grid" else 9
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=(
            2 if coordinate == "psin" and nodes == "uniform" and route != "PJ2" else 0
        ),
        F_count=2 if route == "PJ2" else 0,
        c_counts=(),
        s_counts=(2,),
        Nr=nr,
        Nt=8,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint="none",
        sample_count=sample_count,
    )


def _pressure_sources(
    topology: KernelTopology,
) -> tuple[KernelSource, KernelSource]:
    parameter = (
        make_quadrature(topology.Nr, scheme=topology.quadrature)[0]
        if topology.nodes == "grid"
        else np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    )
    physical_coordinate = parameter * parameter if topology.route == "PP" else parameter
    rho = (
        np.sqrt(physical_coordinate)
        if topology.coordinate == "psin"
        else physical_coordinate
    )
    edge_pressure = 6410.0
    amplitude = 900.0
    if topology.coordinate == "rho":
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate**2)
        pprime = -2.0 * amplitude * physical_coordinate
    else:
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate)
        pprime = np.full(topology.sample_count, -amplitude, dtype=np.float64)

    if topology.route == "PF":
        driver = 1.0 + 0.2 * rho * rho
    elif topology.route == "PP":
        driver = rho * (1.0 + 0.2 * rho * rho)
    elif topology.route == "PI":
        driver = rho * rho * (1.0e6 + 0.2e6 * rho * rho)
    elif topology.route in {"PJ1", "PJ2"}:
        driver = 1.0e6 + 0.2e6 * rho * rho
    else:
        driver = 1.71 + 0.16 * rho * rho
    driver_kwargs = {
        SOURCE_DRIVER_BY_ROUTE[topology.route]: np.asarray(driver, dtype=np.float64)
    }
    return (
        KernelSource(p=pressure, **driver_kwargs),
        KernelSource(
            pprime=pprime,
            p0=edge_pressure,
            **driver_kwargs,
        ),
    )


def _boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=2.0,
        Z0=0.1,
        B0=3.0,
        ka=1.3,
        s_offsets=(float(np.arcsin(0.2)),),
    )


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
def test_cxx_and_numba_share_canonical_pressure_source_contract(
    route: str,
    coordinate: str,
    nodes: str,
    tmp_path: Path,
) -> None:
    topology = _parity_topology(route, coordinate, nodes)
    source_from_p, source_from_pprime = _pressure_sources(topology)
    materialized_p = materialize_kernel_source(topology, source_from_p)
    materialized_pprime = materialize_kernel_source(topology, source_from_pprime)

    assert source_from_p.driver_name == SOURCE_DRIVER_BY_ROUTE[route]
    assert_allclose(
        materialized_p.scaled_pprime,
        materialized_pprime.scaled_pprime,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert_allclose(
        materialized_p.scaled_driver,
        materialized_pprime.scaled_driver,
        rtol=0.0,
        atol=0.0,
    )
    assert materialized_p.scaled_p0 == pytest.approx(materialized_pprime.scaled_p0)

    kernels = {
        "numba": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba"),
        ),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=tmp_path / "kernel-cache",
        ),
    }
    try:
        x = np.linspace(-2.0e-3, 2.0e-3, topology.x_size, dtype=np.float64)
        residuals = {
            (backend, pressure_mode): kernel.residual(x, _boundary(), source)
            for backend, kernel in kernels.items()
            for pressure_mode, source in (
                ("p", source_from_p),
                ("pprime", source_from_pprime),
            )
        }
    finally:
        for kernel in kernels.values():
            kernel.close()

    for residual in residuals.values():
        assert np.all(np.isfinite(residual))
    for backend in ("numba", "cxx"):
        assert_allclose(
            residuals[backend, "p"],
            residuals[backend, "pprime"],
            rtol=2.0e-11,
            atol=5.0e-13,
        )
    assert_allclose(
        residuals["cxx", "p"],
        residuals["numba", "p"],
        rtol=2.0e-9,
        atol=5.0e-12,
    )
