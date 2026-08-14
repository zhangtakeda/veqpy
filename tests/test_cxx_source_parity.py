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
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE, source_driver_for
from veqpy.kernels.abi.source_semantics import materialize_kernel_source
from veqpy.numerics import make_quadrature

pytestmark = pytest.mark.slow

SOURCE_ROUTE_CASES = (
    ("PF", "psin", "uniform"),
    ("PP", "psin", "uniform"),
    ("PI", "r", "uniform"),
    ("PJ1", "psin", "uniform"),
    ("PJ2", "psin", "uniform"),
    ("PJ3", "r", "grid"),
    ("PQ", "r", "grid"),
)

SOURCE_CONSTRAINT_CASES = (
    ("PF", "psin", "uniform", "ip"),
    ("PF", "psin", "uniform", "beta"),
    ("PP", "psin", "uniform", "ip"),
    ("PP", "psin", "uniform", "beta"),
    ("PP", "psin", "uniform", "both"),
    ("PI", "r", "uniform", "ip"),
    ("PI", "r", "uniform", "beta"),
    ("PI", "r", "uniform", "both"),
    ("PJ1", "psin", "uniform", "ip"),
    ("PJ1", "psin", "uniform", "beta"),
    ("PJ1", "psin", "uniform", "both"),
    ("PJ2", "psin", "uniform", "ip"),
    ("PJ2", "psin", "uniform", "beta"),
    ("PJ2", "psin", "uniform", "both"),
    ("PJ3", "r", "grid", "ip"),
    ("PJ3", "r", "grid", "beta"),
    ("PJ3", "r", "grid", "both"),
    ("PQ", "r", "grid", "ip"),
    ("PQ", "r", "grid", "beta"),
    ("PQ", "r", "grid", "both"),
)

SOURCE_ALTERNATE_COORDINATE_CASES = (
    ("PF", "r", "grid", "beta"),
    ("PP", "r", "grid", "beta"),
    ("PI", "psin", "grid", "beta"),
    ("PJ1", "r", "grid", "beta"),
    ("PJ2", "r", "grid", "beta"),
    ("PJ3", "psin", "uniform", "beta"),
    ("PQ", "psin", "uniform", "beta"),
)

_SOURCE_STATE_KEYS = (
    "alpha1",
    "alpha2",
    "scaled_effective_p0",
    "pressure_multiplier",
)


def _parity_topology(
    route: str,
    coordinate: str,
    nodes: str,
    *,
    constraint: str = "none",
) -> KernelTopology:
    nr = 8
    sample_count = nr if nodes == "grid" else 9
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=(
            2
            if coordinate == "psin" and nodes == "uniform" and route not in {"PJ2", "PJ3"}
            else 0
        ),
        F_count=2 if route in {"PJ2", "PJ3"} else 0,
        c_counts=(),
        s_counts=(2,),
        Nr=nr,
        Nt=8,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint=constraint,
        sample_count=sample_count,
    )


def _pressure_sources(
    topology: KernelTopology,
    *,
    amplitude: float = 900.0,
) -> tuple[KernelSource, KernelSource]:
    parameter = (
        make_quadrature(topology.Nr, scheme=topology.quadrature)[0]
        if topology.nodes == "grid"
        else np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    )
    if topology.nodes == "grid":
        r = parameter
        physical_coordinate = r * r if topology.coordinate == "psin" else r
    elif topology.source_parameterization == "sqrt_psin":
        r = parameter
        physical_coordinate = r * r
    elif topology.coordinate == "psin":
        physical_coordinate = parameter
        r = np.sqrt(physical_coordinate)
    else:
        physical_coordinate = parameter
        r = physical_coordinate
    edge_pressure = 6410.0
    if topology.coordinate == "r":
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate**2)
        pprime = -2.0 * amplitude * physical_coordinate
    else:
        pressure = edge_pressure + amplitude * (1.0 - physical_coordinate)
        pprime = np.full(topology.sample_count, -amplitude, dtype=np.float64)

    if topology.route == "PF":
        driver = (
            r * (1.0 + 0.2 * r * r)
            if topology.coordinate == "r"
            else -(1.0 + 0.2 * r * r)
        )
    elif topology.route == "PP":
        driver = r * (1.0 + 0.2 * r * r)
    elif topology.route == "PI":
        driver = r * r * (1.0e6 + 0.2e6 * r * r)
    elif topology.route in {"PJ1", "PJ2", "PJ3"}:
        driver = 1.0e6 + 0.2e6 * r * r
    else:
        driver = 1.71 + 0.16 * r * r
    driver_kwargs = {
        source_driver_for(topology.route, topology.coordinate): np.asarray(
            driver, dtype=np.float64
        )
    }
    if topology.source_uses_ip_constraint:
        driver_kwargs["Ip"] = 1.0e6
    if topology.source_uses_beta_constraint:
        driver_kwargs["beta"] = 0.02
    return (
        KernelSource(p=pressure, **driver_kwargs),
        KernelSource(
            **{PRESSURE_DERIVATIVE_BY_COORDINATE[topology.coordinate]: pprime},
            p0=edge_pressure,
            **driver_kwargs,
        ),
    )


def _runtime_source_state(kernel: Kernel) -> np.ndarray:
    if kernel.recipe.backend == "numba":
        workspace = kernel._impl._solver.runtime.source_workspace
        return np.array(
            [
                workspace.alpha_state[0],
                workspace.alpha_state[1],
                workspace.pressure_state[0],
                workspace.pressure_state[1],
            ],
            dtype=np.float64,
        )
    state = dict(kernel._impl._cxx_solver().source_state())
    return np.array(
        [state[name] for name in _SOURCE_STATE_KEYS],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def cxx_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cxx-source-parity-cache")


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
    cxx_cache_root: Path,
) -> None:
    topology = _parity_topology(route, coordinate, nodes)
    source_from_p, source_from_pprime = _pressure_sources(topology)
    materialized_p = materialize_kernel_source(topology, source_from_p)
    materialized_pprime = materialize_kernel_source(topology, source_from_pprime)

    assert source_from_p.driver_name == source_driver_for(route, topology.coordinate)
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
            cache_root=cxx_cache_root,
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


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
def test_cxx_and_numba_preserve_finite_irregular_axis_samples(
    route: str,
    coordinate: str,
    nodes: str,
    cxx_cache_root: Path,
) -> None:
    topology = _parity_topology(route, coordinate, nodes)
    _, regular_source = _pressure_sources(topology)
    pprime = np.array(regular_source.pressure_profile, copy=True)
    driver = np.array(regular_source.driver_profile, copy=True)
    if coordinate == "r":
        pprime[0] = 0.2 * np.max(np.abs(pprime))
    else:
        pprime[0] = 1.2 * pprime[1]
    driver[0] = 1.2 * driver[1]
    source = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[topology.coordinate]: pprime},
        p0=regular_source.p0,
        **{regular_source.driver_name: driver},
    )
    kernels = {
        "numba": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba"),
        ),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=cxx_cache_root,
        ),
    }
    try:
        x = np.linspace(-2.0e-3, 2.0e-3, topology.x_size, dtype=np.float64)
        residuals = {
            backend: kernel.residual(x, _boundary(), source)
            for backend, kernel in kernels.items()
        }
    finally:
        for kernel in kernels.values():
            kernel.close()

    assert np.all(np.isfinite(residuals["numba"]))
    assert np.all(np.isfinite(residuals["cxx"]))
    assert_allclose(
        residuals["cxx"],
        residuals["numba"],
        rtol=2.0e-9,
        atol=5.0e-10,
    )


@pytest.mark.parametrize(
    ("route", "coordinate", "nodes", "constraint"),
    SOURCE_CONSTRAINT_CASES + SOURCE_ALTERNATE_COORDINATE_CASES,
)
def test_cxx_and_numba_share_constrained_pressure_state(
    route: str,
    coordinate: str,
    nodes: str,
    constraint: str,
    cxx_cache_root: Path,
) -> None:
    topology = _parity_topology(
        route,
        coordinate,
        nodes,
        constraint=constraint,
    )
    _, source = _pressure_sources(topology)
    materialized = materialize_kernel_source(topology, source)
    kernels = {
        "numba": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba"),
        ),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=cxx_cache_root,
        ),
    }
    try:
        x = np.linspace(-2.0e-3, 2.0e-3, topology.x_size, dtype=np.float64)
        residuals = {
            backend: kernel.residual(x, _boundary(), source)
            for backend, kernel in kernels.items()
        }
        numba_state = _runtime_source_state(kernels["numba"])
        cxx_state = _runtime_source_state(kernels["cxx"])
    finally:
        for kernel in kernels.values():
            kernel.close()

    assert np.all(np.isfinite(residuals["numba"]))
    assert np.all(np.isfinite(residuals["cxx"]))
    assert np.all(np.isfinite(numba_state))
    assert np.all(np.isfinite(cxx_state))
    assert_allclose(
        residuals["cxx"],
        residuals["numba"],
        rtol=2.0e-9,
        atol=5.0e-10,
    )
    assert_allclose(
        cxx_state,
        numba_state,
        rtol=2.0e-9,
        atol=5.0e-12,
    )
    assert cxx_state[2] == pytest.approx(
        materialized.scaled_p0 * cxx_state[3],
        rel=2.0e-13,
        abs=2.0e-13,
    )
    if topology.source_uses_beta_constraint:
        assert cxx_state[3] != pytest.approx(1.0)


def test_pf_psin_reversed_current_closes_flux_scale_magnitude_in_both_backends(
    cxx_cache_root: Path,
) -> None:
    topology = _parity_topology("PF", "psin", "uniform", constraint="ip")
    _, reference = _pressure_sources(topology)
    source = KernelSource(
        P_psin=reference.pressure_profile,
        p0=reference.p0,
        FF_psin=reference.driver_profile,
        Ip=-1.0e6,
    )
    kernels = {
        "numba": Kernel(topology=topology, recipe=KernelRecipe(backend="numba")),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=cxx_cache_root,
        ),
    }
    try:
        x = np.linspace(-2.0e-3, 2.0e-3, topology.x_size, dtype=np.float64)
        residuals = {
            backend: kernel.residual(x, _boundary(), source)
            for backend, kernel in kernels.items()
        }
        states = {backend: _runtime_source_state(kernel) for backend, kernel in kernels.items()}
    finally:
        for kernel in kernels.values():
            kernel.close()

    assert states["numba"][1] < 0.0
    assert states["cxx"][1] < 0.0
    assert_allclose(residuals["cxx"], residuals["numba"], rtol=2.0e-9, atol=5.0e-10)
    assert_allclose(states["cxx"], states["numba"], rtol=2.0e-9, atol=5.0e-12)


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
@pytest.mark.parametrize("constraint", ["none", "beta"])
def test_cxx_and_numba_share_constant_pressure_alpha_fallback(
    route: str,
    coordinate: str,
    nodes: str,
    constraint: str,
    cxx_cache_root: Path,
) -> None:
    topology = _parity_topology(
        route,
        coordinate,
        nodes,
        constraint=constraint,
    )
    source_from_p, source_from_pprime = _pressure_sources(topology, amplitude=0.0)
    kernels = {
        "numba": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba"),
        ),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=cxx_cache_root,
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
        numba_state = _runtime_source_state(kernels["numba"])
        cxx_state = _runtime_source_state(kernels["cxx"])
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
        residuals["cxx", "pprime"],
        residuals["numba", "pprime"],
        rtol=2.0e-9,
        atol=5.0e-10,
    )
    assert_allclose(
        cxx_state,
        numba_state,
        rtol=2.0e-9,
        atol=5.0e-12,
    )
    assert abs(cxx_state[0] * cxx_state[1]) == pytest.approx(
        abs(cxx_state[2]),
        rel=2.0e-12,
        abs=2.0e-13,
    )
    if constraint == "none":
        assert cxx_state[3] == pytest.approx(1.0)
    else:
        assert cxx_state[3] != pytest.approx(1.0)
