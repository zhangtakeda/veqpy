from __future__ import annotations

import numpy as np
import pytest

from veqpy import Kernel, KernelConfig, KernelInput, KernelTopology
from veqpy.adapter import VEQAdapter
from veqpy.demo_case import make_demo_plasma


def _topology(
    *,
    route: str = "PF",
    nodes: str = "uniform",
    coordinate: str = "psin",
) -> KernelTopology:
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3 if coordinate == "psin" and route not in {"PJ2", "PJ3"} else 0,
        F_count=2 if route in {"PJ2", "PJ3"} and coordinate == "psin" else 0,
        c_counts=(1, 1, 1),
        s_counts=(1, 1),
        Nr=8,
        Nt=12,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint="ip",
        sample_count=None if nodes == "explicit" else 8,
        source_capacity=16 if nodes == "explicit" else None,
    )


def _prepared_input(topology: KernelTopology) -> KernelInput:
    buffer = KernelInput.allocate(topology)
    if topology.nodes == "explicit":
        nodes = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        buffer.source_count = nodes.size
        buffer.source_nodes[: nodes.size] = nodes
        buffer.pressure[: nodes.size] = -1.0e5
        buffer.driver[: nodes.size] = -1.0
        buffer.Ip = 3.0e6
        buffer.clear_unused_source_tail()
    else:
        VEQAdapter(topology, buffer).fill(make_demo_plasma())
    return buffer


@pytest.mark.slow
def test_cxx_and_numba_match_on_the_declared_shared_kernel_contract() -> None:
    topology = _topology()
    config = KernelConfig(max_evaluations=800)
    numba_input = _prepared_input(topology)
    cxx_input = _prepared_input(topology)
    numba = Kernel(topology=topology, input=numba_input, config=config, backend="numba")
    try:
        cxx = Kernel(topology=topology, input=cxx_input, config=config, backend="cxx")
    except Exception as error:
        numba.close()
        detail = f"{type(error).__name__}: {error}".lower()
        if any(token in detail for token in ("cmake", "compiler", "nanobind", "build", "native")):
            pytest.skip(f"Cxx backend unavailable: {error}")
        raise
    try:
        numba_output = numba.solve()
        cxx_output = cxx.solve()
        assert numba_output.success and cxx_output.success
        np.testing.assert_allclose(cxx_output.x, numba_output.x, rtol=0.0, atol=5.0e-6)
        np.testing.assert_allclose(cxx_output.raw, numba_output.raw, rtol=0.0, atol=5.0e-6)
    finally:
        numba.close()
        cxx.close()


def test_cxx_rejects_only_explicit_nodes_and_rho_with_clear_capability_errors() -> None:
    for topology in (_topology(nodes="explicit"), _topology(coordinate="rho")):
        with pytest.raises(NotImplementedError, match="backend='cxx'|supported only"):
            Kernel(topology=topology, input=_prepared_input(topology), backend="cxx")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("route", "coordinate", "nodes"),
    (
        ("PF", "psin", "uniform"),
        ("PP", "r", "uniform"),
        ("PI", "r", "uniform"),
        ("PJ1", "psin", "uniform"),
        ("PJ2", "psin", "uniform"),
        ("PJ3", "psin", "uniform"),
        ("PQ", "r", "grid"),
    ),
)
def test_cxx_numba_primal_parity_on_supported_route_intersection(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = _topology(route=route, coordinate=coordinate, nodes=nodes)
    config = KernelConfig(max_evaluations=800)
    numba = Kernel(topology=topology, input=_prepared_input(topology), config=config, backend="numba")
    try:
        try:
            cxx = Kernel(
                topology=topology,
                input=_prepared_input(topology),
                config=config,
                backend="cxx",
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}".lower()
            if any(token in detail for token in ("cmake", "compiler", "nanobind", "build", "native")):
                pytest.skip(f"Cxx backend unavailable: {error}")
            raise
        try:
            numba_output = numba.solve()
            cxx_output = cxx.solve()
            assert numba_output.success and cxx_output.success
            np.testing.assert_allclose(cxx_output.x, numba_output.x, rtol=0.0, atol=5.0e-6)
            np.testing.assert_allclose(cxx_output.raw, numba_output.raw, rtol=0.0, atol=5.0e-6)
        finally:
            cxx.close()
    finally:
        numba.close()
