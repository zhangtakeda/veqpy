from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy import KernelSource, KernelTopology
from veqpy.kernels.abi.source_semantics import MU0, materialize_kernel_source
from veqpy.numerics import make_quadrature

SOURCE_ROUTE_CASES = (
    ("PF", "rho", "uniform"),
    ("PP", "psin", "uniform"),
    ("PI", "rho", "uniform"),
    ("PJ1", "psin", "uniform"),
    ("PJ2", "psin", "uniform"),
    ("PQ", "rho", "grid"),
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
    if route == "PJ2":
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
        ip_constraint=True,
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
    heat = (
        rho * (1.0e6 + 0.4e6 * rho * rho)
        if topology.coordinate == "rho"
        else 1.0e6 + 0.4e6 * rho * rho
    )
    if topology.route == "PI":
        current = rho * rho * (1.0e6 + 0.8e6 * rho * rho)
    elif topology.route in {"PJ1", "PJ2"}:
        current = 1.0e6 + 0.8e6 * rho * rho
    elif topology.route == "PP":
        current = rho * (1.0e6 + 0.8e6 * rho * rho)
    elif topology.route == "PF":
        current = (
            rho * (1.0e6 + 0.8e6 * rho * rho)
            if topology.coordinate == "rho"
            else 1.0e6 + 0.8e6 * rho * rho
        )
    else:
        current = 1.0e6 + 0.8e6 * rho * rho
    return heat.astype(np.float64), current.astype(np.float64)


@pytest.mark.parametrize(("route", "coordinate", "nodes"), SOURCE_ROUTE_CASES)
def test_route_source_lowering_preserves_raw_user_physics(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = _topology(route, coordinate, nodes)
    heat, current = _route_source_profiles(topology)
    source = KernelSource(heat_profile=heat, current_profile=current, Ip=3.0e6, beta=0.02)

    materialized = materialize_kernel_source(topology, source)

    assert materialized.scaled_heat.flags.writeable is False
    assert materialized.scaled_current.flags.writeable is False
    assert_allclose(materialized.scaled_heat, heat * MU0)
    assert materialized.scaled_Ip == pytest.approx(3.0e6 * MU0)
    assert materialized.beta == pytest.approx(0.02)
    if route in {"PI", "PJ1", "PJ2"}:
        assert_allclose(materialized.scaled_current, current * MU0)
    else:
        assert_allclose(materialized.scaled_current, current)


def test_source_lowering_accepts_large_physical_profiles_when_constraints_are_valid() -> None:
    topology = _topology("PF", "psin", "uniform")
    heat = np.full(topology.sample_count, 1.0e12, dtype=np.float64)
    current = np.full(topology.sample_count, -1.0e12, dtype=np.float64)
    source = KernelSource(heat_profile=heat, current_profile=current, Ip=3.0e6)

    materialized = materialize_kernel_source(topology, source)

    assert_allclose(materialized.scaled_heat, heat * MU0)
    assert_allclose(materialized.scaled_current, current)


def test_source_lowering_rejects_nonfinite_route_profiles() -> None:
    topology = _topology("PI", "rho", "uniform")
    heat, current = _route_source_profiles(topology)
    current[-1] = np.nan

    with pytest.raises(ValueError, match="current_profile must contain only finite values"):
        materialize_kernel_source(
            topology,
            KernelSource(heat_profile=heat, current_profile=current, Ip=3.0e6),
        )
