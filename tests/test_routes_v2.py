from __future__ import annotations

import pytest

from veqpy import VEQ, KernelConfig, KernelTopology
from veqpy.demo_case import make_demo_plasma


def _route_topology(route: str, coordinate: str) -> KernelTopology:
    nodes = "grid" if coordinate == "rho" else "uniform"
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3 if coordinate == "psin" and route not in {"PJ2", "PJ3"} else 0,
        F_count=2 if route in {"PJ2", "PJ3"} and coordinate == "psin" else 0,
        c_counts=(1, 1, 1),
        s_counts=(1, 1),
        Nr=8,
        Nt=8,
        route=route,
        coordinate=coordinate,
        nodes=nodes,
        constraint="ip",
        sample_count=8,
    )


@pytest.mark.slow
@pytest.mark.parametrize("route", ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ"))
@pytest.mark.parametrize("coordinate", ("r", "psin", "rho"))
def test_numba_public_module_covers_route_coordinate_matrix(route: str, coordinate: str) -> None:
    module = VEQ(
        topology=_route_topology(route, coordinate),
        config=KernelConfig(max_evaluations=300),
    )
    try:
        result = module.run(plasma=make_demo_plasma(), materialize=False)
        assert result.accepted, (route, coordinate, result.detail, result.residual_norm)
        assert result.source_count == 8
    finally:
        module.close()
