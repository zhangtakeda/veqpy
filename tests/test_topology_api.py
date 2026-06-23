from __future__ import annotations

import subprocess
import sys

import pytest

from veqpy.topology import Topology, TopologyError


def make_topology(**overrides: object) -> Topology:
    params: dict[str, object] = {
        "h_count": 3,
        "v_count": 0,
        "kappa_count": 6,
        "psin_count": 6,
        "F_count": 0,
        "c_counts": (0, 0, 0),
        "s_counts": (3, 0, 0),
        "Nr": 32,
        "Nt": 16,
        "route": "pf",
        "coordinate": "PSIN",
        "constraint": "ip",
        "nodes": "uniform",
        "sample_count": 8,
    }
    params.update(overrides)
    return Topology(**params)  # type: ignore[arg-type]


def test_topology_canonicalizes_and_hashes_supported_mvp() -> None:
    topology = make_topology()

    assert topology.route == "PF"
    assert topology.coordinate == "psin"
    assert topology.constraint == "Ip"
    assert topology.nodes == "uniform"
    assert topology.c_counts == ()
    assert topology.s_counts == (3,)
    assert topology.L_max == 5
    assert topology.M_max == 1
    assert topology.K_max == 2
    assert isinstance(topology.artifact_id, str)
    assert topology.compute_artifact_id() == topology.artifact_id
    topology.validate_supported_for_veqlib_mvp()


def test_topology_hash_is_stable_for_inferred_values_and_trailing_zeros() -> None:
    inferred = make_topology(c_counts=(0, 0), s_counts=(3, 0, 0), K_max=None)
    explicit = make_topology(c_counts=(), s_counts=(3,), L_max=5, M_max=1, K_max=2)

    assert inferred.to_canonical_dict() == explicit.to_canonical_dict()
    assert inferred.artifact_id == explicit.artifact_id


def test_topology_hash_is_stable_across_python_processes() -> None:
    code = """
from veqpy.topology import Topology
print(Topology(
    h_count=3, v_count=0, kappa_count=6, psin_count=6, F_count=0,
    c_counts=(0, 0), s_counts=(3, 0), Nr=32, Nt=16, route='PF',
    coordinate='psin', constraint='Ip', nodes='uniform', sample_count=8,
).artifact_id)
"""
    expected = make_topology(c_counts=(0, 0), s_counts=(3, 0)).artifact_id
    actual = subprocess.check_output([sys.executable, "-c", code], text=True).strip()

    assert actual == expected


def test_grid_nodes_infer_sample_count_from_nr() -> None:
    topology = make_topology(nodes="grid", sample_count=None)

    assert topology.sample_count == topology.Nr


def test_uniform_nodes_require_sample_count() -> None:
    with pytest.raises(TopologyError, match="uniform source nodes require"):
        make_topology(sample_count=None)


def test_explicit_l_max_must_match_inferred_value() -> None:
    with pytest.raises(TopologyError, match="L_max is inferred as 5"):
        make_topology(L_max=4)


def test_artifact_id_mismatch_is_rejected() -> None:
    with pytest.raises(TopologyError, match="artifact_id does not match"):
        make_topology(artifact_id="not-the-canonical-id")


def test_mvp_gate_rejects_unsupported_route_shape() -> None:
    topology = make_topology(route="PQ")

    with pytest.raises(TopologyError, match="PF/psin/uniform/Ip"):
        topology.validate_supported_for_veqlib_mvp()
