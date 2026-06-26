from __future__ import annotations

import subprocess
import sys

import pytest

from veqpy.model import Topology, TopologyError


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


def test_topology_canonicalizes_and_keys_supported_mvp() -> None:
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
    assert topology.source_route_code == 1
    assert topology.source_coordinate_code == 2
    assert topology.source_constraint_code == 1
    assert topology.source_nodes_code == 1
    assert topology.source_active_family == "psin"
    assert topology.source_active_family_code == 1
    assert topology.source_parameterization == "identity"
    assert topology.source_parameterization_code == 0
    assert topology.source_supported_constraints == ("Ip", "beta", "null")
    assert topology.source_uses_ip_constraint is True
    assert topology.source_uses_beta_constraint is False
    assert topology.source_policy_dict() == {
        "route_key": ["PF", "psin", "uniform"],
        "active_family": "psin",
        "active_family_code": 1,
        "parameterization": "identity",
        "parameterization_code": 0,
        "supported_constraints": ["Ip", "beta", "null"],
        "uses_Ip": True,
        "uses_beta": False,
    }
    assert topology.layout == "degree"
    assert topology.layout_code == 0
    assert topology.layout_profile_first is False
    assert topology.build_options_dict() == {
        "preset": "fastmath",
        "cmake_build_type": "Release",
        "fp_mode": "RELAXED",
        "enable_enzyme": False,
        "enable_native_optimizations": True,
        "enable_thin_lto": True,
        "analysis": False,
        "enzyme_jacobian_batch_width": 0,
    }
    assert isinstance(topology.key, str)
    assert topology.compute_key() == topology.key
    topology.validate_supported_for_veqlib_mvp()


def test_topology_key_is_stable_for_inferred_values_and_trailing_zeros() -> None:
    inferred = make_topology(c_counts=(0, 0), s_counts=(3, 0, 0), K_max=None)
    explicit = make_topology(c_counts=(), s_counts=(3,), L_max=5, M_max=1, K_max=2)

    assert inferred.to_canonical_dict() == explicit.to_canonical_dict()
    assert inferred.key == explicit.key


def test_topology_keeps_one_basis_row_for_constant_profiles() -> None:
    topology = make_topology(
        h_count=1,
        kappa_count=1,
        psin_count=1,
        s_counts=(1,),
        L_max=None,
    )

    assert topology.L_max == 1
    topology.validate_supported_for_veqlib_mvp()


def test_topology_key_is_stable_across_python_processes() -> None:
    code = """
from veqpy.model import Topology
print(Topology(
    h_count=3, v_count=0, kappa_count=6, psin_count=6, F_count=0,
    c_counts=(0, 0), s_counts=(3, 0), Nr=32, Nt=16, route='PF',
    coordinate='psin', constraint='Ip', nodes='uniform', sample_count=8,
).key)
"""
    expected = make_topology(c_counts=(0, 0), s_counts=(3, 0)).key
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


def test_key_mismatch_is_rejected() -> None:
    with pytest.raises(TopologyError, match="key does not match"):
        make_topology(key="not-the-canonical-key")


def test_mvp_gate_rejects_unsupported_route_shape() -> None:
    topology = make_topology(route="PQ")

    with pytest.raises(TopologyError, match="PF, PP, PI, PJ1"):
        topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_accepts_pf_route_constraint_slice() -> None:
    route_overrides = (
        {"coordinate": "psin", "nodes": "uniform", "psin_count": 6, "sample_count": 8},
        {"coordinate": "rho", "nodes": "uniform", "psin_count": 0, "sample_count": 8},
        {"coordinate": "rho", "nodes": "grid", "psin_count": 0, "sample_count": 32},
        {"coordinate": "psin", "nodes": "grid", "psin_count": 0, "sample_count": 32},
    )
    for constraint in ("null", "Ip", "beta"):
        for overrides in route_overrides:
            make_topology(constraint=constraint, **overrides).validate_supported_for_veqlib_mvp()


def test_mvp_gate_accepts_pp_route_constraint_slice() -> None:
    route_overrides = (
        {"coordinate": "psin", "nodes": "uniform", "psin_count": 6, "sample_count": 8},
        {"coordinate": "rho", "nodes": "uniform", "psin_count": 0, "sample_count": 8},
        {"coordinate": "rho", "nodes": "grid", "psin_count": 0, "sample_count": 32},
        {"coordinate": "psin", "nodes": "grid", "psin_count": 0, "sample_count": 32},
    )
    for constraint in ("null", "Ip", "beta", "Ip_beta"):
        for overrides in route_overrides:
            topology = make_topology(route="PP", constraint=constraint, **overrides)
            topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_accepts_pi_route_constraint_slice() -> None:
    route_overrides = (
        {"coordinate": "psin", "nodes": "uniform", "psin_count": 6, "sample_count": 8},
        {"coordinate": "rho", "nodes": "uniform", "psin_count": 0, "sample_count": 8},
        {"coordinate": "rho", "nodes": "grid", "psin_count": 0, "sample_count": 32},
        {"coordinate": "psin", "nodes": "grid", "psin_count": 0, "sample_count": 32},
    )
    for constraint in ("null", "Ip", "beta", "Ip_beta"):
        for overrides in route_overrides:
            topology = make_topology(route="PI", constraint=constraint, **overrides)
            topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_accepts_pj1_route_constraint_slice() -> None:
    route_overrides = (
        {"coordinate": "psin", "nodes": "uniform", "psin_count": 6, "sample_count": 8},
        {"coordinate": "rho", "nodes": "uniform", "psin_count": 0, "sample_count": 8},
        {"coordinate": "rho", "nodes": "grid", "psin_count": 0, "sample_count": 32},
        {"coordinate": "psin", "nodes": "grid", "psin_count": 0, "sample_count": 32},
    )
    for constraint in ("null", "Ip", "beta", "Ip_beta"):
        for overrides in route_overrides:
            topology = make_topology(route="PJ1", constraint=constraint, **overrides)
            topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_accepts_pj2_one_pass_route_constraint_slice() -> None:
    route_overrides = (
        {"coordinate": "rho", "nodes": "uniform", "sample_count": 8},
        {"coordinate": "rho", "nodes": "grid", "sample_count": 32},
        {"coordinate": "psin", "nodes": "grid", "sample_count": 32},
    )
    for constraint in ("null", "Ip", "beta", "Ip_beta"):
        for overrides in route_overrides:
            topology = make_topology(
                route="PJ2",
                constraint=constraint,
                psin_count=0,
                F_count=6,
                **overrides,
            )
            topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_keeps_pj2_psin_uniform_blocked() -> None:
    topology = make_topology(
        route="PJ2",
        coordinate="psin",
        nodes="uniform",
        psin_count=0,
        F_count=6,
        sample_count=8,
    )

    with pytest.raises(TopologyError, match="PJ2/rho/uniform"):
        topology.validate_supported_for_veqlib_mvp()


def test_mvp_gate_rejects_source_owned_route_with_active_psin_profile() -> None:
    topology = make_topology(coordinate="rho", psin_count=3)

    with pytest.raises(TopologyError, match="source-owned topology"):
        topology.validate_supported_for_veqlib_mvp()


def test_source_topology_rejects_unsupported_route_constraints() -> None:
    with pytest.raises(TopologyError, match="PF source topology does not support"):
        make_topology(constraint="Ip_beta")


def test_source_topology_captures_route_parameterization() -> None:
    topology = make_topology(route="PP", coordinate="psin", nodes="uniform")

    assert topology.source_parameterization == "sqrt_psin"
    assert topology.source_parameterization_code == 1
    assert topology.source_supported_constraints == ("Ip_beta", "Ip", "beta", "null")


def test_source_topology_codes_cover_pj2_f_ownership() -> None:
    topology = make_topology(route="PJ2", psin_count=0, F_count=6)

    assert topology.source_route_code == 5
    assert topology.source_active_family == "F"
    assert topology.source_active_family_code == 2


def test_source_profile_ownership_warnings_are_python_side() -> None:
    with pytest.warns(UserWarning, match="PJ2 source topology uses active F ownership"):
        make_topology(route="PJ2", psin_count=6, F_count=6)

    with pytest.warns(UserWarning, match="Only PJ2 source topology uses active F ownership"):
        make_topology(F_count=3)


def test_layout_aliases_and_build_option_overrides_are_canonicalized() -> None:
    topology = make_topology(
        layout="profile-first",
        build="fastmath-enzyme",
        fp_mode="FMA",
        enable_thin_lto=False,
        analysis=True,
        enzyme_jacobian_batch_width=8,
    )

    assert topology.layout == "family"
    assert topology.layout_code == 1
    assert topology.layout_profile_first is True
    assert topology.build_options_dict() == {
        "preset": "fastmath-enzyme",
        "cmake_build_type": "Release",
        "fp_mode": "FMA",
        "enable_enzyme": True,
        "enable_native_optimizations": True,
        "enable_thin_lto": False,
        "analysis": True,
        "enzyme_jacobian_batch_width": 8,
    }


def test_topology_accepts_exact_cpp_build_modes() -> None:
    for build in ("fastmath", "fastmath-enzyme", "release", "debug"):
        topology = make_topology(build=build)
        assert topology.build == build


def test_topology_rejects_build_mode_aliases_and_case_variants() -> None:
    for build in ("FastMath", "fastmath_enzyme"):
        with pytest.raises(TopologyError, match="build must be one of"):
            make_topology(build=build)
