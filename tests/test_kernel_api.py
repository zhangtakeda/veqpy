from __future__ import annotations

import json

import numpy as np
from helpers import MU0, tiny_boundary, tiny_pf_problem
from numpy.testing import assert_allclose

import veqpy.kernel as kernel
from veqpy.cpp import INITIAL_POLICY_COLD, RESIDUAL_NORMALIZATION_FAST, SOLVER_METHOD_POWELL
from veqpy.kernel import KernelBuild, KernelInput, KernelSolve, KernelTopology
from veqpy.model import Topology


def make_kernel_topology(**overrides: object) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 3,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
        "Nr": 8,
        "Nt": 8,
        "route": "pf",
        "coordinate": "PSIN",
        "constraint": "ip",
        "nodes": "uniform",
        "sample_count": 9,
    }
    params.update(overrides)
    return KernelTopology(**params)  # type: ignore[arg-type]


def test_kernel_public_exports_are_stable() -> None:
    assert kernel.__all__ == [
        "Kernel",
        "KernelBuild",
        "KernelInput",
        "KernelSolve",
        "KernelTopology",
        "build",
    ]


def test_kernel_topology_is_build_layout_free_and_lowers_to_legacy_topology() -> None:
    topology = make_kernel_topology(c_counts=(0, 0), s_counts=(2, 0, 0), K_max=None)
    same_shape = make_kernel_topology(c_counts=(), s_counts=(2,), L_max=2, M_max=1, K_max=2)
    build = KernelBuild(layout="profile-first", build="release", fp_mode="FMA")

    legacy = topology.to_legacy_topology(build)

    assert topology.to_canonical_dict() == same_shape.to_canonical_dict()
    assert topology.key == same_shape.key
    assert isinstance(legacy, Topology)
    assert legacy.layout == "family"
    assert legacy.build == "release"
    assert legacy.fp_mode == "FMA"
    assert legacy.route == "PF"
    assert legacy.coordinate == "psin"
    assert legacy.constraint == "Ip"
    assert legacy.sample_count == 9
    assert topology.packed_size(build=build) == 9


def test_kernel_build_lowers_options_into_legacy_topology() -> None:
    topology = make_kernel_topology()
    build = KernelBuild(
        layout="family",
        build="fastmath-enzyme",
        enable_thin_lto=False,
        analysis=True,
        enzyme_jacobian_batch_width=8,
    )

    legacy = topology.to_legacy_topology(build)

    assert legacy.layout == "family"
    assert legacy.build_options_dict() == {
        "preset": "fastmath-enzyme",
        "cmake_build_type": "Release",
        "fp_mode": "RELAXED",
        "enable_enzyme": True,
        "enable_native_optimizations": True,
        "enable_thin_lto": False,
        "analysis": True,
        "enzyme_jacobian_batch_width": 8,
    }


def test_kernel_input_lowers_problem_to_engine_scaled_payload() -> None:
    problem = tiny_pf_problem()
    input_data = KernelInput.from_problem(problem, fix_rho=0.125, case_name="tiny")
    payload = input_data.to_payload_dict()

    assert payload["case_name"] == "tiny"
    assert payload["boundary"]["a"] == problem.boundary.a
    assert payload["constraints"]["fix_rho"] == 0.125
    assert_allclose(payload["source"]["scaled_heat"], problem.heat_input * MU0)
    assert_allclose(payload["source"]["scaled_current"], problem.current_input)
    assert_allclose(payload["constraints"]["scaled_Ip"], problem.Ip * MU0)
    assert "beta" not in payload["constraints"]
    assert not input_data.scaled_heat.flags.writeable
    assert not input_data.scaled_current.flags.writeable


def test_kernel_input_direct_constructor_validates_shape() -> None:
    try:
        KernelInput(
            boundary=tiny_boundary(),
            scaled_heat=np.ones((2, 1), dtype=np.float64),
            scaled_current=np.ones(2, dtype=np.float64),
        )
    except ValueError as exc:
        assert "scaled_heat must be 1D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("KernelInput accepted non-1D source data")


def test_kernel_solve_payload_uses_codes_and_x_size_default_budget() -> None:
    solve = KernelSolve(method="powell", initial="cold", norm="fast")

    payload = solve.to_payload_dict(x_size=7)

    assert payload["method_code"] == SOLVER_METHOD_POWELL
    assert payload["initial_policy_code"] == INITIAL_POLICY_COLD
    assert payload["residual_normalization_code"] == RESIDUAL_NORMALIZATION_FAST
    assert payload["max_evaluations"] == 49


def test_kernel_handle_build_dry_run_and_payload_json(tmp_path) -> None:
    topology = make_kernel_topology()
    handle = kernel.build(topology, cache_root=tmp_path, dry_run=True)
    payload = json.loads(handle.payload_json(tiny_pf_problem(), case_name="payload-smoke"))

    assert handle.x_size == 9
    assert handle.legacy_topology.key != topology.key
    assert payload["case_name"] == "payload-smoke"
    assert payload["solver"]["max_evaluations"] == 81
    assert payload["source"]["scaled_heat"]
