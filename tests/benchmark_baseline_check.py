from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "benchmark_non_timing.json"
BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.py"

FLOAT_CASE_KEYS = (
    "residual_norm_final",
    "state_norm",
    "state_min",
    "state_max",
    "shape_error",
    "psi_r_rel_rms_error",
    "psi_r_rel_max_error",
    "ff_psi_rel_rms_error",
    "ff_psi_rel_max_error",
    "mu0_p_psi_rel_rms_error",
    "mu0_p_psi_rel_max_error",
)

INT_CASE_KEYS = (
    "function_evaluations",
    "jacobian_evaluations",
    "iterations",
    "state_size",
    "psi_r_head_sign_changes",
    "psi_r_tail_sign_changes",
    "ff_psi_head_sign_changes",
    "ff_psi_tail_sign_changes",
    "mu0_p_psi_head_sign_changes",
    "mu0_p_psi_tail_sign_changes",
)

EXACT_CASE_KEYS = (
    "case_name",
    "mode",
    "coordinate",
    "constraint",
    "input_kind",
    "success",
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("veqpy_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_non_timing_matches_baseline() -> None:
    if not BASELINE_PATH.exists():
        raise AssertionError(
            f"baseline not generated yet: run "
            f"`{sys.executable} tests/benchmark.py --write-baseline`"
        )

    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    benchmark = _load_benchmark_module()
    actual = benchmark.build_benchmark_baseline_payload(show_progress=False)

    assert actual.keys() == expected.keys()
    for key in (
        "schema_version",
        "reference_cache_version",
        "case_count",
        "modes",
        "input_kinds",
        "mode_constraints",
        "shape_match_tol",
        "test",
    ):
        assert actual[key] == expected[key]

    for key in (
        "case",
        "grid",
        "source_sample_count",
        "Ip",
        "scaled_Ip",
        "function_evaluations",
        "jacobian_evaluations",
        "iterations",
    ):
        assert actual["reference"][key] == expected["reference"][key]
    assert_allclose(
        actual["reference"]["residual_norm_final"],
        expected["reference"]["residual_norm_final"],
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    assert len(actual["cases"]) == len(expected["cases"])
    for actual_case, expected_case in zip(actual["cases"], expected["cases"], strict=True):
        assert actual_case.keys() == expected_case.keys()
        for key in EXACT_CASE_KEYS:
            assert actual_case[key] == expected_case[key]
        for key in INT_CASE_KEYS:
            assert actual_case[key] == expected_case[key]
        for key in FLOAT_CASE_KEYS:
            assert np.isfinite(actual_case[key])
            assert_allclose(
                actual_case[key],
                expected_case[key],
                rtol=1.0e-6,
                atol=1.0e-8,
            )


def test_pf_rho_source_reversal_preserves_solved_shape() -> None:
    benchmark = _load_benchmark_module()
    reference = benchmark._solve_reference(show_progress=False)

    for constraint in ("null", "beta", "Ip"):
        for input_kind in benchmark.BENCHMARK_INPUT_KINDS:
            spec = benchmark.BenchmarkCaseSpec(
                mode="PF",
                coordinate="rho",
                constraint=constraint,
                input_kind=input_kind,
            )
            case = benchmark._make_benchmark_case(spec, reference)
            initial_coeffs = benchmark._case_profile_coeffs(spec)
            result, equilibrium, shape_x = benchmark._solve_once(case, initial_coeffs)

            reversed_case = case.replace(
                heat_input=-case.heat_input,
                current_input=-case.current_input,
            )
            reversed_result, reversed_equilibrium, reversed_shape_x = benchmark._solve_once(
                reversed_case,
                initial_coeffs,
            )

            n = min(shape_x.shape[0], reversed_shape_x.shape[0])
            assert result.success
            assert reversed_result.success
            assert_allclose(reversed_shape_x[:n], shape_x[:n], rtol=0.0, atol=1.0e-12)
            assert_allclose(
                reversed_equilibrium.beta_t,
                equilibrium.beta_t,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            if constraint == "Ip":
                assert_allclose(
                    reversed_equilibrium.Ip,
                    equilibrium.Ip,
                    rtol=1.0e-12,
                    atol=1.0e-6,
                )
