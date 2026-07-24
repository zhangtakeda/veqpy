from __future__ import annotations

import pytest
from _helpers import (
    assert_finite,
    assert_runtime_passed,
    benchmark_args,
    skip_if_native_unavailable,
)

from benchmarks import cxx_routes, numba_routes, numba_variant_sweep
from benchmarks._common import (
    CASE_KEYS,
    CONFIG_LABELS,
    ROUTE_SHAPE_MATCH_TOL,
    SYNTHETIC_SOLVER_MAX_EVALUATIONS,
    SYNTHETIC_SOLVER_MAX_RESIDUAL,
    RouteBenchmarkSpec,
    geqdsk_kernel_case,
    geqdsk_signature,
)

ROUTE_REGRESSION_CASES = (
    RouteBenchmarkSpec("PF", "rho", "uniform", "ip"),
    RouteBenchmarkSpec("PP", "psin", "uniform", "ip"),
    RouteBenchmarkSpec("PJ2", "psin", "uniform", "ip"),
    RouteBenchmarkSpec("PQ", "rho", "uniform", "ip"),
)


@pytest.mark.parametrize("case_key", CASE_KEYS)
@pytest.mark.parametrize("config_label", CONFIG_LABELS)
def test_geqdsk_benchmark_cases_materialize_from_reference_inputs(
    case_key: str,
    config_label: str,
) -> None:
    signature = geqdsk_signature(case_key, config_label)
    case = geqdsk_kernel_case(case_key, config_label)

    assert signature
    assert case.topology.x_size > 0
    assert case.boundary.B0 > 0.0
    assert case.source.pprime.size == case.topology.sample_count
    assert case.source.driver_profile.size == case.topology.sample_count


def test_variant_sweep_benchmark_plans_geqdsk_pareto_rows() -> None:
    args = benchmark_args(no_run=True)

    rows = []
    for case_key in CASE_KEYS:
        rows.extend(numba_variant_sweep._measure_case_sweep(args, case_key, CONFIG_LABELS))

    assert len(rows) == len(CASE_KEYS) * len(CONFIG_LABELS)
    assert {row["row"] for row in rows} == {
        f"{case_key}:{config_label.lower()}"
        for case_key in CASE_KEYS
        for config_label in CONFIG_LABELS
    }
    assert all(row["runtime"]["status"] == "not_requested" for row in rows)


@pytest.mark.slow
@pytest.mark.parametrize("spec", ROUTE_REGRESSION_CASES, ids=lambda spec: spec.case_name)
def test_numba_route_benchmark_regresses_against_synthetic_reference(
    spec: RouteBenchmarkSpec,
) -> None:
    args = benchmark_args(max_evaluations=SYNTHETIC_SOLVER_MAX_EVALUATIONS)

    row = numba_routes._measure_row(args, spec)

    assert_runtime_passed(row)
    runtime = row["runtime"]
    engine = runtime["engines"]["numba-hybr"]
    diagnostics = runtime["diagnostics"]
    assert engine["success_all"] is True
    assert_finite(engine["raw_norm"], name=f"{spec.case_name} raw_norm")
    assert engine["raw_norm"] <= SYNTHETIC_SOLVER_MAX_RESIDUAL * 10.0
    assert diagnostics["shape_error"] <= ROUTE_SHAPE_MATCH_TOL * 1.5
    assert diagnostics["psi_r_rel_rms_error"] <= 1.0e-2
    assert diagnostics["ff_psi_rel_rms_error"] <= 7.0e-2
    assert diagnostics["mu0_p_psi_rel_rms_error"] <= 2.0e-2


@pytest.mark.slow
@pytest.mark.parametrize("spec", ROUTE_REGRESSION_CASES, ids=lambda spec: spec.case_name)
def test_cxx_route_benchmark_matches_numba_reference(spec: RouteBenchmarkSpec) -> None:
    args = benchmark_args(max_evaluations=SYNTHETIC_SOLVER_MAX_EVALUATIONS)

    row = cxx_routes._measure_row(args, spec)

    skip_if_native_unavailable(row.get("runtime", row))
    assert_runtime_passed(row)
    runtime = row["runtime"]
    closeness = runtime["closeness_to_numba"]
    assert closeness["x_max_abs"] <= 5.0e-6
    assert closeness["raw_max_abs"] <= 5.0e-6
    assert closeness["within_atol"] is True
    for engine in runtime["engines"].values():
        assert engine["success_all"] is True
        assert_finite(engine["raw_norm"], name=f"{spec.case_name} raw_norm")
