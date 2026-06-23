#!/usr/bin/env python3
"""Four-case VEQlib-vs-VEQPy benchmark gate.

This script is intentionally Python-facing: it exercises the same Topology ->
nanobind artifact -> KernelSolver path used by VEQPy, then compares against the
VEQPy/Numba solver for the 18-parameter PF benchmark plus the three GEQDSK cases
used by the Figure 06 workflow (solovev, chease, efit).

Write outputs outside the repository by default so benchmark runs do not dirty
tracked reference data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/tmp/veqlib_4case_compare.json")
DEFAULT_MPLCONFIG = Path("/tmp/veqpy-mpl")

# Keep BLAS/runtime thread counts deterministic before importing numpy-heavy code.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_cpp_geqdsk_incremental import (  # noqa: E402
    _case_bundle,
    _case_specs,
    _kernel_payload,
)
from veqpy.cpp import KernelRegistry, VEQlibSolver  # noqa: E402
from veqpy.operator import Operator  # noqa: E402
from veqpy.solver import Solver  # noqa: E402
from veqpy.topology import Topology  # noqa: E402


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quantile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    return values_sorted[int((len(values_sorted) - 1) * q)]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "p05_ms": float(_quantile(values, 0.05)),
        "p95_ms": float(_quantile(values, 0.95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _int_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def _max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    if lhs_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def _family_counts(profile_coeffs: dict[str, Any], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name in profile_coeffs
        if len(name) > 1 and name[0] == prefix and name[1:].isdigit()
    ]
    if not orders:
        return ()
    counts = []
    for order in range(first, max(orders) + 1):
        values = profile_coeffs.get(f"{prefix}{order}")
        counts.append(0 if values is None else int(np.asarray(values, dtype=np.float64).size))
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def _profile_count(profile_coeffs: dict[str, Any], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


@dataclass(frozen=True)
class CaseData:
    name: str
    topology: Topology
    payload_json: str
    py_measure: Any
    py_operator: Any
    x_size: int
    metadata: dict[str, Any]


def _benchmark_case_data(repeat: int, warmup: int) -> CaseData:
    benchmark = _load_module(
        "veqpy_route_benchmark_for_cpp_4case",
        REPO_ROOT / "tests" / "benchmark.py",
    )
    ref = benchmark._solve_reference(show_progress=False)
    spec = benchmark.BenchmarkCaseSpec(
        mode="PF", coordinate="psin", input_kind="uniform", constraint="Ip"
    )
    case = benchmark._make_benchmark_case(spec, ref)
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    op = Operator(grid, case)
    x0 = op.pack_coefficients(benchmark._coefficients_from_coeffs(coeffs))
    boundary = case.boundary
    c_offsets = np.asarray(boundary.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(boundary.s_offsets, dtype=np.float64)
    active_boundary_m = max(
        int(c_offsets.size) - 1 if c_offsets.size else 0,
        int(s_offsets.size) - 1 if s_offsets.size else 0,
        1,
    )
    topology = Topology(
        h_count=_profile_count(coeffs, "h"),
        v_count=_profile_count(coeffs, "v"),
        kappa_count=_profile_count(coeffs, "k"),
        psin_count=_profile_count(coeffs, "psin"),
        F_count=_profile_count(coeffs, "F"),
        c_counts=_family_counts(coeffs, "c", 0),
        s_counts=_family_counts(coeffs, "s", 1),
        Nr=int(grid.Nr),
        Nt=int(grid.Nt),
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=int(np.asarray(case.heat_input, dtype=np.float64).size),
        M_max=active_boundary_m,
        K_max=max(2, active_boundary_m),
        build="fastmath",
    )
    source_plan = op.plan.source_plan
    payload = {
        "case_name": spec.case_name,
        "boundary": {
            "a": float(boundary.a),
            "R0": float(boundary.R0),
            "Z0": float(boundary.Z0),
            "B0": float(boundary.B0),
            "ka": float(boundary.ka),
            "c_offsets": c_offsets.tolist(),
            "s_offsets": s_offsets.tolist(),
        },
        "source": {
            "scaled_heat": source_plan.scaled_heat.tolist(),
            "scaled_current": source_plan.scaled_current.tolist(),
        },
        "constraints": {"scaled_Ip": float(source_plan.scaled_Ip), "fix_rho": float(op.fix_rho)},
        "solver": {
            "method_code": 1,
            "max_residual": float(benchmark.CONFIG.max_residual),
            "max_evaluations": int(x0.size) ** 2,
            "accepted_residual_factor": 10.0,
            "accepted_residual_floor": 1.0e-5,
            "initial_policy_code": 3,
            "residual_normalization_code": 1,
            "residual_normalization_floor": float(benchmark.CONFIG.residual_normalization_floor),
            "residual_normalization_max_ratio": float(
                benchmark.CONFIG.residual_normalization_max_ratio
            ),
            "residual_normalization_huber_tau": float(
                benchmark.CONFIG.residual_normalization_huber_tau
            ),
            "residual_normalization_probe_count": int(
                benchmark.CONFIG.residual_normalization_probe_count
            ),
            "residual_normalization_probe_step": float(
                benchmark.CONFIG.residual_normalization_probe_step
            ),
            "residual_normalization_sensitivity_lambda": float(
                benchmark.CONFIG.residual_normalization_sensitivity_lambda
            ),
        },
    }

    def measure_numba() -> dict[str, Any]:
        warm_solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
        warm_solver.solve(
            x0=x0,
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        for _ in range(warmup):
            solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
            solver.solve(
                x0=x0,
                method=benchmark.CONFIG.method,
                max_residual=benchmark.CONFIG.max_residual,
                max_evaluations=benchmark.CONFIG.max_evaluations,
                enable_verbose=False,
                enable_history=False,
            )
        wall_values: list[float] = []
        nfev_values: list[int] = []
        last_solver = None
        for _ in range(repeat):
            solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
            start = time.perf_counter_ns()
            solver.solve(
                x0=x0,
                method=benchmark.CONFIG.method,
                max_residual=benchmark.CONFIG.max_residual,
                max_evaluations=benchmark.CONFIG.max_evaluations,
                enable_verbose=False,
                enable_history=False,
            )
            wall_values.append((time.perf_counter_ns() - start) / 1.0e6)
            if solver.result is None:
                raise RuntimeError("benchmark numba solve produced no result")
            nfev_values.append(int(solver.result.function_evaluations))
            last_solver = solver
        assert last_solver is not None and last_solver.result is not None
        raw = last_solver.operator.residual_var(last_solver.result.x)
        return {
            "full_wall": _stats(wall_values),
            "nfev": _int_stats(nfev_values),
            "success_all": True,
            "x": np.asarray(last_solver.result.x, dtype=np.float64),
            "raw": np.asarray(raw, dtype=np.float64),
            "raw_norm": float(np.linalg.norm(raw)),
            "result_success": bool(last_solver.result.success),
            "message": str(last_solver.result.message),
        }

    return CaseData(
        name=spec.case_name,
        topology=topology,
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        py_measure=measure_numba,
        py_operator=op,
        x_size=int(x0.size),
        metadata={
            "kind": "benchmark",
            "M_max": int(topology.M_max),
            "sample_count": int(topology.sample_count),
            "grid": [int(grid.Nr), int(grid.Nt)],
        },
    )


def _geqdsk_case_data(spec: Any, repeat: int, warmup: int) -> CaseData:
    fig06, _geqdsk, case, _boundary, topology, payload = _case_bundle(spec)
    grid = fig06.Grid(Nr=int(spec.solve_nr), Nt=int(spec.solve_nt))
    op_for_compare = Operator(grid, case.copy())
    payload_json = json.dumps(
        _kernel_payload(fig06, spec, case, payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    def measure_numba() -> dict[str, Any]:
        warm_solver = fig06.build_solver(case, grid)
        fig06.solve_existing_solver_once(warm_solver)
        for _ in range(warmup):
            solver = fig06.build_solver(case, grid)
            fig06.solve_existing_solver_once(solver)
        wall_values: list[float] = []
        nfev_values: list[int] = []
        last_solver = None
        for _ in range(repeat):
            solver = fig06.build_solver(case, grid)
            start = time.perf_counter_ns()
            solved, _elapsed_ms, _wall_ms = fig06.solve_existing_solver_once(solver)
            wall_values.append((time.perf_counter_ns() - start) / 1.0e6)
            if solved.result is None:
                raise RuntimeError(f"{spec.case_key} numba solve produced no result")
            nfev_values.append(int(solved.result.function_evaluations))
            last_solver = solved
        assert last_solver is not None and last_solver.result is not None
        raw = last_solver.operator.residual_var(last_solver.result.x)
        return {
            "full_wall": _stats(wall_values),
            "nfev": _int_stats(nfev_values),
            "success_all": True,
            "x": np.asarray(last_solver.result.x, dtype=np.float64),
            "raw": np.asarray(raw, dtype=np.float64),
            "raw_norm": float(np.linalg.norm(raw)),
            "result_success": bool(last_solver.result.success),
            "message": str(last_solver.result.message),
        }

    return CaseData(
        name=str(spec.case_key),
        topology=topology,
        payload_json=payload_json,
        py_measure=measure_numba,
        py_operator=op_for_compare,
        x_size=int(topology.h_count)
        + int(topology.kappa_count)
        + int(topology.psin_count)
        + sum(int(value) for value in topology.c_counts)
        + sum(int(value) for value in topology.s_counts)
        + int(topology.F_count),
        metadata={
            "kind": "geqdsk",
            "M_max": int(topology.M_max),
            "sample_count": int(topology.sample_count),
            "grid": [int(topology.Nr), int(topology.Nt)],
        },
    )


def _measure_cxx(
    case_data: CaseData,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    solver = VEQlibSolver(case_data.topology, registry=registry, solver="powell")
    build_start = time.perf_counter()
    artifact = solver.build(force=False, dry_run=False)
    build_wall_ms = (time.perf_counter() - build_start) * 1000.0
    solver.set_case_json(case_data.payload_json)
    for _ in range(warmup):
        solver.solve_direct()
    wall_values: list[float] = []
    elapsed_values: list[float] = []
    nfev_values: list[int] = []
    success_values: list[bool] = []
    raw_norms: list[float] = []
    final_result = None
    for _ in range(repeat):
        start_ns = time.perf_counter_ns()
        result = solver.solve_direct()
        wall_values.append((time.perf_counter_ns() - start_ns) / 1.0e6)
        elapsed_values.append(float(result[0]))
        success_values.append(bool(result[1]))
        nfev_values.append(int(result[3]))
        raw_norms.append(float(result[9]))
        final_result = result
    assert final_result is not None
    x = np.asarray(final_result[11], dtype=np.float64).copy()
    raw = np.asarray(final_result[12], dtype=np.float64).copy()
    py_raw_at_cxx = np.asarray(case_data.py_operator.residual_var(x), dtype=np.float64)
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_reused": bool(artifact.reused),
        "build_wall_ms": float(build_wall_ms),
        "build_elapsed_ms": float(artifact.metadata["build"]["elapsed_ms"]),
        "full_wall": _stats(wall_values),
        "internal_elapsed": _stats(elapsed_values),
        "nfev": _int_stats(nfev_values),
        "success_all": all(success_values),
        "x": x,
        "raw": raw,
        "raw_norm": float(statistics.median(raw_norms)),
        "py_raw_at_cxx_norm": float(np.linalg.norm(py_raw_at_cxx)),
    }


def _compact(engine: dict[str, Any]) -> dict[str, Any]:
    out = dict(engine)
    out.pop("x", None)
    out.pop("raw", None)
    return out


def _compare(cxx: dict[str, Any], py: dict[str, Any]) -> dict[str, float]:
    return {
        "x_max_abs": _max_abs(cxx["x"], py["x"]),
        "raw_max_abs": _max_abs(cxx["raw"], py["raw"]),
        "raw_norm_ratio_to_py": float(cxx["raw_norm"] / py["raw_norm"])
        if py["raw_norm"]
        else float("inf"),
        "py_raw_at_cxx_norm_ratio_to_py": (
            float(cxx.get("py_raw_at_cxx_norm", float("nan")) / py["raw_norm"])
            if py["raw_norm"]
            else float("inf")
        ),
    }


def _make_cases(repeat: int, warmup: int, selected: set[str] | None) -> list[CaseData]:
    all_cases = [_benchmark_case_data(repeat, warmup)]
    all_cases.extend(_geqdsk_case_data(spec, repeat, warmup) for spec in _case_specs())
    if selected is None:
        return all_cases
    cases = [case for case in all_cases if case.name in selected]
    missing = selected.difference(case.name for case in cases)
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    return cases


def _write_report(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--case",
        action="append",
        choices=("PF_psin_uniform_Ip", "solovev", "chease", "efit"),
        help="Run only the selected case; repeat the option for multiple cases.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    selected = set(args.case) if args.case else None
    cases = _make_cases(args.repeat, args.warmup, selected)
    cache = args.cache_root or Path(tempfile.mkdtemp(prefix="veqlib-4case-"))
    registry = KernelRegistry(cache_root=cache, source_dir=REPO_ROOT / "veqlib")
    rows = []
    for case in cases:
        print(f"measuring {case.name} ...", flush=True)
        py = case.py_measure()
        cxx = _measure_cxx(case, registry=registry, warmup=args.warmup, repeat=args.repeat)
        rows.append(
            {
                "case": case.name,
                "x_size": case.x_size,
                "metadata": case.metadata,
                "engines": {
                    "veqlib-fastmath-powell": _compact(cxx),
                    "veqpy-numba-hybr": _compact(py),
                },
                "closeness_to_numba": _compare(cxx, py),
            }
        )

    payload = {
        "schema": "veqlib.fastmath_vs_numba_4case.v1",
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cache": str(cache),
        "rows": rows,
    }
    if not args.no_write:
        _write_report(payload, args.output)
        print(f"json: {args.output}")

    print(
        f"repeat={args.repeat} warmup={args.warmup} "
        f"affinity={payload['cpu_affinity']} env={payload['env']}"
    )
    print("\nSolve wall-time median [ms]")
    print("| case | x_size | VEQlib fastmath | VEQPy Numba | speedup |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        cxx_median = row["engines"]["veqlib-fastmath-powell"]["full_wall"]["median_ms"]
        py_median = row["engines"]["veqpy-numba-hybr"]["full_wall"]["median_ms"]
        print(
            f"| {row['case']} | {row['x_size']} | {cxx_median:.6f} | "
            f"{py_median:.6f} | {py_median / cxx_median:.3f}x |"
        )

    print("\nnfev median")
    print("| case | VEQlib fastmath | VEQPy Numba |")
    print("|---|---:|---:|")
    for row in rows:
        cxx_nfev = row["engines"]["veqlib-fastmath-powell"]["nfev"]["median"]
        py_nfev = row["engines"]["veqpy-numba-hybr"]["nfev"]["median"]
        print(f"| {row['case']} | {cxx_nfev} | {py_nfev} |")

    print("\nCloseness to VEQPy Numba")
    print("| case | x_max_abs | raw_max_abs | py_raw_at_cxx/raw_py |")
    print("|---|---:|---:|---:|")
    for row in rows:
        closeness = row["closeness_to_numba"]
        print(
            f"| {row['case']} | {closeness['x_max_abs']:.3e} | "
            f"{closeness['raw_max_abs']:.3e} | "
            f"{closeness['py_raw_at_cxx_norm_ratio_to_py']:.3e} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
