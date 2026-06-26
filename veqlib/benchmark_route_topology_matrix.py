#!/usr/bin/env python3
"""Route/topology coverage gate for the VEQlib backend.

The script reuses ``tests/benchmark.py`` case definitions, but keeps topology
coverage separate from native-kernel coverage:

* every benchmark case is translated into a VEQlib ``Topology`` and, by
  default, dry-run artifact metadata;
* only topologies accepted by the current native backend are executed through
  ``VEQlibSolver`` and compared with the VEQPy/Numba solve.

Outputs are written under ``/tmp`` by default so route exploration does not
dirty repository reference artifacts.
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
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/tmp/veqlib_route_topology_matrix.json")
DEFAULT_MPLCONFIG = Path("/tmp/veqpy-mpl")
VALIDATION_ATOL = 1.0e-6

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veqpy.cpp import (  # noqa: E402
    INITIAL_POLICY_COLD,
    RESIDUAL_NORMALIZATION_FAST,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_POWELL,
    KernelRegistry,
    VEQlibSolver,
    build_kernel,
)
from veqpy.model import Topology, TopologyError  # noqa: E402
from veqpy.operator import Operator  # noqa: E402
from veqpy.solver import Solver  # noqa: E402


@dataclass(frozen=True, slots=True)
class RuntimeCaseData:
    spec: Any
    topology: Topology
    payload_json: str
    py_operator: Any
    py_measure: Any
    solver_method_code: int
    solver_engine_label: str
    x_size: int


_VEQLIB_SOLVER_ENGINE_LABELS = {
    SOLVER_METHOD_POWELL: "veqlib-fastmath-powell",
    SOLVER_METHOD_LEVENBERG_MARQUARDT: "veqlib-fastmath-lm",
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _benchmark_module() -> ModuleType:
    return _load_module(
        "veqpy_route_benchmark_for_veqlib_matrix",
        REPO_ROOT / "tests" / "benchmark.py",
    )


@lru_cache(maxsize=1)
def _benchmark_reference() -> Any:
    return _benchmark_module()._solve_reference(show_progress=False)


def _spec_label(spec: Any) -> str:
    return str(spec.case_name)


def _spec_selector(spec: Any) -> str:
    return f"{spec.mode}:{spec.coordinate}:{spec.input_kind}:{spec.constraint}"


def _cxx_solver_method_for_spec(spec: Any) -> int:
    if (
        str(spec.mode) == "PJ2"
        and str(spec.coordinate) == "psin"
        and str(spec.input_kind) == "grid"
        and str(spec.constraint) == "Ip"
    ):
        return SOLVER_METHOD_LEVENBERG_MARQUARDT
    return SOLVER_METHOD_POWELL


def _iter_route_specs(benchmark: ModuleType, *, include_grid: bool) -> tuple[Any, ...]:
    input_kinds = list(benchmark.BENCHMARK_INPUT_KINDS)
    if include_grid and "grid" not in input_kinds:
        input_kinds.append("grid")
    specs = []
    for mode in benchmark.BENCHMARK_MODES:
        for coordinate in ("rho", "psin"):
            for input_kind in input_kinds:
                for constraint in benchmark.BENCHMARK_MODE_CONSTRAINTS[mode]:
                    specs.append(
                        benchmark.BenchmarkCaseSpec(
                            mode=mode,
                            coordinate=coordinate,
                            constraint=constraint,
                            input_kind=input_kind,
                        )
                    )
    return tuple(specs)


def _filter_specs(specs: tuple[Any, ...], selected: set[str] | None) -> tuple[Any, ...]:
    if selected is None:
        return specs
    selected_lower = {item.lower() for item in selected}
    retained = tuple(
        spec
        for spec in specs
        if _spec_label(spec).lower() in selected_lower
        or _spec_selector(spec).lower() in selected_lower
    )
    matched = {_spec_label(spec).lower() for spec in retained}
    matched.update(_spec_selector(spec).lower() for spec in retained)
    missing = selected_lower.difference(matched)
    if missing:
        raise ValueError(f"unknown case selector(s): {', '.join(sorted(missing))}")
    return retained


def _family_counts(profile_coeffs: dict[str, Any], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, values in profile_coeffs.items()
        if values is not None
        and len(name) > 1
        and name[0] == prefix
        and name[1:].isdigit()
        and _profile_count(profile_coeffs, name) > 0
    ]
    if not orders:
        return ()
    counts = [
        _profile_count(profile_coeffs, f"{prefix}{order}")
        for order in range(first, max(orders) + 1)
    ]
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def _profile_count(profile_coeffs: dict[str, Any], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


def _boundary_m_max(boundary: Any) -> int:
    c_offsets = np.asarray(boundary.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(boundary.s_offsets, dtype=np.float64)
    return max(
        int(c_offsets.size) - 1 if c_offsets.size else 0,
        int(s_offsets.size) - 1 if s_offsets.size else 0,
        1,
    )


def _sample_count_for_spec(benchmark: ModuleType, spec: Any) -> int:
    if str(spec.input_kind).lower() == "grid":
        return int(benchmark.TEST_GRID.Nr)
    return int(benchmark.TEST_SOURCE_SAMPLE_COUNT)


def _topology_from_spec(
    benchmark: ModuleType,
    spec: Any,
    *,
    build: str,
    layout: str = "degree",
) -> tuple[Topology, tuple[str, ...]]:
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    m_max = _boundary_m_max(benchmark.BOUNDARY)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
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
            route=str(spec.mode),
            coordinate=str(spec.coordinate),
            constraint=str(spec.constraint),
            nodes=str(spec.input_kind),
            sample_count=_sample_count_for_spec(benchmark, spec),
            M_max=m_max,
            K_max=max(2, m_max),
            build=build,
            layout=layout,
        )
    warning_messages = tuple(str(item.message) for item in caught)
    return topology, warning_messages


def _stats(values: list[float]) -> dict[str, float | int | list[float]]:
    if not values:
        return {
            "median_ms": float("nan"),
            "mean_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "count": 0,
            "samples_ms": [],
        }
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "count": int(len(values)),
        "samples_ms": [float(value) for value in values],
    }


def _int_stats(values: list[int]) -> dict[str, float | int | list[int]]:
    if not values:
        return {"median": 0, "mean": 0.0, "min": 0, "max": 0, "samples": []}
    return {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
        "samples": [int(value) for value in values],
    }


def _max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    if lhs_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def _runtime_case_data(benchmark: ModuleType, spec: Any, topology: Topology) -> RuntimeCaseData:
    reference = _benchmark_reference()
    case = benchmark._make_benchmark_case(spec, reference)
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    operator = Operator(grid, case)
    x0 = operator.pack_coefficients(benchmark._coefficients_from_coeffs(coeffs))
    boundary = case.boundary
    source_plan = operator.plan.source_plan
    constraints = {"fix_rho": float(operator.fix_rho)}
    if np.isfinite(float(source_plan.scaled_Ip)):
        constraints["scaled_Ip"] = float(source_plan.scaled_Ip)
    if np.isfinite(float(source_plan.beta)):
        constraints["beta"] = float(source_plan.beta)

    solver_method_code = _cxx_solver_method_for_spec(spec)
    payload = {
        "case_name": _spec_label(spec),
        "boundary": {
            "a": float(boundary.a),
            "R0": float(boundary.R0),
            "Z0": float(boundary.Z0),
            "B0": float(boundary.B0),
            "ka": float(boundary.ka),
            "c_offsets": np.asarray(boundary.c_offsets, dtype=np.float64).tolist(),
            "s_offsets": np.asarray(boundary.s_offsets, dtype=np.float64).tolist(),
        },
        "source": {
            "scaled_heat": source_plan.scaled_heat.tolist(),
            "scaled_current": source_plan.scaled_current.tolist(),
        },
        "constraints": constraints,
        "solver": {
            "method_code": solver_method_code,
            "max_residual": float(benchmark.CONFIG.max_residual),
            "max_evaluations": int(x0.size) ** 2,
            "accepted_residual_factor": 10.0,
            "accepted_residual_floor": 1.0e-5,
            "initial_policy_code": INITIAL_POLICY_COLD,
            "residual_normalization_code": RESIDUAL_NORMALIZATION_FAST,
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

    def measure_py(*, warmup: int, repeat: int) -> dict[str, Any]:
        for _ in range(max(1, warmup)):
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
        success_values: list[bool] = []
        for _ in range(repeat):
            solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
            started = time.perf_counter_ns()
            solver.solve(
                x0=x0,
                method=benchmark.CONFIG.method,
                max_residual=benchmark.CONFIG.max_residual,
                max_evaluations=benchmark.CONFIG.max_evaluations,
                enable_verbose=False,
                enable_history=False,
            )
            wall_values.append(float(time.perf_counter_ns() - started) / 1.0e6)
            if solver.result is None:
                raise RuntimeError(f"{_spec_label(spec)} VEQPy solve produced no result")
            nfev_values.append(int(solver.result.function_evaluations))
            success_values.append(bool(solver.result.success))
            last_solver = solver

        if last_solver is None or last_solver.result is None:
            raise RuntimeError(f"{_spec_label(spec)} VEQPy solve did not run")
        raw = last_solver.operator.residual_var(last_solver.result.x)
        return {
            "full_wall": _stats(wall_values),
            "nfev": _int_stats(nfev_values),
            "success_all": all(success_values),
            "x": np.asarray(last_solver.result.x, dtype=np.float64),
            "raw": np.asarray(raw, dtype=np.float64),
            "raw_norm": float(np.linalg.norm(raw)),
            "result_success": bool(last_solver.result.success),
            "message": str(last_solver.result.message),
        }

    return RuntimeCaseData(
        spec=spec,
        topology=topology,
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        py_operator=operator,
        py_measure=measure_py,
        solver_method_code=solver_method_code,
        solver_engine_label=_VEQLIB_SOLVER_ENGINE_LABELS[solver_method_code],
        x_size=int(x0.size),
    )


def _measure_cxx(
    case_data: RuntimeCaseData,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    solver = VEQlibSolver(
        case_data.topology,
        registry=registry,
        solver=case_data.solver_method_code,
    )
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
        wall_values.append(float(time.perf_counter_ns() - start_ns) / 1.0e6)
        elapsed_values.append(float(result[0]))
        success_values.append(bool(result[1]))
        nfev_values.append(int(result[3]))
        raw_norms.append(float(result[9]))
        final_result = result

    if final_result is None:
        raise RuntimeError(f"{_spec_label(case_data.spec)} VEQlib solve did not run")
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


def _compare(cxx: dict[str, Any], py: dict[str, Any]) -> dict[str, float | bool]:
    raw_norm_ratio = float(cxx["raw_norm"] / py["raw_norm"]) if py["raw_norm"] else float("inf")
    py_raw_at_cxx_ratio = (
        float(cxx.get("py_raw_at_cxx_norm", float("nan")) / py["raw_norm"])
        if py["raw_norm"]
        else float("inf")
    )
    x_max_abs = _max_abs(cxx["x"], py["x"])
    raw_max_abs = _max_abs(cxx["raw"], py["raw"])
    return {
        "x_max_abs": x_max_abs,
        "raw_max_abs": raw_max_abs,
        "raw_norm_ratio_to_py": raw_norm_ratio,
        "py_raw_at_cxx_norm_ratio_to_py": py_raw_at_cxx_ratio,
        "within_atol": bool(
            np.isfinite(x_max_abs)
            and np.isfinite(raw_max_abs)
            and x_max_abs <= VALIDATION_ATOL
            and raw_max_abs <= VALIDATION_ATOL
        ),
    }


def _plan_row(
    benchmark: ModuleType,
    spec: Any,
    *,
    build: str,
    layout: str,
    cache_root: Path,
    source_dir: Path,
    skip_artifact_dry_run: bool,
) -> tuple[dict[str, Any], Topology | None]:
    try:
        topology, warning_messages = _topology_from_spec(
            benchmark,
            spec,
            build=build,
            layout=layout,
        )
    except Exception as exc:
        return (
            {
                "case": _spec_label(spec),
                "selector": _spec_selector(spec),
                "topology": {
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "warnings": [],
                },
                "runtime": {"status": "skipped_invalid_topology"},
            },
            None,
        )

    topology_payload: dict[str, Any] = {
        "status": "planned",
        "key": topology.key,
        "source": topology.source_policy_dict(),
        "layout": {
            "packed": topology.layout,
            "profile_first": topology.layout_profile_first,
            "code": topology.layout_code,
        },
        "profile_counts": {
            "h": topology.h_count,
            "v": topology.v_count,
            "kappa": topology.kappa_count,
            "psin": topology.psin_count,
            "F": topology.F_count,
            "c": list(topology.c_counts),
            "s": list(topology.s_counts),
        },
        "grid": {"Nr": topology.Nr, "Nt": topology.Nt},
        "sample_count": topology.sample_count,
        "warnings": list(warning_messages),
    }

    if not skip_artifact_dry_run:
        artifact = build_kernel(
            topology,
            cache_root=cache_root,
            source_dir=source_dir,
            dry_run=True,
        )
        topology_payload["artifact"] = {
            "status": artifact.metadata["artifact"]["status"],
            "artifact_id": artifact.artifact_id,
            "metadata_path": str(artifact.metadata_path),
            "reused": bool(artifact.reused),
        }

    runtime_status: dict[str, Any]
    try:
        topology.validate_supported_for_veqlib_mvp()
    except TopologyError as exc:
        runtime_status = {
            "status": "blocked_unsupported_native_kernel",
            "reason": str(exc),
        }
    else:
        runtime_status = {"status": "ready_supported_native_kernel"}

    return (
        {
            "case": _spec_label(spec),
            "selector": _spec_selector(spec),
            "route": str(spec.mode),
            "coordinate": str(spec.coordinate),
            "nodes": str(spec.input_kind),
            "constraint": str(spec.constraint),
            "topology": topology_payload,
            "runtime": runtime_status,
        },
        topology,
    )


def _run_supported_row(
    benchmark: ModuleType,
    spec: Any,
    topology: Topology,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    case_data = _runtime_case_data(benchmark, spec, topology)
    py = case_data.py_measure(warmup=warmup, repeat=repeat)
    cxx = _measure_cxx(case_data, registry=registry, warmup=warmup, repeat=repeat)
    compare = _compare(cxx, py)
    status = (
        "passed"
        if cxx["success_all"] and py["success_all"] and compare["within_atol"]
        else "failed"
    )
    return {
        "status": status,
        "x_size": case_data.x_size,
        "engines": {
            case_data.solver_engine_label: _compact(cxx),
            "veqpy-numba-hybr": _compact(py),
        },
        "closeness_to_numba": compare,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "topology_planned": 0,
        "topology_invalid": 0,
        "native_ready": 0,
        "native_blocked": 0,
        "runtime_passed": 0,
        "runtime_failed": 0,
        "runtime_not_requested": 0,
    }
    for row in rows:
        topology_status = row["topology"]["status"]
        if topology_status == "planned":
            summary["topology_planned"] += 1
        elif topology_status == "invalid":
            summary["topology_invalid"] += 1
        runtime_status = row["runtime"]["status"]
        if runtime_status == "ready_supported_native_kernel":
            summary["native_ready"] += 1
        elif runtime_status == "blocked_unsupported_native_kernel":
            summary["native_blocked"] += 1
        elif runtime_status == "passed":
            summary["native_ready"] += 1
            summary["runtime_passed"] += 1
        elif runtime_status == "failed":
            summary["native_ready"] += 1
            summary["runtime_failed"] += 1
        elif runtime_status == "not_requested":
            summary["native_ready"] += 1
            summary["runtime_not_requested"] += 1
    return summary


def _write_report(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-grid", action="store_true")
    parser.add_argument("--case", action="append", help="Case name or route:coord:nodes:constraint")
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--layout", default="degree")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "veqlib")
    parser.add_argument("--no-run", action="store_true", help="Do not execute supported kernels")
    parser.add_argument("--skip-artifact-dry-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    benchmark = _benchmark_module()
    specs = _filter_specs(
        _iter_route_specs(benchmark, include_grid=args.include_grid),
        set(args.case) if args.case else None,
    )
    cache_root = args.cache_root or Path(tempfile.mkdtemp(prefix="veqlib-route-matrix-"))
    source_dir = args.source_dir.resolve()
    registry = KernelRegistry(cache_root=cache_root, source_dir=source_dir)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row, topology = _plan_row(
            benchmark,
            spec,
            build=args.build,
            layout=args.layout,
            cache_root=cache_root,
            source_dir=source_dir,
            skip_artifact_dry_run=args.skip_artifact_dry_run,
        )
        if row["runtime"]["status"] == "ready_supported_native_kernel":
            if args.no_run:
                row["runtime"] = {"status": "not_requested"}
            elif topology is not None:
                print(f"running supported VEQlib case {_spec_label(spec)} ...", flush=True)
                try:
                    row["runtime"] = _run_supported_row(
                        benchmark,
                        spec,
                        topology,
                        registry=registry,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                except Exception as exc:
                    row["runtime"] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        rows.append(row)

    payload = {
        "schema": "veqlib.route_topology_matrix.v1",
        "include_grid": bool(args.include_grid),
        "build": str(args.build),
        "layout": str(args.layout),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "validation_atol": VALIDATION_ATOL,
        "cache_root": str(cache_root),
        "source_dir": str(source_dir),
        "summary": _summarize(rows),
        "rows": rows,
    }
    if not args.no_write:
        _write_report(payload, args.output)
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)
    return 0 if payload["summary"]["runtime_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
