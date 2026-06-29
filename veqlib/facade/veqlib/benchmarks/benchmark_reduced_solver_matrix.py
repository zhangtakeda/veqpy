#!/usr/bin/env python3
"""Benchmark Figure 07 reduced cases across VEQPy and VEQlib solver backends."""

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
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
VEQLIB_ROOT = THIS_FILE.parents[3]
REPO_ROOT = THIS_FILE.parents[4]
FACADE_ROOT = VEQLIB_ROOT / "facade"
CORE_DIR = VEQLIB_ROOT / "core"
SCRIPT_DIR = REPO_ROOT / "scripts"
FIG07_PATH = SCRIPT_DIR / "07-pareto-analysis.py"
DEFAULT_OUTPUT = Path("/tmp/veqlib_reduced_solver_matrix.json")
DEFAULT_MPLCONFIG = Path("/tmp/veqpy-mpl")
VALIDATION_ATOL = 1.0e-6

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG))

for path in (FACADE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    CASE_KEYS,
    REDUCED_CONFIG_LABELS,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REFERENCE_SOLVER_MAXFEV,
    SOLVER_INITIAL_POLICY,
    build_pf_case,
    build_pf_reference_case,
    load_pf_benchmark,
)

from veqlib.kernel import (  # noqa: E402
    INITIAL_POLICY_COLD,
    RESIDUAL_NORMALIZATION_FAST,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_NEWTON_KRYLOV,
    SOLVER_METHOD_NEWTON_RAPHSON,
    SOLVER_METHOD_POWELL,
    KernelRegistry,
    KernelTopology,
    VEQlibSolver,
)

Topology = KernelTopology

SOLVERS: tuple[tuple[str, str, int | None], ...] = (
    ("veqpy-hybr", "veqpy", None),
    ("powell-fd", "fastmath", SOLVER_METHOD_POWELL),
    ("powell-ad", "fastmath-enzyme", SOLVER_METHOD_POWELL),
    ("lm-fd", "fastmath", SOLVER_METHOD_LEVENBERG_MARQUARDT),
    ("lm-ad", "fastmath-enzyme", SOLVER_METHOD_LEVENBERG_MARQUARDT),
    ("nr-fd", "fastmath", SOLVER_METHOD_NEWTON_RAPHSON),
    ("nr-ad", "fastmath-enzyme", SOLVER_METHOD_NEWTON_RAPHSON),
    ("nk-fd", "fastmath", SOLVER_METHOD_NEWTON_KRYLOV),
    ("nk-ad", "fastmath-enzyme", SOLVER_METHOD_NEWTON_KRYLOV),
)


@dataclass(frozen=True, slots=True)
class ReducedCase:
    case_key: str
    config_label: str
    row_label: str
    signature: dict[str, int]
    topology_base: Topology
    payload_json: str
    py_measure: Any
    py_operator: Any
    x_size: int


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _fig07_module() -> ModuleType:
    return _load_module("veqpy_figure07_for_veqlib_matrix", FIG07_PATH)


def _reference_layout_m_max() -> int:
    return int(_fig07_module().REFERENCE_LAYOUT_M_MAX)


def _stats(samples_ms: list[float]) -> dict[str, float | int | list[float]]:
    if not samples_ms:
        return {
            "median_ms": float("nan"),
            "mean_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "samples_ms": [],
            "count": 0,
        }
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.mean(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "samples_ms": [float(value) for value in samples_ms],
        "count": len(samples_ms),
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


def _family_counts(signature: dict[str, int], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, length in signature.items()
        if int(length) > 0 and len(name) > 1 and name[0] == prefix and name[1:].isdigit()
    ]
    if not orders:
        return ()
    counts = [int(signature.get(f"{prefix}{order}", 0)) for order in range(first, max(orders) + 1)]
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def _profile_count(signature: dict[str, int], name: str) -> int:
    return int(signature.get(name, 0))


def _topology_for_signature(signature: dict[str, int], case: Any, *, build: str) -> Topology:
    m_max = _reference_layout_m_max()
    return Topology(
        h_count=_profile_count(signature, "h"),
        v_count=_profile_count(signature, "v"),
        kappa_count=_profile_count(signature, "k"),
        psin_count=_profile_count(signature, "psin"),
        F_count=_profile_count(signature, "F"),
        c_counts=_family_counts(signature, "c", 0),
        s_counts=_family_counts(signature, "s", 1),
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=int(np.asarray(case.heat_input, dtype=np.float64).size),
        M_max=m_max,
        K_max=max(2, m_max),
        build=build,
    )


def _payload_for_case(
    benchmark: Any,
    case_key: str,
    config_label: str,
    case: Any,
    operator: Any,
) -> str:
    boundary = case.boundary
    source_plan = operator.plan.source_plan
    solver_config = benchmark.CONFIG
    payload = {
        "case_name": f"{case_key}-{config_label.lower()}",
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
        "constraints": {
            "scaled_Ip": float(source_plan.scaled_Ip),
            "fix_rho": float(operator.fix_rho),
        },
        "solver": {
            "method_code": SOLVER_METHOD_POWELL,
            "max_residual": float(solver_config.max_residual),
            "max_evaluations": int(REFERENCE_SOLVER_MAXFEV),
            "accepted_residual_factor": 10.0,
            "accepted_residual_floor": 1.0e-5,
            "initial_policy_code": INITIAL_POLICY_COLD,
            "residual_normalization_code": RESIDUAL_NORMALIZATION_FAST,
            "residual_normalization_floor": float(solver_config.residual_normalization_floor),
            "residual_normalization_max_ratio": float(
                solver_config.residual_normalization_max_ratio
            ),
            "residual_normalization_huber_tau": float(
                solver_config.residual_normalization_huber_tau
            ),
            "residual_normalization_probe_count": int(
                solver_config.residual_normalization_probe_count
            ),
            "residual_normalization_probe_step": float(
                solver_config.residual_normalization_probe_step
            ),
            "residual_normalization_sensitivity_lambda": float(
                solver_config.residual_normalization_sensitivity_lambda
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _measure_veqpy(
    benchmark: Any,
    case: Any,
    grid: Any,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(max(1, warmup)):
        solver = benchmark.Solver(
            operator=benchmark.Operator(grid, case.copy()),
            config=benchmark.CONFIG,
        )
        solver.solve(
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )

    samples_ms: list[float] = []
    nfev: list[int] = []
    njev: list[int] = []
    last_solver = None
    success_all = True
    for _ in range(repeat):
        solver = benchmark.Solver(
            operator=benchmark.Operator(grid, case.copy()),
            config=benchmark.CONFIG,
        )
        started = time.perf_counter_ns()
        solver.solve(
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        samples_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
        if solver.result is None:
            raise RuntimeError("VEQPy solve produced no SolverResult")
        success_all = success_all and bool(solver.result.success)
        nfev.append(int(solver.result.function_evaluations))
        njev.append(int(solver.result.jacobian_evaluations))
        last_solver = solver

    if last_solver is None or last_solver.result is None:
        raise RuntimeError("VEQPy timing loop did not run")
    raw = np.asarray(last_solver.operator.residual_var(last_solver.result.x), dtype=np.float64)
    return {
        "success": bool(success_all),
        "timing": _stats(samples_ms),
        "nfev": _int_stats(nfev),
        "njev": _int_stats(njev),
        "x": np.asarray(last_solver.result.x, dtype=np.float64).copy(),
        "raw": raw,
        "raw_norm": float(np.linalg.norm(raw)),
        "message": str(last_solver.result.message),
    }


def _measure_veqlib(
    reduced_case: ReducedCase,
    *,
    registry: KernelRegistry,
    solver_label: str,
    build: str,
    solver_code: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    topology = _topology_with_build(reduced_case.topology_base, build)
    solver = VEQlibSolver(topology, registry=registry, solver=solver_code)
    build_started = time.perf_counter_ns()
    artifact = solver.build(force=False, dry_run=False)
    build_wall_ms = float(time.perf_counter_ns() - build_started) / 1.0e6
    solver.set_case_json(_payload_with_solver(reduced_case.payload_json, solver_code))
    for _ in range(warmup):
        solver.solve_direct()

    wall_ms: list[float] = []
    inner_ms: list[float] = []
    nfev: list[int] = []
    njev: list[int] = []
    jaccomp: list[int] = []
    success_all = True
    final_result = None
    for _ in range(repeat):
        started = time.perf_counter_ns()
        result = solver.solve_direct()
        wall_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
        inner_ms.append(float(result[0]))
        success_all = success_all and bool(result[1])
        nfev.append(int(result[3]))
        njev.append(int(result[4]))
        jaccomp.append(int(result[6]))
        final_result = result

    if final_result is None:
        raise RuntimeError(f"{solver_label} timing loop did not run")
    x = np.asarray(final_result[11], dtype=np.float64).copy()
    raw = np.asarray(final_result[12], dtype=np.float64).copy()
    return {
        "success": bool(success_all),
        "info": int(final_result[2]),
        "timing": _stats(wall_ms),
        "inner_timing": _stats(inner_ms),
        "nfev": _int_stats(nfev),
        "njev": _int_stats(njev),
        "jacobian_component_evaluations": _int_stats(jaccomp),
        "x": x,
        "raw": raw,
        "raw_norm": float(final_result[9]),
        "scaled_norm": float(final_result[10]),
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "reused": bool(artifact.reused),
            "build": build,
            "build_wall_ms": build_wall_ms,
            "build_elapsed_ms": float(artifact.metadata["build"]["elapsed_ms"]),
        },
    }


def _topology_with_build(topology: Topology, build: str) -> Topology:
    data = topology.to_canonical_dict()
    profiles = data["profiles"]
    grid = data["grid"]
    source = data["source"]
    return Topology(
        h_count=int(profiles["h_count"]),
        v_count=int(profiles["v_count"]),
        kappa_count=int(profiles["kappa_count"]),
        psin_count=int(profiles["psin_count"]),
        F_count=int(profiles["F_count"]),
        c_counts=tuple(int(value) for value in profiles["c_counts"]),
        s_counts=tuple(int(value) for value in profiles["s_counts"]),
        Nr=int(grid["Nr"]),
        Nt=int(grid["Nt"]),
        route=str(source["route"]),
        coordinate=str(source["coordinate"]),
        constraint=str(source["constraint"]),
        nodes=str(source["nodes"]),
        sample_count=int(source["sample_count"]),
        quadrature=str(grid["quadrature"]),
        calculus=str(grid["calculus"]),
        L_max=int(profiles["L_max"]),
        M_max=int(profiles["M_max"]),
        K_max=int(profiles["K_max"]),
        build=build,
    )


def _payload_with_solver(payload_json: str, solver_code: int) -> str:
    payload = json.loads(payload_json)
    payload["solver"]["method_code"] = int(solver_code)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


def _comparison(engine: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | bool]:
    x_err = _max_abs(np.asarray(engine["x"]), np.asarray(baseline["x"]))
    raw_err = _max_abs(np.asarray(engine["raw"]), np.asarray(baseline["raw"]))
    ok = bool(engine.get("success")) and x_err <= VALIDATION_ATOL and raw_err <= VALIDATION_ATOL
    return {
        "x_max_abs": x_err,
        "raw_max_abs": raw_err,
        "within_tolerance": ok,
    }


def _compact(engine: dict[str, Any]) -> dict[str, Any]:
    result = dict(engine)
    result.pop("x", None)
    result.pop("raw", None)
    return result


def _make_reduced_cases(selected: set[str] | None) -> list[ReducedCase]:
    table_signatures = _fig07_module().TABLE05_SELECTED_SIGNATURES
    benchmark = load_pf_benchmark("numba")
    grid = benchmark.Grid(
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    cases: list[ReducedCase] = []
    for case_key in CASE_KEYS:
        reference = build_pf_reference_case(case_key)
        signatures = table_signatures[case_key]
        for config_label, signature in zip(REDUCED_CONFIG_LABELS, signatures, strict=True):
            row_label = f"{case_key}:{config_label.lower()}"
            if selected is not None and row_label not in selected:
                continue
            normalized_signature = {
                str(name): int(length)
                for name, length in sorted(signature.items())
                if int(length) > 0
            }
            case = build_pf_case(benchmark, reference, normalized_signature)
            operator = benchmark.Operator(grid, case)
            topology = _topology_for_signature(normalized_signature, case, build="fastmath")
            payload_json = _payload_for_case(benchmark, case_key, config_label, case, operator)
            x_size = sum(normalized_signature.values())

            def py_measure(
                warmup: int,
                repeat: int,
                case_value: Any = case,
                grid_value: Any = grid,
                benchmark_value: Any = benchmark,
            ) -> Any:
                return _measure_veqpy(
                    benchmark_value,
                    case_value,
                    grid_value,
                    warmup=warmup,
                    repeat=repeat,
                )

            cases.append(
                ReducedCase(
                    case_key=case_key,
                    config_label=config_label,
                    row_label=row_label,
                    signature=normalized_signature,
                    topology_base=topology,
                    payload_json=payload_json,
                    py_measure=py_measure,
                    py_operator=operator,
                    x_size=x_size,
                )
            )
    if selected is not None:
        missing = selected.difference(case.row_label for case in cases)
        if missing:
            raise ValueError(f"unknown reduced case(s): {', '.join(sorted(missing))}")
    return cases


def _case_rows(
    cases: list[ReducedCase],
    *,
    cache_root: Path,
    warmup: int,
    repeat: int,
) -> list[dict[str, Any]]:
    registry = KernelRegistry(cache_root=cache_root, source_dir=CORE_DIR)
    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"[matrix] {case.row_label}: VEQPy hybr baseline", flush=True)
        baseline = case.py_measure(warmup, repeat)
        engines: dict[str, Any] = {"veqpy-hybr": _compact(baseline)}
        comparisons: dict[str, Any] = {
            "veqpy-hybr": {"x_max_abs": 0.0, "raw_max_abs": 0.0, "within_tolerance": True}
        }
        for solver_label, build, solver_code in SOLVERS[1:]:
            assert solver_code is not None
            print(f"[matrix] {case.row_label}: {solver_label}", flush=True)
            try:
                measured = _measure_veqlib(
                    case,
                    registry=registry,
                    solver_label=solver_label,
                    build=build,
                    solver_code=solver_code,
                    warmup=warmup,
                    repeat=repeat,
                )
                comparison = _comparison(measured, baseline)
                engines[solver_label] = _compact(measured)
                comparisons[solver_label] = comparison
            except Exception as exc:  # noqa: BLE001 - benchmark matrix records failures per cell.
                engines[solver_label] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing": _stats([]),
                }
                comparisons[solver_label] = {
                    "x_max_abs": float("inf"),
                    "raw_max_abs": float("inf"),
                    "within_tolerance": False,
                }
        rows.append(
            {
                "case": case.case_key,
                "config": case.config_label,
                "row_label": case.row_label,
                "x_size": case.x_size,
                "signature": case.signature,
                "topology": case.topology_base.to_canonical_dict(),
                "engines": engines,
                "comparison_to_veqpy": comparisons,
            }
        )
    return rows


def _format_speedup(row: dict[str, Any], solver_label: str) -> str:
    if solver_label == "veqpy-hybr":
        return "1.00x"
    engine = row["engines"].get(solver_label, {})
    comparison = row["comparison_to_veqpy"].get(solver_label, {})
    if not engine.get("success") or not comparison.get("within_tolerance"):
        return "-"
    base_ms = float(row["engines"]["veqpy-hybr"]["timing"]["median_ms"])
    cur_ms = float(engine["timing"]["median_ms"])
    if not np.isfinite(base_ms) or not np.isfinite(cur_ms) or cur_ms <= 0.0:
        return "-"
    return f"{base_ms / cur_ms:.2f}x"


def _print_matrix(rows: list[dict[str, Any]], *, repeat: int, warmup: int) -> None:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    print(f"CPU affinity: {affinity}; warmup: {warmup}; repeat: {repeat}")
    headers = [label for label, _build, _code in SOLVERS]
    print("| case | " + " | ".join(headers) + " |")
    print("|---|" + "|".join("---:" for _ in headers) + "|")
    for row in rows:
        cells = [_format_speedup(row, label) for label in headers]
        print(f"| {row['case']}({row['config'].lower()}) | " + " | ".join(cells) + " |")


def _write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--case",
        action="append",
        help="Reduced row such as solovev:low; repeat for multiple rows.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    selected = {value.strip().lower() for value in args.case} if args.case else None
    cache_root = args.cache_root or Path(tempfile.mkdtemp(prefix="veqlib-reduced-matrix-"))
    cases = _make_reduced_cases(selected)
    rows = _case_rows(cases, cache_root=cache_root, warmup=args.warmup, repeat=args.repeat)
    payload = {
        "schema": "veqlib.reduced_solver_matrix.v1",
        "validation_atol": VALIDATION_ATOL,
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cache_root": str(cache_root),
        "layout": {
            "Nr": REFERENCE_LAYOUT_NR,
            "Nt": REFERENCE_LAYOUT_NT,
            "M_max": _reference_layout_m_max(),
            "solver_initial_policy": SOLVER_INITIAL_POLICY,
        },
        "solvers": [label for label, _build, _code in SOLVERS],
        "rows": rows,
    }
    if not args.no_write:
        _write_json(payload, args.output)
        print(f"json: {args.output}")
    _print_matrix(rows, repeat=args.repeat, warmup=args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
