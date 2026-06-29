#!/usr/bin/env python3
"""GEQDSK Low/Medium/High/Ref benchmark for VEQlib against VEQPy.

This is the result-and-speed comparison benchmark.  It evaluates the three
GEQDSK-backed PF/psin/uniform/Ip cases (solovev, chease, efit) over the four
configuration labels used by the manuscript data: Low, Medium, High, and Ref.
The VEQlib path uses the current typed facade runtime with a ``fastmath`` build
by default; JSON payloads are not used as runtime input.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from veqlib.benchmarks._common import (
    CORE_DIR,
    SCRIPTS_DIR,
    cpu_affinity,
    family_counts,
    float_stats,
    int_stats,
    max_abs,
    measure_native_solver,
    profile_count,
    runtime_env,
    write_json,
)
from veqlib.facade import (
    KernelBoundary,
    KernelBuild,
    KernelInput,
    KernelRegistry,
    KernelSolve,
    KernelTopology,
    VEQlibSolver,
    default_kernel_cache_root,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    CASE_KEYS,
    CONFIG_LABELS,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REFERENCE_SOLVER_MAXFEV,
    SOLVER_INITIAL_POLICY,
    build_pf_case,
    build_pf_reference_case,
    load_pf_benchmark,
    load_reduced_equilibrium_manifest,
    load_reference_equilibrium_manifest,
    manifest_entry,
    reference_manifest_entry,
    signature_from_metadata,
)

DEFAULT_OUTPUT = Path("/tmp/veqlib_geqdsk_configs.json")
VALIDATION_ATOL = 1.0e-6
Topology = KernelTopology


@dataclass(frozen=True, slots=True)
class GeqdskConfigCase:
    case_key: str
    config_label: str
    row_label: str
    signature: dict[str, int]
    topology: Topology
    kernel_input: KernelInput
    kernel_solve: KernelSolve
    py_operator: Any
    py_measure: Any
    x_size: int


def _coeffs_from_signature(signature: dict[str, int]) -> dict[str, list[float] | None]:
    return {name: [0.0] * int(length) for name, length in signature.items() if int(length) > 0}


def _topology_for_case(signature: dict[str, int], case: Any, *, build: str, grid: Any) -> Topology:
    coeffs = _coeffs_from_signature(signature)
    boundary = case.boundary
    c_offsets = np.asarray(boundary.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(boundary.s_offsets, dtype=np.float64)
    m_max = max(
        int(c_offsets.size) - 1 if c_offsets.size else 0,
        int(s_offsets.size) - 1 if s_offsets.size else 0,
        1,
    )
    topology = KernelTopology(
        h_count=profile_count(coeffs, "h"),
        v_count=profile_count(coeffs, "v"),
        kappa_count=profile_count(coeffs, "k"),
        psin_count=profile_count(coeffs, "psin"),
        F_count=profile_count(coeffs, "F"),
        c_counts=family_counts(coeffs, "c", 0),
        s_counts=family_counts(coeffs, "s", 1),
        Nr=int(grid.Nr),
        Nt=int(grid.Nt),
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=int(np.asarray(case.heat_input, dtype=np.float64).size),
        M_max=m_max,
        K_max=max(2, m_max),
    )
    return topology.with_build(KernelBuild(build=build, layout="degree"))


def _kernel_boundary_from_case(case: Any) -> KernelBoundary:
    boundary = case.boundary
    return KernelBoundary(
        a=float(boundary.a),
        R0=float(boundary.R0),
        Z0=float(boundary.Z0),
        B0=float(boundary.B0),
        ka=float(boundary.ka),
        c_offsets=np.asarray(boundary.c_offsets, dtype=np.float64),
        s_offsets=np.asarray(boundary.s_offsets, dtype=np.float64),
    )


def _kernel_input_from_operator(
    case_key: str,
    config_label: str,
    case: Any,
    operator: Any,
) -> KernelInput:
    source_plan = operator.plan.source_plan
    return KernelInput(
        boundary=_kernel_boundary_from_case(case),
        scaled_heat=np.asarray(source_plan.scaled_heat, dtype=np.float64),
        scaled_current=np.asarray(source_plan.scaled_current, dtype=np.float64),
        scaled_Ip=float(source_plan.scaled_Ip),
        beta=float(source_plan.beta),
        fix_rho=float(operator.fix_rho),
        case_name=f"{case_key}-{config_label.lower()}",
    )


def _kernel_solve_from_config(config: Any, *, x_size: int) -> KernelSolve:
    return KernelSolve(
        method="powell",
        max_residual=float(config.max_residual),
        max_evaluations=int(REFERENCE_SOLVER_MAXFEV),
        initial="cold",
        norm="fast",
        residual_normalization_floor=float(config.residual_normalization_floor),
        residual_normalization_max_ratio=float(config.residual_normalization_max_ratio),
        residual_normalization_huber_tau=float(config.residual_normalization_huber_tau),
        residual_normalization_probe_count=int(config.residual_normalization_probe_count),
        residual_normalization_probe_step=float(config.residual_normalization_probe_step),
        residual_normalization_sensitivity_lambda=float(
            config.residual_normalization_sensitivity_lambda
        ),
    )


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

    wall_ms: list[float] = []
    nfev: list[int] = []
    njev: list[int] = []
    success: list[bool] = []
    last_solver = None
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
        wall_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
        if solver.result is None:
            raise RuntimeError("VEQPy solve produced no SolverResult")
        success.append(bool(solver.result.success))
        nfev.append(int(solver.result.function_evaluations))
        njev.append(int(solver.result.jacobian_evaluations))
        last_solver = solver

    if last_solver is None or last_solver.result is None:
        raise RuntimeError("VEQPy timing loop did not run")
    raw = np.asarray(last_solver.operator.residual_var(last_solver.result.x), dtype=np.float64)
    return {
        "success_all": all(success),
        "timing": float_stats(wall_ms),
        "nfev": int_stats(nfev),
        "njev": int_stats(njev),
        "x": np.asarray(last_solver.result.x, dtype=np.float64).copy(),
        "raw": raw,
        "raw_norm": float(np.linalg.norm(raw)),
        "message": str(last_solver.result.message),
    }


def _case_from_signature(
    benchmark: Any,
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    build: str,
) -> GeqdskConfigCase:
    reference = build_pf_reference_case(case_key)
    grid = benchmark.Grid(
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    case = build_pf_case(benchmark, reference, signature)
    operator = benchmark.Operator(grid, case)
    topology = _topology_for_case(signature, case, build=build, grid=grid)
    kernel_input = _kernel_input_from_operator(case_key, config_label, case, operator)
    x_size = int(topology.packed_size())
    kernel_solve = _kernel_solve_from_config(benchmark.CONFIG, x_size=x_size)

    def measure_py(*, warmup: int, repeat: int) -> dict[str, Any]:
        return _measure_veqpy(benchmark, case, grid, warmup=warmup, repeat=repeat)

    return GeqdskConfigCase(
        case_key=case_key,
        config_label=config_label,
        row_label=f"{case_key}:{config_label.lower()}",
        signature=dict(signature),
        topology=topology,
        kernel_input=kernel_input,
        kernel_solve=kernel_solve,
        py_operator=operator,
        py_measure=measure_py,
        x_size=x_size,
    )


def _make_cases(
    *,
    build: str,
    selected_cases: set[str] | None,
    selected_configs: set[str] | None,
) -> list[GeqdskConfigCase]:
    benchmark = load_pf_benchmark("numba")
    reduced_manifest = load_reduced_equilibrium_manifest()
    reference_manifest = load_reference_equilibrium_manifest()
    rows: list[GeqdskConfigCase] = []
    for case_key in CASE_KEYS:
        if selected_cases is not None and case_key not in selected_cases:
            continue
        for config_label in CONFIG_LABELS:
            if selected_configs is not None and config_label.lower() not in selected_configs:
                continue
            entry = (
                reference_manifest_entry(reference_manifest, case_key)
                if config_label == "Ref"
                else manifest_entry(reduced_manifest, case_key, config_label)
            )
            signature = signature_from_metadata(entry)
            rows.append(
                _case_from_signature(
                    benchmark,
                    case_key=case_key,
                    config_label=config_label,
                    signature=signature,
                    build=build,
                )
            )
    return rows


def _measure_veqlib(
    case: GeqdskConfigCase,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    solver = VEQlibSolver(case.topology, registry=registry, solver="powell")
    build_start = time.perf_counter_ns()
    artifact = solver.build(force=False, dry_run=False)
    build_wall_ms = float(time.perf_counter_ns() - build_start) / 1.0e6

    def configure() -> None:
        solver.set_kernel_runtime(
            *case.kernel_input.runtime_args(),
            *case.kernel_solve.runtime_args(x_size=case.x_size),
        )

    timing = measure_native_solver(solver, configure, warmup=warmup, repeat=repeat)
    native_x = timing.result.x
    native_raw = timing.result.raw
    py_raw_at_native = np.asarray(case.py_operator.residual_var(native_x), dtype=np.float64)
    payload = timing.compact()
    payload.update(
        {
            "artifact": {
                "artifact_id": artifact.artifact_id,
                "reused": bool(artifact.reused),
                "build_wall_ms": float(build_wall_ms),
                "build_elapsed_ms": float(artifact.metadata["build"]["elapsed_ms"]),
            },
            "raw_norm": float(np.linalg.norm(native_raw)),
            "py_raw_at_veqlib_x_norm": float(np.linalg.norm(py_raw_at_native)),
        }
    )
    return payload


def _compact_py(engine: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_all": bool(engine["success_all"]),
        "timing": engine["timing"],
        "nfev": engine["nfev"],
        "njev": engine["njev"],
        "raw_norm": float(engine["raw_norm"]),
        "message": str(engine["message"]),
        "x": np.asarray(engine["x"], dtype=np.float64).tolist(),
        "raw": np.asarray(engine["raw"], dtype=np.float64).tolist(),
    }


def _compare(cxx: dict[str, Any], py: dict[str, Any]) -> dict[str, Any]:
    x_diff = max_abs(cxx["x"], py["x"])
    raw_diff = max_abs(cxx["raw"], py["raw"])
    return {
        "x_max_abs": x_diff,
        "raw_max_abs": raw_diff,
        "within_atol": bool(x_diff <= VALIDATION_ATOL and raw_diff <= VALIDATION_ATOL),
    }


def _row(
    case: GeqdskConfigCase,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    print(f"[geqdsk] {case.row_label}: VEQPy", flush=True)
    py = case.py_measure(warmup=warmup, repeat=repeat)
    print(f"[geqdsk] {case.row_label}: VEQlib", flush=True)
    cxx = _measure_veqlib(case, registry=registry, warmup=warmup, repeat=repeat)
    compare = _compare(cxx, py)
    passed = cxx["success_all"] and py["success_all"] and compare["within_atol"]
    return {
        "status": "passed" if passed else "failed",
        "case": case.case_key,
        "config": case.config_label,
        "row": case.row_label,
        "x_size": case.x_size,
        "signature": case.signature,
        "topology": {
            "key": case.topology.key,
            "grid": {"Nr": case.topology.Nr, "Nt": case.topology.Nt},
            "sample_count": case.topology.sample_count,
            "M_max": case.topology.M_max,
        },
        "engines": {
            "veqlib-fastmath-powell": cxx,
            "veqpy-numba-hybr": _compact_py(py),
        },
        "closeness_to_numba": compare,
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "| status | case | config | x_size | VEQlib median ms | VEQPy median ms | "
        "speedup | x_diff | raw_diff |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        cxx = row["engines"]["veqlib-fastmath-powell"]
        py = row["engines"]["veqpy-numba-hybr"]
        cxx_ms = float(cxx["timing"]["median_ms"])
        py_ms = float(py["timing"]["median_ms"])
        compare = row["closeness_to_numba"]
        print(
            f"| {row['status']} | {row['case']} | {row['config']} | {row['x_size']} | "
            f"{cxx_ms:.6f} | {py_ms:.6f} | {py_ms / cxx_ms if cxx_ms > 0 else float('inf'):.3f}x | "
            f"{compare['x_max_abs']:.3e} | {compare['raw_max_abs']:.3e} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--config", action="append", choices=CONFIG_LABELS)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    selected_cases = set(args.case) if args.case else None
    selected_configs = {value.lower() for value in args.config} if args.config else None
    cases = _make_cases(
        build=args.build,
        selected_cases=selected_cases,
        selected_configs=selected_configs,
    )
    cache_root = args.cache_root or default_kernel_cache_root()
    registry = KernelRegistry(cache_root=cache_root, source_dir=args.source_dir.resolve())
    rows = [_row(case, registry=registry, warmup=args.warmup, repeat=args.repeat) for case in cases]
    payload = {
        "schema": "veqlib.geqdsk_configs.v1",
        "build": str(args.build),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "validation_atol": VALIDATION_ATOL,
        "cache_root": str(cache_root),
        "source_dir": str(args.source_dir.resolve()),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "layout": {
            "Nr": REFERENCE_LAYOUT_NR,
            "Nt": REFERENCE_LAYOUT_NT,
            "solver_initial_policy": SOLVER_INITIAL_POLICY,
        },
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        print(f"json: {args.output}")
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
