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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from benchmarks._common import (
    CASE_KEYS,
    CONFIG_LABELS,
    CORE_DIR,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REFERENCE_SOLVER_MAXFEV,
    REPO_ROOT,
    build_pf_case,
    build_pf_reference_case,
    cpu_affinity,
    family_counts,
    float_stats,
    int_stats,
    load_pf_benchmark,
    load_reduced_equilibrium_manifest,
    load_reference_equilibrium_manifest,
    manifest_entry,
    max_abs,
    measure_native_solver,
    profile_count,
    reference_manifest_entry,
    runtime_env,
    signature_from_metadata,
    write_json,
)
from benchmarks._common import (
    SOLVER_INITIAL_POLICY as REFERENCE_SOLVER_INITIAL_POLICY,
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

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqlib_geqdsk.json"
VALIDATION_ATOL = 1.0e-6
NATIVE_SOLVER_METHOD = "powell"
NATIVE_SOLVER_INITIAL_POLICY = "cold"
NATIVE_SOLVER_CONTINUATION_POLICY = "cold"
NATIVE_SOLVER_NORMALIZATION = "fast"
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")
Topology = KernelTopology


def _console() -> Console:
    return Console(highlight=False)


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
        method=NATIVE_SOLVER_METHOD,
        max_residual=float(config.max_residual),
        max_evaluations=int(REFERENCE_SOLVER_MAXFEV),
        initial=NATIVE_SOLVER_INITIAL_POLICY,
        continuation=NATIVE_SOLVER_CONTINUATION_POLICY,
        norm=NATIVE_SOLVER_NORMALIZATION,
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
    for _ in range(warmup):
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
    solver = VEQlibSolver(case.topology, registry=registry, solver=NATIVE_SOLVER_METHOD)
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
    progress: bool,
) -> dict[str, Any]:
    if progress:
        print(f"[geqdsk] {case.row_label}: VEQPy", flush=True)
    py = case.py_measure(warmup=warmup, repeat=repeat)
    if progress:
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


def _progress_context(console: Console, *, quiet: bool) -> Any:
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn("[dim]{task.fields[current]:<24.24}[/]"),
        BarColumn(
            bar_width=48,
            complete_style="cyan",
            finished_style="green",
            pulse_style="cyan",
        ),
        MofNCompleteColumn(),
        TextColumn("{task.fields[phase]:>8}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _print_config_tree(
    console: Console,
    *,
    cases: list[GeqdskConfigCase],
    build: str,
    repeat: int,
    warmup: int,
) -> None:
    console.print(Text("[config]", style="bold cyan"))
    lines = (
        f"cases: [green]{len(cases)}[/]",
        f"build: [green]{build}[/]",
        f"initial: [green]{NATIVE_SOLVER_INITIAL_POLICY}[/]",
        f"continue: [green]{NATIVE_SOLVER_CONTINUATION_POLICY}[/]",
        f"norm: [green]{NATIVE_SOLVER_NORMALIZATION}[/]",
        f"warmup: [green]{warmup}[/]",
        f"repeat: [green]{repeat}[/]",
    )
    for index, line in enumerate(lines):
        branch = "└──" if index == len(lines) - 1 else "├──"
        console.print(f"  {branch} {line}")


def _print_outputs_tree(console: Console, outputs: dict[str, Path]) -> None:
    if not outputs:
        return
    console.print(Text("[outputs]", style="bold cyan"))
    paths: list[Path] = []
    for path in outputs.values():
        try:
            display_path = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        paths.append(display_path)
    for index, path in enumerate(paths):
        branch = "└──" if index == len(paths) - 1 else "├──"
        console.print(f"  {branch} [green]{path}[/]")


def _status_cell(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    return text


def _progress_phase(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    return "[dim]done[/]"


def _format_speedup(py_ms: float, cxx_ms: float) -> str:
    if cxx_ms <= 0.0:
        return "inf"
    return f"{py_ms / cxx_ms:.3f}x"


def _print_summary(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(
        box=REPORT_TABLE_BOX,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("case", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("config", no_wrap=True)
    table.add_column("x", justify="right")
    table.add_column(Text("Cxx (ms)"), justify="right")
    table.add_column(Text("Numba (ms)"), justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("diff", justify="right")
    for row in rows:
        cxx = row["engines"]["veqlib-fastmath-powell"]
        py = row["engines"]["veqpy-numba-hybr"]
        cxx_ms = float(cxx["timing"]["median_ms"])
        py_ms = float(py["timing"]["median_ms"])
        compare = row["closeness_to_numba"]
        table.add_row(
            str(row["case"]),
            _status_cell(row["status"]),
            str(row["config"]),
            str(row["x_size"]),
            f"{cxx_ms:.6f}",
            f"{py_ms:.6f}",
            _format_speedup(py_ms, cxx_ms),
            f"{float(compare['x_max_abs']):.2e}",
        )
    console.print(table)


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
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
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
    console = _console()
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        _print_config_tree(
            console,
            cases=cases,
            build=str(args.build),
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with _progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "geqdsk",
                total=len(cases),
                current="-",
                phase="[cyan]run[/]",
            )
        for case in cases:
            if progress is not None and task_id is not None:
                progress.update(task_id, current=case.row_label, phase="[cyan]run[/]")
            row = _row(
                case,
                registry=registry,
                warmup=args.warmup,
                repeat=args.repeat,
                progress=False,
            )
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=_progress_phase(row["status"]))
                progress.advance(task_id)
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
        "native_solver_policy": {
            "method": NATIVE_SOLVER_METHOD,
            "initial": NATIVE_SOLVER_INITIAL_POLICY,
            "continue": NATIVE_SOLVER_CONTINUATION_POLICY,
            "norm": NATIVE_SOLVER_NORMALIZATION,
        },
        "layout": {
            "Nr": REFERENCE_LAYOUT_NR,
            "Nt": REFERENCE_LAYOUT_NT,
            "solver_initial_policy": REFERENCE_SOLVER_INITIAL_POLICY,
        },
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        console.print()
        _print_outputs_tree(console, {"json": args.output})
    _print_summary(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
