#!/usr/bin/env python3
"""GEQDSK-backed VEQPy route/constraint benchmark matrix.

This pure VEQPy/Numba benchmark is self-contained under ``benchmarks``. It does
not import legacy scripts from ``tests``; a real timing run defaults to the
Solovev GEQDSK and can be pointed at another input via ``--geqdsk``.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from benchmarks._common import (
    CASE_REFERENCE_GFILES,
    REPO_ROOT,
    ROUTE_SHAPE_MATCH_TOL,
    ROUTE_TEST_SOURCE_SAMPLE_COUNT,
    RouteBenchmarkSpec,
    active_profiles_from_coeffs,
    benchmark_route_case_diagnostics,
    coefficients_from_coeffs,
    cpu_affinity,
    extract_shape_x,
    filter_route_specs,
    float_stats,
    format_optional_float,
    format_optional_sci,
    int_stats,
    iter_route_specs,
    make_route_problem,
    nfev_median,
    route_spec_label,
    route_spec_selector,
    runtime_engine_payload,
    runtime_env,
    runtime_progress_phase,
    runtime_status_cell,
    solve_route_reference,
    summarize_runtime_rows,
    timing_median_ms,
    write_json,
)
from veqpy.engine import backend_abi
from veqpy.model import Boundary, Geqdsk, Grid, Problem
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqpy_geqdsk_routes.json"
DEFAULT_SCOPE = "ip-uniform"
DEFAULT_GEQDSK_CASE = "efit"
DEFAULT_GEQDSK = Path(CASE_REFERENCE_GFILES[DEFAULT_GEQDSK_CASE])
ENGINE_LABEL = "veqpy-numba-lm"
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")
DEFAULT_SOLVE_NR = 32
DEFAULT_SOLVE_NT = 32

PROFILE_RMS_TOL = 5.0e-2
GEQDSK_CONFIG = SolverConfig(
    method="lm",
    max_residual=1.0e-7,
    max_evaluations=2000,
    initial_policy="geometric-refined",
    enable_verbose=False,
    enable_fallback=False,
    enable_history=False,
)
GEQDSK_PROFILE_COEFFS: dict[str, list[float]] = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "c3": [0.0] * 5,
    "c4": [0.0] * 5,
    "c5": [0.0] * 5,
    "c6": [0.0] * 5,
    "c7": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
}


def _geqdsk_reference(
    geqdsk_path: str,
    *,
    boundary_fit_m: int,
    boundary_fit_n: int,
    boundary_maxtol: float,
    reference_nr: int,
    reference_nt: int,
    max_evaluations: int,
):
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk(str(geqdsk_path))
    boundary = Boundary.from_geqdsk(
        geqdsk,
        M=int(boundary_fit_m),
        N=int(boundary_fit_n),
        maxtol=float(boundary_maxtol),
    )
    grid = Grid(Nr=int(reference_nr), Nt=int(reference_nt), quadrature_scheme="legendre")
    config = SolverConfig(
        method=GEQDSK_CONFIG.method,
        max_residual=GEQDSK_CONFIG.max_residual,
        max_evaluations=int(max_evaluations),
        initial_policy=GEQDSK_CONFIG.initial_policy,
        enable_fallback=False,
        enable_verbose=False,
        enable_history=False,
    )
    problem = Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(GEQDSK_PROFILE_COEFFS),
        boundary=boundary,
        heat_input=np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=float(geqdsk.Ip),
    )
    return solve_route_reference(problem, grid, config, GEQDSK_PROFILE_COEFFS)


def _geqdsk_profile_coeffs_for_case(
    spec: RouteBenchmarkSpec,
    initial_coeffs: dict[str, np.ndarray] | None = None,
) -> dict[str, list[float]]:
    coeffs = {name: list(values) for name, values in GEQDSK_PROFILE_COEFFS.items()}
    route_key = (str(spec.mode).upper(), str(spec.coordinate).lower(), str(spec.input_kind).lower())
    if route_key not in backend_abi.PROFILE_OWNED_PSIN_ROUTE_KEYS:
        coeffs.pop("psin", None)
    if spec.mode == "PJ2":
        coeffs.setdefault("F", [0.0] * 5)
    if initial_coeffs:
        for name in tuple(coeffs):
            values = initial_coeffs.get(name)
            if values is not None and len(values) == len(coeffs[name]):
                coeffs[name] = [float(value) for value in values]
    return coeffs


def _console() -> Console:
    return Console(highlight=False)


def _iter_route_specs(*, scope: str) -> tuple[RouteBenchmarkSpec, ...]:
    return iter_route_specs(scope=scope, default_scope=DEFAULT_SCOPE, allow_grid=False)


def _route_config(max_evaluations: int) -> SolverConfig:
    return SolverConfig(
        method=GEQDSK_CONFIG.method,
        max_residual=GEQDSK_CONFIG.max_residual,
        max_evaluations=int(max_evaluations),
        initial_policy=None,
        enable_fallback=False,
        enable_verbose=False,
        enable_history=False,
    )


def _measure_spec(
    reference: Any,
    spec: RouteBenchmarkSpec,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    coeffs = _geqdsk_profile_coeffs_for_case(spec, reference.profile_coeffs)
    solve_grid = Grid(
        Nr=int(args.solve_nr),
        Nt=int(args.solve_nt),
        quadrature_scheme="legendre",
    )
    case = make_route_problem(
        spec,
        reference,
        coeffs,
        grid=solve_grid,
        sample_count=int(args.source_sample_count),
    )
    config = _route_config(int(args.max_evaluations))
    operator = Operator(solve_grid, case.copy())
    x0 = operator.pack_coefficients(coefficients_from_coeffs(coeffs))

    for _ in range(max(0, int(args.warmup))):
        solver = Solver(operator=Operator(solve_grid, case.copy()), config=config)
        solver.solve(
            x0=x0,
            method=config.method,
            max_residual=config.max_residual,
            max_evaluations=config.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )

    wall_ms: list[float] = []
    nfev: list[int] = []
    njev: list[int] = []
    iterations: list[int] = []
    success: list[bool] = []
    last_solver = None
    for _ in range(int(args.repeat)):
        solver = Solver(operator=Operator(solve_grid, case.copy()), config=config)
        started = time.perf_counter_ns()
        solver.solve(
            x0=x0,
            method=config.method,
            max_residual=config.max_residual,
            max_evaluations=config.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        wall_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
        result = solver.result
        if result is None:
            raise RuntimeError(f"{route_spec_label(spec)} VEQPy solve produced no SolverResult")
        success.append(bool(result.success))
        nfev.append(int(result.function_evaluations))
        njev.append(int(result.jacobian_evaluations))
        iterations.append(int(result.iterations))
        last_solver = solver

    if last_solver is None or last_solver.result is None:
        raise RuntimeError("GEQDSK route timing loop did not run")

    result = last_solver.result
    equilibrium = last_solver.build_equilibrium()
    shape_x = extract_shape_x(last_solver.operator.problem.active_profiles, result.x)
    diagnostics = benchmark_route_case_diagnostics(reference, equilibrium, shape_x)
    shape_ok = float(diagnostics["shape_error"]) <= ROUTE_SHAPE_MATCH_TOL
    psi_ok = float(diagnostics["psi_r_rel_rms_error"]) <= PROFILE_RMS_TOL
    ff_ok = float(diagnostics["ff_psi_rel_rms_error"]) <= PROFILE_RMS_TOL
    passed = all(success) and shape_ok and psi_ok and ff_ok
    failure_reason = None
    if not all(success):
        failure_reason = "solver_failed"
    elif not shape_ok:
        failure_reason = "shape_tolerance_failed"
    elif not psi_ok:
        failure_reason = "psi_r_tolerance_failed"
    elif not ff_ok:
        failure_reason = "ff_psi_tolerance_failed"

    runtime: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "x_size": int(result.x.size),
        "engine": ENGINE_LABEL,
        "engines": {
            ENGINE_LABEL: {
                "success_all": all(success),
                "timing": float_stats(wall_ms),
                "nfev": int_stats(nfev),
                "njev": int_stats(njev),
                "iterations": int_stats(iterations),
                "residual_norm_final": float(result.residual_norm_final),
                "message": str(result.message),
            }
        },
        "diagnostics": {
            "shape_match_tol": ROUTE_SHAPE_MATCH_TOL,
            "profile_rms_tol": PROFILE_RMS_TOL,
            **diagnostics,
        },
    }
    if failure_reason is not None:
        runtime["failure_reason"] = failure_reason
    return runtime


def _plan_row(geqdsk_path: Path | None, spec: RouteBenchmarkSpec) -> dict[str, Any]:
    return {
        "geqdsk": None if geqdsk_path is None else str(geqdsk_path),
        "case": route_spec_label(spec),
        "selector": route_spec_selector(spec),
        "route": str(spec.mode),
        "coordinate": str(spec.coordinate),
        "nodes": str(spec.input_kind),
        "constraint": str(spec.constraint),
        "runtime": {"status": "not_requested"},
    }


def _progress_context(console: Console, *, quiet: bool) -> Any:
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn("[dim]{task.fields[current]:<24.24}[/]"),
        BarColumn(bar_width=48, complete_style="cyan", finished_style="green", pulse_style="cyan"),
        MofNCompleteColumn(),
        TextColumn("{task.fields[phase]:>8}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _print_config_tree(
    console: Console,
    *,
    geqdsk_path: Path | None,
    scope: str,
    specs: tuple[RouteBenchmarkSpec, ...],
    repeat: int,
    warmup: int,
    no_run: bool,
) -> None:
    console.print(Text("[config]", style="bold cyan"))
    lines = (
        f"scope: [green]{scope}[/]",
        f"geqdsk: [green]{geqdsk_path if geqdsk_path is not None else 'not configured'}[/]",
        f"route cases: [green]{len(specs)}[/]",
        f"engine: [green]{ENGINE_LABEL}[/]",
        f"mode: [green]{'plan-only' if no_run else 'run'}[/]",
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


def _print_summary(console: Console, summary: dict[str, int]) -> None:
    counts = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    counts.add_column("summary")
    counts.add_column("count", justify="right")
    for key in ("total", "runtime_passed", "runtime_failed", "runtime_not_requested"):
        counts.add_row(key.replace("_", " "), str(summary[key]))
    console.print(counts)


def _print_failures(console: Console, rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row["runtime"]["status"] == "failed"]
    if not failed:
        return
    console.print()
    tree = Tree(Text("[failures]", style="bold red"))
    for row in failed:
        runtime = row["runtime"]
        tree.add(f"{row.get('case', 'n/a')}: {runtime.get('failure_reason', 'failed')}")
    console.print(tree)
    console.print()


def _print_timing_table(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("case", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("x", justify="right")
    table.add_column(Text("Numba (ms)"), justify="right")
    table.add_column("nfev", justify="right")
    table.add_column("residual", justify="right")
    table.add_column("shape", justify="right")
    table.add_column("psi_r", justify="right")
    table.add_column("FF_psi", justify="right")
    for row in rows:
        runtime = row["runtime"]
        engine = runtime_engine_payload(runtime, ENGINE_LABEL)
        diagnostics = runtime.get("diagnostics") if isinstance(runtime, dict) else None
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        table.add_row(
            str(row.get("case", "n/a")),
            runtime_status_cell(runtime["status"]),
            str(runtime.get("x_size", "n/a")),
            format_optional_float(timing_median_ms(engine)),
            nfev_median(engine),
            format_optional_sci(None if engine is None else engine.get("residual_norm_final")),
            format_optional_sci(diagnostics.get("shape_error")),
            format_optional_sci(diagnostics.get("psi_r_rel_rms_error")),
            format_optional_sci(diagnostics.get("ff_psi_rel_rms_error")),
        )
    console.print(table)


def _add_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-sample-count", type=int, default=ROUTE_TEST_SOURCE_SAMPLE_COUNT)
    parser.add_argument("--reference-nr", type=int, default=64)
    parser.add_argument("--reference-nt", type=int, default=32)
    parser.add_argument("--solve-nr", type=int, default=DEFAULT_SOLVE_NR)
    parser.add_argument("--solve-nt", type=int, default=DEFAULT_SOLVE_NT)
    parser.add_argument("--boundary-fit-m", type=int, default=10)
    parser.add_argument("--boundary-fit-n", type=int, default=10)
    parser.add_argument("--boundary-maxtol", type=float, default=1.0)
    parser.add_argument("--max-evaluations", type=int, default=2000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=(DEFAULT_SCOPE, "uniform"), default=DEFAULT_SCOPE)
    parser.add_argument("--case", action="append", help="Case name or route:coord:nodes:constraint")
    parser.add_argument(
        "--geqdsk",
        type=Path,
        default=DEFAULT_GEQDSK,
        help=f"GEQDSK path for real runs; defaults to {DEFAULT_GEQDSK_CASE}.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    _add_geometry_args(parser)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = _console()
    specs = filter_route_specs(
        _iter_route_specs(scope=args.scope),
        set(args.case) if args.case else None,
    )
    geqdsk_path = args.geqdsk.expanduser().resolve()
    effective_no_run = bool(args.no_run)
    rows = [_plan_row(geqdsk_path, spec) for spec in specs]

    if not args.quiet_progress:
        _print_config_tree(
            console,
            geqdsk_path=geqdsk_path,
            scope=str(args.scope),
            specs=specs,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            no_run=effective_no_run,
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))

    if not effective_no_run:
        reference = _geqdsk_reference(
            str(geqdsk_path),
            boundary_fit_m=int(args.boundary_fit_m),
            boundary_fit_n=int(args.boundary_fit_n),
            boundary_maxtol=float(args.boundary_maxtol),
            reference_nr=int(args.reference_nr),
            reference_nt=int(args.reference_nt),
            max_evaluations=int(args.max_evaluations),
        )
        with _progress_context(console, quiet=args.quiet_progress) as progress:
            task_id = None
            if progress is not None:
                task_id = progress.add_task(
                    "veqpy-geqdsk-routes",
                    total=len(specs),
                    current="-",
                    phase="[cyan]run[/]",
                )
            for index, spec in enumerate(specs):
                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        current=route_spec_selector(spec),
                        phase="[cyan]run[/]",
                    )
                try:
                    rows[index]["runtime"] = _measure_spec(reference, spec, args=args)
                except Exception as exc:
                    rows[index]["runtime"] = {
                        "status": "failed",
                        "failure_reason": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        phase=runtime_progress_phase(rows[index]["runtime"]["status"]),
                    )
                    progress.advance(task_id)

    summary = summarize_runtime_rows(rows)
    payload = {
        "schema": "veqpy.geqdsk_routes.v1",
        "scope": str(args.scope),
        "geqdsk": None if geqdsk_path is None else str(geqdsk_path),
        "case_count": len(rows),
        "engine": ENGINE_LABEL,
        "run_mode": "plan-only" if effective_no_run else "run",
        "default_geqdsk_case": DEFAULT_GEQDSK_CASE,
        "skip_reason": "no_run" if effective_no_run else None,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "timing_note": (
            "wall time around Solver.solve(...) only; Operator construction and x0 packing "
            "are excluded to match benchmarks.veqlib_geqdsk_pareto's Numba column"
        ),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "layout": {
            "reference_grid": {"Nr": int(args.reference_nr), "Nt": int(args.reference_nt)},
            "solve_grid": {
                "Nr": int(args.solve_nr),
                "Nt": int(args.solve_nt),
                "quadrature_scheme": "legendre",
            },
            "source_sample_count": int(args.source_sample_count),
            "solver": {
                "method": GEQDSK_CONFIG.method,
                "max_residual": float(GEQDSK_CONFIG.max_residual),
                "max_evaluations": int(args.max_evaluations),
                "canonical_initial_policy": GEQDSK_CONFIG.initial_policy,
                "route_initial_policy": None,
            },
            "equivalence_thresholds": {
                "shape_error": ROUTE_SHAPE_MATCH_TOL,
                "psi_r_rel_rms_error": PROFILE_RMS_TOL,
                "ff_psi_rel_rms_error": PROFILE_RMS_TOL,
            },
        },
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        console.print()
        _print_outputs_tree(console, {"json": args.output})
    _print_summary(console, summary)
    _print_failures(console, rows)
    _print_timing_table(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
