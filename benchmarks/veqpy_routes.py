#!/usr/bin/env python3
"""VEQPy route/constraint benchmark matrix.

This is the pure Python/Numba counterpart to ``benchmarks.veqlib_routes``. It is
self-contained under ``benchmarks`` and calls only the public VEQPy runtime; it
does not import legacy scripts from ``tests``.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from functools import lru_cache
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
    MU0,
    REPO_ROOT,
    ROUTE_SHAPE_MATCH_TOL,
    ROUTE_TEST_GRID,
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
    grid_payload,
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
from veqpy.model import Boundary, Grid, Problem
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqpy_routes.json"
DEFAULT_SCOPE = "ip-uniform"
ENGINE_LABEL = "veqpy-numba-hybr"
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")

REFERENCE_SOURCE_SAMPLE_COUNT = 51
REFERENCE_GRID = Grid(Nr=64, Nt=32, quadrature_scheme="legendre")
SYNTHETIC_CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=False,
    enable_history=False,
)
SYNTHETIC_REFERENCE_CONFIG = SolverConfig(
    method=SYNTHETIC_CONFIG.method,
    max_residual=SYNTHETIC_CONFIG.max_residual,
    max_evaluations=SYNTHETIC_CONFIG.max_evaluations,
    initial_policy=None,
    enable_verbose=False,
    enable_history=False,
)

BASE_COEFFS: dict[str, list[float]] = {
    "h": [0.0] * 3,
    "k": [0.0] * 6,
    "s1": [0.0] * 3,
}
PSIN_ROBUST_COEFFS: dict[str, list[float]] = {
    **BASE_COEFFS,
    "psin": [0.0] * 6,
}
SYNTHETIC_BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)
SYNTHETIC_IP = 3.0e6


def _pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def _synthetic_reference_problem() -> Problem:
    rho_src = np.linspace(0.0, 1.0, REFERENCE_SOURCE_SAMPLE_COUNT)
    psin_src = rho_src * rho_src
    ffn_psin_src, pn_psin_src = _pf_reference_profiles(psin_src)
    ffn_r_src = ffn_psin_src * (2.0 * rho_src)
    pn_r_src = pn_psin_src * (2.0 * rho_src)
    return Problem(
        route="PF",
        coordinate="rho",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(BASE_COEFFS),
        boundary=SYNTHETIC_BOUNDARY,
        heat_input=pn_r_src / MU0,
        current_input=ffn_r_src,
        Ip=SYNTHETIC_IP,
    )


@lru_cache(maxsize=1)
def _synthetic_reference():
    return solve_route_reference(
        _synthetic_reference_problem(),
        REFERENCE_GRID,
        SYNTHETIC_REFERENCE_CONFIG,
        BASE_COEFFS,
    )


def _synthetic_profile_coeffs_for_case(spec: RouteBenchmarkSpec) -> dict[str, list[float]]:
    route_key = (str(spec.mode).upper(), str(spec.coordinate).lower(), str(spec.input_kind).lower())
    if route_key in backend_abi.PROFILE_OWNED_PSIN_ROUTE_KEYS:
        coeffs = {name: list(values) for name, values in PSIN_ROBUST_COEFFS.items()}
    else:
        coeffs = {name: list(values) for name, values in BASE_COEFFS.items()}
    if spec.mode == "PJ2":
        coeffs["F"] = [0.0] * 6
    return coeffs


def _console() -> Console:
    return Console(highlight=False)


def _iter_route_specs(*, scope: str) -> tuple[RouteBenchmarkSpec, ...]:
    return iter_route_specs(scope=scope, default_scope=DEFAULT_SCOPE, allow_grid=True)


def _measure_case(spec: RouteBenchmarkSpec, *, warmup: int, repeat: int) -> dict[str, Any]:
    reference = _synthetic_reference()
    coeffs = _synthetic_profile_coeffs_for_case(spec)
    case = make_route_problem(
        spec,
        reference,
        coeffs,
        grid=ROUTE_TEST_GRID,
        sample_count=ROUTE_TEST_SOURCE_SAMPLE_COUNT,
    )
    operator = Operator(ROUTE_TEST_GRID, case.copy())
    x0 = operator.pack_coefficients(coefficients_from_coeffs(coeffs))

    for _ in range(max(0, warmup)):
        solver = Solver(operator=Operator(ROUTE_TEST_GRID, case.copy()), config=SYNTHETIC_CONFIG)
        solver.solve(
            x0=x0,
            method=SYNTHETIC_CONFIG.method,
            max_residual=SYNTHETIC_CONFIG.max_residual,
            max_evaluations=SYNTHETIC_CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )

    wall_ms: list[float] = []
    nfev: list[int] = []
    njev: list[int] = []
    iterations: list[int] = []
    success: list[bool] = []
    last_solver = None
    for _ in range(repeat):
        solver = Solver(operator=Operator(ROUTE_TEST_GRID, case.copy()), config=SYNTHETIC_CONFIG)
        started = time.perf_counter_ns()
        solver.solve(
            x0=x0,
            method=SYNTHETIC_CONFIG.method,
            max_residual=SYNTHETIC_CONFIG.max_residual,
            max_evaluations=SYNTHETIC_CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        wall_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
        result = solver.result
        if result is None:
            raise RuntimeError("VEQPy route solve produced no SolverResult")
        success.append(bool(result.success))
        nfev.append(int(result.function_evaluations))
        njev.append(int(result.jacobian_evaluations))
        iterations.append(int(result.iterations))
        last_solver = solver

    if last_solver is None or last_solver.result is None:
        raise RuntimeError("VEQPy route timing loop did not run")

    result = last_solver.result
    equilibrium = last_solver.build_equilibrium()
    shape_x = extract_shape_x(last_solver.operator.problem.active_profiles, result.x)
    diagnostics = benchmark_route_case_diagnostics(reference, equilibrium, shape_x)
    raw = np.asarray(last_solver.operator.residual_var(result.x), dtype=np.float64)
    shape_error = float(diagnostics["shape_error"])
    passed = all(success) and shape_error <= ROUTE_SHAPE_MATCH_TOL
    failure_reason = None
    if not all(success):
        failure_reason = "solver_failed"
    elif shape_error > ROUTE_SHAPE_MATCH_TOL:
        failure_reason = "shape_tolerance_failed"

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
                "raw_norm": float(np.linalg.norm(raw)),
                "message": str(result.message),
            }
        },
        "diagnostics": {
            "shape_error": shape_error,
            "shape_match_tol": ROUTE_SHAPE_MATCH_TOL,
            **diagnostics,
        },
    }
    if failure_reason is not None:
        runtime["failure_reason"] = failure_reason
    return runtime


def _plan_row(spec: RouteBenchmarkSpec) -> dict[str, Any]:
    return {
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
    scope: str,
    specs: tuple[RouteBenchmarkSpec, ...],
    repeat: int,
    warmup: int,
    no_run: bool,
) -> None:
    console.print(Text("[config]", style="bold cyan"))
    lines = (
        f"scope: [green]{scope}[/]",
        f"cases: [green]{len(specs)}[/]",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(DEFAULT_SCOPE, "uniform", "full"),
        default=DEFAULT_SCOPE,
    )
    parser.add_argument("--case", action="append", help="Case name or route:coord:nodes:constraint")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = _console()
    specs = filter_route_specs(
        _iter_route_specs(scope=args.scope),
        set(args.case) if args.case else None,
    )
    rows = [_plan_row(spec) for spec in specs]
    if not args.quiet_progress:
        _print_config_tree(
            console,
            scope=str(args.scope),
            specs=specs,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            no_run=bool(args.no_run),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    if not args.no_run:
        with _progress_context(console, quiet=args.quiet_progress) as progress:
            task_id = None
            if progress is not None:
                task_id = progress.add_task(
                    "veqpy-routes",
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
                    rows[index]["runtime"] = _measure_case(
                        spec,
                        warmup=int(args.warmup),
                        repeat=int(args.repeat),
                    )
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
        "schema": "veqpy.routes.v1",
        "scope": str(args.scope),
        "case_count": len(rows),
        "engine": ENGINE_LABEL,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "timing_note": (
            "wall time around Solver.solve(...) only; Operator construction and x0 packing "
            "are excluded to match benchmarks.veqlib_routes' Numba column"
        ),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "layout": {
            "reference_grid": grid_payload(REFERENCE_GRID),
            "test_grid": grid_payload(ROUTE_TEST_GRID),
            "source_sample_count": ROUTE_TEST_SOURCE_SAMPLE_COUNT,
            "solver": {
                "method": SYNTHETIC_CONFIG.method,
                "max_residual": float(SYNTHETIC_CONFIG.max_residual),
                "max_evaluations": int(SYNTHETIC_CONFIG.max_evaluations),
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
