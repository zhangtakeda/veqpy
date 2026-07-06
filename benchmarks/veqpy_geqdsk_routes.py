#!/usr/bin/env python3
"""GEQDSK-backed VEQPy/Numba route benchmark matrix on the Kernel API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    CASE_REFERENCE_GFILES,
    DEFAULT_ROUTE_K_MAX,
    DEFAULT_ROUTE_L_MAX,
    DEFAULT_ROUTE_M_MAX,
    DEFAULT_ROUTE_SAMPLE_COUNT,
    GEQDSK_ROUTE_PROFILE_SIGNATURE,
    REPO_ROOT,
    RouteBenchmarkSpec,
    cpu_affinity,
    filter_route_specs,
    geqdsk_kernel_case,
    grid_payload,
    iter_route_specs,
    measure_solver,
    route_spec_label,
    route_spec_selector,
    runtime_env,
    runtime_payload,
    solve_numba_case,
    summarize_runtime_rows,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    nfev_median,
    print_config_tree,
    print_outputs_tree,
    print_runtime_failures,
    print_runtime_summary,
    progress_context,
    progress_phase,
    runtime_engine_payload,
    status_cell,
    timing_median_ms,
)
from benchmarks._reporting import (
    console as reporting_console,
)

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqpy_geqdsk_routes.json"
DEFAULT_SCOPE = "ip-uniform"
DEFAULT_GEQDSK_CASE = "efit"
DEFAULT_GEQDSK = CASE_REFERENCE_GFILES[DEFAULT_GEQDSK_CASE]
ENGINE_LABEL = "veqpy-numba-lm"
DEFAULT_SOLVE_NR = 32
DEFAULT_SOLVE_NT = 32
PROFILE_RMS_TOL = 5.0e-2
GEQDSK_METHOD = "levenberg-marquardt"
GEQDSK_MAX_RESIDUAL = 1.0e-7
GEQDSK_MAX_EVALUATIONS = 2000


def _measure_row(args: argparse.Namespace, spec: RouteBenchmarkSpec) -> dict[str, Any]:
    base_row = _plan_row(args.geqdsk, spec)
    case = geqdsk_kernel_case(
        args.geqdsk_case,
        "Route",
        geqdsk_path=args.geqdsk,
        route_spec=spec,
        signature=GEQDSK_ROUTE_PROFILE_SIGNATURE,
        nr=args.solve_nr,
        nt=args.solve_nt,
        sample_count=args.source_sample_count,
        method=args.method,
        max_residual=GEQDSK_MAX_RESIDUAL,
        max_evaluations=args.max_evaluations,
        initial=args.initial,
        norm=args.norm,
        boundary_fit_m=args.boundary_fit_m,
        boundary_fit_n=args.boundary_fit_n,
        boundary_maxtol=args.boundary_maxtol,
    )
    if args.no_run:
        return base_row
    try:
        measure = measure_solver(
            lambda: solve_numba_case(case),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        status = "passed" if measure["success"] else "failed"
        base_row["runtime"] = runtime_payload(
            status=status,
            x_size=case.topology.x_size,
            engine=ENGINE_LABEL,
            measure=measure,
            diagnostics={
                "shape_match_tol": 1.0e-2,
                "profile_rms_tol": PROFILE_RMS_TOL,
            },
            failure_reason=None if status == "passed" else "solver_failed",
        )
    except Exception as exc:
        base_row["runtime"] = {
            "status": "failed",
            "failure_reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        measure_obj = locals().get("measure")
        if isinstance(measure_obj, dict):
            kernel = measure_obj.get("kernel")
            close = getattr(kernel, "close", None)
            if close is not None:
                close()
    return base_row


def _plan_row(geqdsk_path: Path, spec: RouteBenchmarkSpec) -> dict[str, Any]:
    return {
        "geqdsk": str(geqdsk_path),
        "case": route_spec_label(spec),
        "selector": route_spec_selector(spec),
        "route": str(spec.mode),
        "coordinate": str(spec.coordinate),
        "nodes": str(spec.nodes),
        "constraint": str(spec.constraint),
        "runtime": {"status": "not_requested"},
    }


def _print_timing_table(console, rows: list[dict[str, Any]]) -> None:
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
        diagnostics = runtime.get("diagnostics", {})
        table.add_row(
            str(row.get("case", "n/a")),
            status_cell(runtime["status"]),
            str(runtime.get("x_size", "n/a")),
            format_optional_float(timing_median_ms(engine)),
            nfev_median(engine),
            format_optional_sci(None if engine is None else engine.get("raw_norm")),
            format_optional_sci(diagnostics.get("shape_error")),
            format_optional_sci(diagnostics.get("psi_r_rel_rms_error")),
            format_optional_sci(diagnostics.get("ff_psi_rel_rms_error")),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=(DEFAULT_SCOPE, "uniform"), default=DEFAULT_SCOPE)
    parser.add_argument(
        "--case", action="append", help="Case name or route:coordinate:nodes:constraint"
    )
    parser.add_argument("--geqdsk", type=Path, default=DEFAULT_GEQDSK)
    parser.add_argument(
        "--geqdsk-case", choices=("solovev", "chease", "efit"), default=DEFAULT_GEQDSK_CASE
    )
    parser.add_argument("--source-sample-count", type=int, default=DEFAULT_ROUTE_SAMPLE_COUNT)
    parser.add_argument("--reference-nr", type=int, default=64)
    parser.add_argument("--reference-nt", type=int, default=32)
    parser.add_argument("--solve-nr", type=int, default=DEFAULT_SOLVE_NR)
    parser.add_argument("--solve-nt", type=int, default=DEFAULT_SOLVE_NT)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--method", default=GEQDSK_METHOD)
    parser.add_argument("--initial", default="cold")
    parser.add_argument("--norm", default="none")
    parser.add_argument("--max-evaluations", type=int, default=GEQDSK_MAX_EVALUATIONS)
    parser.add_argument("--boundary-fit-m", type=int, default=10)
    parser.add_argument("--boundary-fit-n", type=int, default=10)
    parser.add_argument("--boundary-maxtol", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = reporting_console()
    args.geqdsk = args.geqdsk.expanduser().resolve()
    specs = filter_route_specs(
        iter_route_specs(args.scope, default_scope=DEFAULT_SCOPE, allow_grid=False),
        args.case,
    )
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"scope: [green]{args.scope}[/]",
                f"geqdsk: [green]{args.geqdsk}[/]",
                f"route cases: [green]{len(specs)}[/]",
                f"engine: [green]{ENGINE_LABEL}[/]",
                f"mode: [green]{'plan-only' if args.no_run else 'run'}[/]",
                f"warmup: [green]{args.warmup}[/]",
                f"repeat: [green]{args.repeat}[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "veqpy-geqdsk-routes",
                total=len(specs),
                current="-",
                phase="[cyan]run[/]",
            )
        for spec in specs:
            if progress is not None and task_id is not None:
                progress.update(task_id, current=route_spec_selector(spec), phase="[cyan]run[/]")
            row = _measure_row(args, spec)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=progress_phase(row["runtime"]["status"]))
                progress.advance(task_id)
    summary = summarize_runtime_rows(rows)
    payload = {
        "schema": "veqpy.geqdsk_routes.v1",
        "scope": args.scope,
        "geqdsk": str(args.geqdsk),
        "case_count": len(rows),
        "engine": ENGINE_LABEL,
        "run_mode": "plan-only" if args.no_run else "run",
        "default_geqdsk_case": DEFAULT_GEQDSK_CASE,
        "skip_reason": "no_run" if args.no_run else None,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "timing_note": (
            "Kernel solve elapsed time after runtime case setup; "
            "Kernel handle construction is excluded per repeat"
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
                "method": args.method,
                "max_residual": GEQDSK_MAX_RESIDUAL,
                "max_evaluations": int(args.max_evaluations),
                "canonical_initial_policy": "geometric-refined",
                "route_initial_policy": None,
            },
            "equivalence_thresholds": {
                "shape_error": 1.0e-2,
                "psi_r_rel_rms_error": PROFILE_RMS_TOL,
                "ff_psi_rel_rms_error": PROFILE_RMS_TOL,
            },
            "profile_signature": GEQDSK_ROUTE_PROFILE_SIGNATURE,
            "grid": grid_payload(
                nr=int(args.solve_nr),
                nt=int(args.solve_nt),
                l_max=DEFAULT_ROUTE_L_MAX,
                m_max=DEFAULT_ROUTE_M_MAX,
                k_max=DEFAULT_ROUTE_K_MAX,
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        if not args.quiet_progress:
            console.print()
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)
    print_runtime_summary(
        console,
        summary,
        ("total", "runtime_passed", "runtime_failed", "runtime_not_requested"),
    )
    print_runtime_failures(console, rows)
    _print_timing_table(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
