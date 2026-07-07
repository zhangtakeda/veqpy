#!/usr/bin/env python3
"""Numba backend route benchmark matrix on the Kernel API."""

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
    DEFAULT_ROUTE_K_MAX,
    DEFAULT_ROUTE_L_MAX,
    DEFAULT_ROUTE_M_MAX,
    DEFAULT_ROUTE_NR,
    DEFAULT_ROUTE_NT,
    DEFAULT_ROUTE_SAMPLE_COUNT,
    DEFAULT_ROUTE_SCOPE,
    REPO_ROOT,
    ROUTE_SHAPE_MATCH_TOL,
    SYNTHETIC_SOLVER_LABEL,
    SYNTHETIC_SOLVER_MAX_EVALUATIONS,
    SYNTHETIC_SOLVER_MAX_RESIDUAL,
    SYNTHETIC_SOLVER_METHOD,
    benchmark_route_case_diagnostics,
    cpu_affinity,
    extract_shape_x,
    filter_route_specs,
    grid_payload,
    iter_route_specs,
    measure_kernel_case,
    route_kernel_case,
    route_spec_label,
    route_spec_selector,
    runtime_env,
    runtime_payload,
    summarize_runtime_rows,
    synthetic_route_reference,
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
from veqpy import KernelRecipe

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "numba_routes.json"


def _measure_row(args: argparse.Namespace, spec) -> dict[str, Any]:
    base_row = _plan_row(spec)
    case = route_kernel_case(
        spec,
        method=args.method,
        max_residual=SYNTHETIC_SOLVER_MAX_RESIDUAL,
        max_evaluations=args.max_evaluations,
        initial=args.initial,
        norm=args.norm,
    )
    if args.no_run:
        return base_row
    try:
        measure = measure_kernel_case(
            case,
            recipe=KernelRecipe(backend="numba", layout="degree"),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        kernel = measure["kernel"]
        diagnostics = benchmark_route_case_diagnostics(
            synthetic_route_reference(),
            kernel.build_equilibrium(),
            extract_shape_x(case.topology, measure["result"].x),
        )
        shape_ok = float(diagnostics["shape_error"]) <= ROUTE_SHAPE_MATCH_TOL
        status = "passed" if measure["success"] and shape_ok else "failed"
        failure_reason = None
        if not measure["success"]:
            failure_reason = "solver_failed"
        elif not shape_ok:
            failure_reason = "shape_tolerance_failed"
        base_row["runtime"] = runtime_payload(
            status=status,
            x_size=case.topology.x_size,
            engine=SYNTHETIC_SOLVER_LABEL,
            measure=measure,
            diagnostics=diagnostics,
            failure_reason=failure_reason,
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


def _plan_row(spec) -> dict[str, Any]:
    return {
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
        engine = runtime_engine_payload(runtime, SYNTHETIC_SOLVER_LABEL)
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
    parser.add_argument(
        "--scope", choices=("ip-uniform", "uniform", "full"), default=DEFAULT_ROUTE_SCOPE
    )
    parser.add_argument(
        "--case", action="append", help="Case name or route:coordinate:nodes:constraint"
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--method", default=SYNTHETIC_SOLVER_METHOD)
    parser.add_argument("--initial", default="cold")
    parser.add_argument("--norm", default="fast")
    parser.add_argument("--max-evaluations", type=int, default=SYNTHETIC_SOLVER_MAX_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = reporting_console()
    specs = filter_route_specs(iter_route_specs(args.scope), args.case)
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"scope: [green]{args.scope}[/]",
                f"cases: [green]{len(specs)}[/]",
                f"engine: [green]{SYNTHETIC_SOLVER_LABEL}[/]",
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
                "numba-routes",
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
        "schema": "veqpy.numba.routes.v1",
        "scope": args.scope,
        "case_count": len(rows),
        "engine": SYNTHETIC_SOLVER_LABEL,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "timing_note": (
            "Kernel solve elapsed time after runtime case setup; "
            "Kernel handle construction is excluded per repeat"
        ),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "layout": {
            "reference_grid": grid_payload(
                nr=64,
                nt=32,
                l_max=DEFAULT_ROUTE_L_MAX,
                m_max=DEFAULT_ROUTE_M_MAX,
                k_max=None,
            ),
            "test_grid": grid_payload(
                nr=DEFAULT_ROUTE_NR,
                nt=DEFAULT_ROUTE_NT,
                l_max=DEFAULT_ROUTE_L_MAX,
                m_max=DEFAULT_ROUTE_M_MAX,
                k_max=DEFAULT_ROUTE_K_MAX,
            ),
            "source_sample_count": DEFAULT_ROUTE_SAMPLE_COUNT,
            "solver": {
                "method": args.method,
                "max_residual": SYNTHETIC_SOLVER_MAX_RESIDUAL,
                "max_evaluations": int(args.max_evaluations),
                "norm": args.norm,
            },
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
