#!/usr/bin/env python3
"""VEQPy/Numba route benchmark matrix on the Kernel API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    SYNTHETIC_SOLVER_LABEL,
    SYNTHETIC_SOLVER_MAX_EVALUATIONS,
    SYNTHETIC_SOLVER_MAX_RESIDUAL,
    SYNTHETIC_SOLVER_METHOD,
    cpu_affinity,
    filter_route_specs,
    format_float,
    grid_payload,
    iter_route_specs,
    measure_solver,
    route_kernel_case,
    route_spec_label,
    route_spec_selector,
    runtime_env,
    runtime_payload,
    solve_numba_case,
    summarize_runtime_rows,
    timing_median_ms,
    write_json,
)

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqpy_routes.json"


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
        measure = measure_solver(
            lambda: solve_numba_case(case),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        status = "passed" if measure["success"] else "failed"
        base_row["runtime"] = runtime_payload(
            status=status,
            x_size=case.topology.x_size,
            engine=SYNTHETIC_SOLVER_LABEL,
            measure=measure,
            diagnostics={},
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


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print("case,status,x,median_ms,nfev,raw_norm")
    for row in rows:
        runtime = row["runtime"]
        engine = runtime.get("engines", {}).get(SYNTHETIC_SOLVER_LABEL)
        print(
            ",".join(
                [
                    str(row["case"]),
                    str(runtime["status"]),
                    str(runtime.get("x_size", "n/a")),
                    format_float(timing_median_ms(engine)),
                    format_float(float(engine["nfev"]["median"]) if engine else float("nan")),
                    format_float(
                        float(engine["raw_norm"]) if engine else float("nan"), precision=3
                    ),
                ]
            )
        )


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
    parser.add_argument("--norm", default="none")
    parser.add_argument("--max-evaluations", type=int, default=SYNTHETIC_SOLVER_MAX_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    specs = filter_route_specs(iter_route_specs(args.scope), args.case)
    rows = [_measure_row(args, spec) for spec in specs]
    summary = summarize_runtime_rows(rows)
    payload = {
        "schema": "veqpy.routes.v1",
        "scope": args.scope,
        "case_count": len(rows),
        "engine": SYNTHETIC_SOLVER_LABEL,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "timing_note": (
            "wall time around Kernel.solve(...); "
            "Kernel handle construction is included per repeat"
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
            },
        },
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
    if not args.quiet_progress:
        _print_rows(rows)
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
