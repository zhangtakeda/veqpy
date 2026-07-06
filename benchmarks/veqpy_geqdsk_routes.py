#!/usr/bin/env python3
"""GEQDSK-backed VEQPy/Numba route benchmark matrix on the Kernel API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    format_float,
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
    timing_median_ms,
    write_json,
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


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print("case,status,x,median_ms,nfev,raw_norm")
    for row in rows:
        runtime = row["runtime"]
        engine = runtime.get("engines", {}).get(ENGINE_LABEL)
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

    args.geqdsk = args.geqdsk.expanduser().resolve()
    specs = filter_route_specs(
        iter_route_specs(args.scope, default_scope=DEFAULT_SCOPE, allow_grid=False),
        args.case,
    )
    rows = [_measure_row(args, spec) for spec in specs]
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
            "wall time around Kernel.solve(...); "
            "Kernel handle construction is included per repeat"
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
        _print_rows(rows)
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
