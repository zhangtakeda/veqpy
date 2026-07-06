#!/usr/bin/env python3
"""Route benchmark comparing VEQlib native Kernel with VEQPy Numba Kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    CORE_DIR,
    DEFAULT_ROUTE_SCOPE,
    REPO_ROOT,
    SYNTHETIC_SOLVER_LABEL,
    SYNTHETIC_SOLVER_MAX_EVALUATIONS,
    SYNTHETIC_SOLVER_MAX_RESIDUAL,
    cpu_affinity,
    engine_payload,
    filter_route_specs,
    format_float,
    iter_route_specs,
    max_abs,
    measure_solver,
    route_kernel_case,
    route_spec_label,
    route_spec_selector,
    route_topology_payload,
    runtime_env,
    solve_native_case,
    solve_numba_case,
    summarize_runtime_rows,
    timing_median_ms,
    write_json,
)
from veqlib.facade import KernelRecipe
from veqlib.facade.builder import default_kernel_cache_root

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqlib_routes.json"
VALIDATION_ATOL = 1.0e-6
NATIVE_SOLVER_INITIAL_POLICY = "cold"
NATIVE_SOLVER_CONTINUATION_POLICY = "cold"
NATIVE_SOLVER_NORMALIZATION = "fast"


def _recipe(args: argparse.Namespace) -> KernelRecipe:
    return KernelRecipe(
        backend="cxx",
        build=args.build,
        layout=args.layout,
        cmake_build_type=args.cmake_build_type,
        fp_mode=args.fp_mode,
        enzyme_jacobian_batch_width=args.enzyme_jacobian_batch_width,
    )


def _native_engine_label(args: argparse.Namespace) -> str:
    suffix = "lm" if args.method == "levenberg-marquardt" else "powell"
    return f"veqlib-{args.build}-{suffix}"


def _measure_row(args: argparse.Namespace, spec) -> dict[str, Any]:
    case = route_kernel_case(
        spec,
        method=args.method,
        max_residual=SYNTHETIC_SOLVER_MAX_RESIDUAL,
        max_evaluations=args.max_evaluations,
        pj2_f_count=5,
        initial=args.initial,
        norm=args.norm,
    )
    recipe = _recipe(args)
    base_row: dict[str, Any] = {
        "case": route_spec_label(spec),
        "selector": route_spec_selector(spec),
        "route": case.topology.route,
        "coordinate": case.topology.coordinate,
        "nodes": case.topology.nodes,
        "constraint": case.topology.constraint_label,
        "topology": route_topology_payload(case.topology, recipe),
        "runtime": {"status": "not_requested"},
    }
    if args.no_run:
        return base_row

    try:
        numba_measure = measure_solver(
            lambda: solve_numba_case(case),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        native_measure = measure_solver(
            lambda: solve_native_case(case, recipe=recipe),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        native_engine = engine_payload(native_measure)
        numba_engine = engine_payload(numba_measure)
        compare = {
            "x_max_abs": max_abs(native_engine["x"], numba_engine["x"]),
            "raw_max_abs": max_abs(native_engine["raw"], numba_engine["raw"]),
        }
        compare["within_atol"] = bool(
            compare["x_max_abs"] <= VALIDATION_ATOL and compare["raw_max_abs"] <= VALIDATION_ATOL
        )
        passed = (
            native_engine["success_all"] and numba_engine["success_all"] and compare["within_atol"]
        )
        runtime: dict[str, Any] = {
            "status": "passed" if passed else "failed",
            "x_size": case.topology.x_size,
            "solver_policy": {
                "initial": NATIVE_SOLVER_INITIAL_POLICY,
                "continue": NATIVE_SOLVER_CONTINUATION_POLICY,
                "norm": NATIVE_SOLVER_NORMALIZATION,
            },
            "engines": {
                _native_engine_label(args): native_engine,
                SYNTHETIC_SOLVER_LABEL: numba_engine,
            },
            "closeness_to_numba": compare,
            "reference_dofs": {},
        }
        if not passed:
            runtime["failure_reason"] = "validation_mismatch"
        base_row["runtime"] = runtime
    except Exception as exc:
        base_row["runtime"] = {
            "status": "failed",
            "failure_reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        for name in ("numba_measure", "native_measure"):
            measure = locals().get(name)
            if isinstance(measure, dict):
                kernel = measure.get("kernel")
                close = getattr(kernel, "close", None)
                if close is not None:
                    close()
    return base_row


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print("case,status,x,cxx_ms,numba_ms,speedup,x_max_abs")
    for row in rows:
        runtime = row["runtime"]
        engines = runtime.get("engines", {})
        native_key = next((key for key in engines if key.startswith("veqlib-")), "")
        native = engines.get(native_key)
        numba = engines.get(SYNTHETIC_SOLVER_LABEL)
        cxx_ms = timing_median_ms(native)
        numba_ms = timing_median_ms(numba)
        speedup = numba_ms / cxx_ms if cxx_ms > 0.0 else float("nan")
        print(
            ",".join(
                [
                    str(row["case"]),
                    str(runtime["status"]),
                    str(runtime.get("x_size", "n/a")),
                    format_float(cxx_ms),
                    format_float(numba_ms),
                    format_float(speedup),
                    format_float(
                        float(runtime.get("closeness_to_numba", {}).get("x_max_abs", float("nan"))),
                        precision=3,
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
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--layout", default="degree")
    parser.add_argument("--cmake-build-type", default=None)
    parser.add_argument("--fp-mode", default=None)
    parser.add_argument("--enzyme-jacobian-batch-width", type=int, default=None)
    parser.add_argument("--method", default="powell")
    parser.add_argument("--initial", default="cold")
    parser.add_argument("--norm", default="fast")
    parser.add_argument("--max-evaluations", type=int, default=SYNTHETIC_SOLVER_MAX_EVALUATIONS)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--run-native-in-process", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-artifact-dry-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cache_root = args.cache_root or default_kernel_cache_root()

    specs = filter_route_specs(iter_route_specs(args.scope), args.case)
    rows = [_measure_row(args, spec) for spec in specs]
    summary = summarize_runtime_rows(rows)
    payload = {
        "schema": "veqlib.routes.v2",
        "scope": args.scope,
        "case_count": len(rows),
        "build": args.build,
        "recipe_overrides": {},
        "layout": args.layout,
        "solver_policy": {
            "initial": NATIVE_SOLVER_INITIAL_POLICY,
            "continue": NATIVE_SOLVER_CONTINUATION_POLICY,
            "norm": NATIVE_SOLVER_NORMALIZATION,
            "method": args.method,
            "max_residual": SYNTHETIC_SOLVER_MAX_RESIDUAL,
            "max_evaluations": int(args.max_evaluations),
        },
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "native_isolation": not bool(args.run_native_in_process),
        "validation_atol": VALIDATION_ATOL,
        "cache_root": str(cache_root),
        "source_dir": str(args.source_dir.resolve()),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
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
