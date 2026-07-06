#!/usr/bin/env python3
"""GEQDSK Cxx backend vs VEQPy Numba Kernel benchmark."""

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
    CASE_KEYS,
    CONFIG_LABELS,
    CORE_DIR,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REFERENCE_SOLVER_MAXFEV,
    REPO_ROOT,
    RouteBenchmarkSpec,
    cpu_affinity,
    default_kernel_cache_root,
    engine_payload,
    geqdsk_kernel_case,
    geqdsk_signature,
    max_abs,
    measure_solver,
    runtime_env,
    selected_cases,
    selected_configs,
    solve_native_case,
    solve_numba_case,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    format_optional_speedup,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
    timing_median_ms,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import KernelRecipe

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqlib_geqdsk.json"
VALIDATION_ATOL = 1.0e-6
NATIVE_SOLVER_METHOD = "powell"
NATIVE_SOLVER_INITIAL_POLICY = "cold"
NATIVE_SOLVER_CONTINUATION_POLICY = "cold"
NATIVE_SOLVER_NORMALIZATION = "fast"
REFERENCE_SOLVER_INITIAL_POLICY = "auto"


def _recipe(args: argparse.Namespace) -> KernelRecipe:
    return KernelRecipe(backend="cxx", build=args.build, layout="degree")


def _recipe_payload(recipe: KernelRecipe) -> dict[str, Any]:
    return {
        "backend": recipe.backend,
        "preset": recipe.build,
        "layout": {"packed": recipe.layout},
    }


def _native_engine_label(args: argparse.Namespace) -> str:
    suffix = "lm" if args.method == "levenberg-marquardt" else "powell"
    return f"veqlib-{args.build}-{suffix}"


def _measure_case(args: argparse.Namespace, case_key: str, config_label: str) -> dict[str, Any]:
    route_spec = RouteBenchmarkSpec("PF", "psin", "uniform", "Ip")
    signature = geqdsk_signature(case_key, config_label)
    case = geqdsk_kernel_case(
        case_key,
        config_label,
        route_spec=route_spec,
        method=args.method,
        max_residual=1.0e-6,
        max_evaluations=args.max_evaluations,
        initial=args.initial,
        norm=args.norm,
        boundary_fit_m=args.boundary_fit_m,
        boundary_fit_n=args.boundary_fit_n,
        boundary_maxtol=args.boundary_maxtol,
    )
    recipe = _recipe(args)
    base_row: dict[str, Any] = {
        "case": case_key,
        "config": config_label,
        "row": f"{case_key}:{config_label.lower()}",
        "status": "planned",
        "x_size": case.topology.x_size,
        "signature": signature,
        "topology": {
            "key": case.topology.key,
            "recipe": _recipe_payload(recipe),
            "grid": {"Nr": case.topology.Nr, "Nt": case.topology.Nt},
            "sample_count": case.topology.sample_count,
            "M_max": case.topology.M_max,
        },
    }
    if args.no_run:
        return base_row

    try:
        numba_measure = measure_solver(
            lambda: solve_numba_case(case), warmup=args.warmup, repeat=args.repeat
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
        base_row.update(
            {
                "status": "passed" if passed else "failed",
                "engines": {
                    _native_engine_label(args): native_engine,
                    "veqpy-numba-hybr": numba_engine,
                },
                "closeness_to_numba": compare,
            }
        )
    except Exception as exc:
        base_row.update(
            {
                "status": "failed",
                "failure_reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        for name in ("numba_measure", "native_measure"):
            measure = locals().get(name)
            if isinstance(measure, dict):
                kernel = measure.get("kernel")
                close = getattr(kernel, "close", None)
                if close is not None:
                    close()
    return base_row


def _print_summary(console, rows: list[dict[str, Any]]) -> None:
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("case", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("config", no_wrap=True)
    table.add_column("x", justify="right")
    table.add_column(Text("Cxx (ms)"), justify="right")
    table.add_column(Text("Numba (ms)"), justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("diff", justify="right")
    for row in rows:
        engines = row.get("engines", {})
        native_key = next((key for key in engines if key.startswith("veqlib-")), "")
        native = engines.get(native_key)
        numba = engines.get("veqpy-numba-hybr")
        cxx_ms = timing_median_ms(native)
        numba_ms = timing_median_ms(numba)
        compare = row.get("closeness_to_numba")
        compare = compare if isinstance(compare, dict) else {}
        table.add_row(
            str(row.get("case", "n/a")),
            status_cell(row.get("status", "n/a")),
            str(row.get("config", "n/a")),
            str(row.get("x_size", "n/a")),
            format_optional_float(cxx_ms),
            format_optional_float(numba_ms),
            format_optional_speedup(numba_ms, cxx_ms),
            format_optional_sci(compare.get("x_max_abs")),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--config", action="append", choices=CONFIG_LABELS)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--method", default=NATIVE_SOLVER_METHOD)
    parser.add_argument("--initial", default="cold")
    parser.add_argument("--norm", default="fast")
    parser.add_argument("--max-evaluations", type=int, default=REFERENCE_SOLVER_MAXFEV)
    parser.add_argument("--boundary-fit-m", type=int, default=10)
    parser.add_argument("--boundary-fit-n", type=int, default=10)
    parser.add_argument("--boundary-maxtol", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = reporting_console()
    cache_root = args.cache_root or default_kernel_cache_root()

    case_keys = selected_cases(args.case)
    config_labels = selected_configs(args.config)
    row_plan = [
        (case_key, config_label)
        for case_key in case_keys
        for config_label in config_labels
    ]
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{len(row_plan)}[/]",
                f"build: [green]{args.build}[/]",
                f"initial: [green]{NATIVE_SOLVER_INITIAL_POLICY}[/]",
                f"continue: [green]{NATIVE_SOLVER_CONTINUATION_POLICY}[/]",
                f"norm: [green]{NATIVE_SOLVER_NORMALIZATION}[/]",
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
                "geqdsk",
                total=len(row_plan),
                current="-",
                phase="[cyan]run[/]",
            )
        for case_key, config_label in row_plan:
            current = f"{case_key}:{config_label.lower()}"
            if progress is not None and task_id is not None:
                progress.update(task_id, current=current, phase="[cyan]run[/]")
            row = _measure_case(args, case_key, config_label)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=progress_phase(row.get("status")))
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
        if not args.quiet_progress:
            console.print()
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)
    _print_summary(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
