#!/usr/bin/env python3
"""Numba Kernel.pareto() screening benchmark on GEQDSK Ref topologies."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    CASE_KEYS,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REPO_ROOT,
    RouteBenchmarkSpec,
    cpu_affinity,
    geqdsk_kernel_case,
    runtime_env,
    selected_cases,
    topology_profile_counts,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import Kernel, KernelRecipe, KernelTopology, ParetoResult, ParetoSample

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "numba_pareto.json"
DEFAULT_THRESHOLD_SCALES = (1.0e-2, 5.0e-3, 1.0e-3)
DEFAULT_MAX_CANDIDATES = 10000
DEFAULT_MAX_EVALUATIONS = 2000


def _run_case(
    args: argparse.Namespace,
    case_key: str,
    threshold_scales: tuple[float, ...],
) -> dict[str, Any]:
    route_spec = RouteBenchmarkSpec("PF", "psin", "uniform", "Ip")
    kernel_case = geqdsk_kernel_case(
        case_key,
        "Ref",
        route_spec=route_spec,
        nr=args.nr,
        nt=args.nt,
        max_evaluations=args.max_evaluations,
    )
    thresholds = _thresholds_for_boundary(kernel_case.boundary.a, threshold_scales)
    row = _planned_row(case_key, kernel_case.topology, kernel_case.boundary, thresholds)
    kernel = None
    started = time.perf_counter_ns()
    try:
        kernel = Kernel(
            topology=kernel_case.topology,
            recipe=KernelRecipe(backend="numba", layout="degree"),
            config=kernel_case.config,
        )
        result = kernel.pareto(
            kernel_case.boundary,
            kernel_case.source,
            config=kernel_case.config,
            max_shape_error=tuple(threshold["meters"] for threshold in thresholds),
            pareto_by=args.pareto_by,
            strategy=args.strategy,
            metric=args.metric,
            max_candidates=args.max_candidates,
        )
        elapsed_ms = float(time.perf_counter_ns() - started) / 1.0e6
        row["runtime"] = _runtime_payload(result, thresholds, elapsed_ms)
    except Exception as exc:
        elapsed_ms = float(time.perf_counter_ns() - started) / 1.0e6
        row["runtime"] = {
            "status": "failed",
            "failure_reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": elapsed_ms,
        }
    finally:
        if kernel is not None:
            kernel.close()
    return row


def _planned_row(
    case_key: str,
    topology: KernelTopology,
    boundary,
    thresholds: tuple[dict[str, float], ...],
) -> dict[str, Any]:
    return {
        "case": case_key,
        "config": "Ref",
        "route": "PF_psin_uniform_Ip",
        "capacity": _topology_payload(topology),
        "boundary": _boundary_payload(boundary),
        "thresholds": list(thresholds),
        "runtime": {"status": "not_requested"},
    }


def _topology_payload(topology: KernelTopology) -> dict[str, Any]:
    return {
        "key": topology.key,
        "x_size": int(topology.x_size),
        "profile_counts": topology_profile_counts(topology),
        "grid": {
            "Nr": int(topology.Nr),
            "Nt": int(topology.Nt),
            "L_max": int(topology.L_max),
            "M_max": int(topology.M_max),
            "K_max": int(topology.K_max),
        },
        "sample_count": int(topology.sample_count),
    }


def _boundary_payload(boundary) -> dict[str, Any]:
    return {
        "source": "benchmarks._common.GEQDSK_BOUNDARY_PARAMETERS",
        "fit_backend": "numpy",
        "fit_method": boundary.fit_method,
        "fit_rms": None if boundary.fit_rms is None else float(boundary.fit_rms),
        "fit_max_curve_error": (
            None
            if boundary.fit_max_curve_error is None
            else float(boundary.fit_max_curve_error)
        ),
        "fit_c_order": boundary.fit_c_order,
        "fit_s_order": boundary.fit_s_order,
        "fit_note": (
            "Frozen GEQDSK LCFS least-square fit; no boundary fitting is performed "
            "inside this Pareto benchmark run."
        ),
    }


def _thresholds_for_boundary(
    minor_radius: float,
    scales: tuple[float, ...],
) -> tuple[dict[str, float], ...]:
    return tuple(
        {"scale": float(scale), "meters": float(scale) * float(minor_radius)}
        for scale in scales
    )


def _runtime_payload(
    result: ParetoResult,
    thresholds: tuple[dict[str, float], ...],
    elapsed_ms: float,
) -> dict[str, Any]:
    candidate_count = len(result.samples)
    valid_count = sum(1 for sample in result.samples if sample.result.success)
    reference_ms = float(result.reference.time)
    candidate_solve_ms = float(sum(sample.time for sample in result.samples))
    solver_elapsed_ms = reference_ms + candidate_solve_ms
    selected = {
        f"{threshold['scale']:.16g}": _sample_payload(
            result.selected[threshold["meters"]]
        )
        for threshold in thresholds
        if threshold["meters"] in result.selected
    }
    return {
        "status": "passed" if result.reference.result.success else "failed",
        "elapsed_ms": float(elapsed_ms),
        "reference_solve_ms": reference_ms,
        "candidate_solve_ms": candidate_solve_ms,
        "solver_elapsed_ms": solver_elapsed_ms,
        "overhead_ms": float(elapsed_ms) - solver_elapsed_ms,
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(valid_count),
        "frontier_count": int(len(result.frontier)),
        "evaluations_per_second": _evaluations_per_second(candidate_count, elapsed_ms),
        "reference": _sample_payload(result.reference),
        "samples": [_sample_payload(sample) for sample in result.samples],
        "frontier": [_sample_payload(sample) for sample in result.frontier],
        "selected": selected,
    }


def _evaluations_per_second(candidate_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return float("nan")
    return 1000.0 * float(candidate_count) / float(elapsed_ms)


def _sample_payload(sample: ParetoSample) -> dict[str, Any]:
    return {
        "signature": sample.signature.to_variant_kwargs(),
        "counts": int(sample.counts),
        "time_ms": float(sample.time),
        "complexity": int(sample.complexity),
        "shape_error": float(sample.shape_error),
        "success": bool(sample.result.success),
        "nfev": int(sample.result.nfev),
        "raw_norm": float(sample.result.raw_norm),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "runtime_passed": 0,
        "runtime_failed": 0,
        "runtime_not_requested": 0,
    }
    for row in rows:
        status = row.get("runtime", {}).get("status")
        if status == "passed":
            counts["runtime_passed"] += 1
        elif status == "failed":
            counts["runtime_failed"] += 1
        elif status == "not_requested":
            counts["runtime_not_requested"] += 1
    return counts


def _print_summary(console, rows: list[dict[str, Any]]) -> None:
    summary = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    summary.add_column("case", no_wrap=True)
    summary.add_column("status", no_wrap=True)
    summary.add_column("ref x", justify="right")
    summary.add_column("evals", justify="right")
    summary.add_column("frontier", justify="right")
    summary.add_column(Text("elapsed (ms)"), justify="right")
    summary.add_column(Text("eval/s"), justify="right")

    for row in rows:
        runtime = row.get("runtime", {})
        reference = runtime.get("reference", {})
        summary.add_row(
            str(row.get("case", "n/a")),
            status_cell(runtime.get("status", "n/a")),
            str(reference.get("counts", row.get("capacity", {}).get("x_size", "n/a"))),
            str(runtime.get("candidate_count", "n/a")),
            str(runtime.get("frontier_count", "n/a")),
            format_optional_float(runtime.get("elapsed_ms"), precision=1),
            format_optional_float(runtime.get("evaluations_per_second"), precision=2),
        )
    console.print(summary)
    console.print()

    selected_table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    selected_table.add_column("case", no_wrap=True)
    selected_table.add_column("tol", justify="right")
    selected_table.add_column(Text("tol (m)"), justify="right")
    selected_table.add_column("x", justify="right")
    selected_table.add_column("complexity", justify="right")
    selected_table.add_column(Text("time (ms)"), justify="right")
    selected_table.add_column(Text("R error (m)"), justify="right")
    for row in rows:
        selected = row.get("runtime", {}).get("selected", {})
        thresholds = row.get("thresholds", [])
        for threshold in thresholds:
            scale = float(threshold["scale"])
            sample = selected.get(f"{scale:.16g}")
            selected_table.add_row(
                str(row.get("case", "n/a")),
                f"{scale:g} a",
                format_optional_sci(threshold["meters"]),
                str(sample["counts"]) if sample else "n/a",
                str(sample["complexity"]) if sample else "n/a",
                format_optional_float(sample["time_ms"], precision=3) if sample else "n/a",
                format_optional_sci(sample["shape_error"]) if sample else "n/a",
            )
    console.print(selected_table)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument(
        "--strategy",
        choices=("tail", "energy", "adaptive", "balanced"),
        default="adaptive",
    )
    parser.add_argument("--metric", choices=("rms", "max"), default="rms")
    parser.add_argument("--pareto-by", choices=("counts", "time", "complexity"), default="counts")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--nr", type=int, default=REFERENCE_LAYOUT_NR)
    parser.add_argument("--nt", type=int, default=REFERENCE_LAYOUT_NT)
    parser.add_argument(
        "--threshold-scale",
        action="append",
        type=float,
        default=None,
        help="Shape-error tolerance as a fraction of boundary minor radius a; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = reporting_console()
    case_keys = selected_cases(args.case)
    threshold_scales = tuple(
        float(scale) for scale in (args.threshold_scale or DEFAULT_THRESHOLD_SCALES)
    )

    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{', '.join(case_keys)}[/]",
                "backend: [green]numba[/]",
                "capacity: [green]GEQDSK Ref topology[/]",
                "route: [green]PF/psin/uniform/Ip[/]",
                f"grid: [green]{args.nr} x {args.nt}[/]",
                f"strategy: [green]{args.strategy}[/]",
                f"metric: [green]{args.metric}[/]",
                f"pareto_by: [green]{args.pareto_by}[/]",
                f"max candidates/case: [green]{args.max_candidates}[/]",
                f"thresholds: [green]{', '.join(f'{scale:g}*a' for scale in threshold_scales)}[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))

    rows: list[dict[str, Any]] = []
    with progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "numba-pareto",
                total=len(case_keys),
                current="-",
                phase="[cyan]run[/]",
            )
        for case_key in case_keys:
            row = _run_case(args, case_key, threshold_scales)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    current=case_key,
                    phase=progress_phase(row.get("runtime", {}).get("status")),
                )
                progress.advance(task_id)

    payload = {
        "schema": "veqpy.numba.pareto_geqdsk.v1",
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "run_note": (
            "GEQDSK Ref-capacity Pareto screening benchmark. max_candidates is the "
            "candidate solve budget per case and excludes the reference solve. "
            "threshold_scale values are multiplied by each boundary minor radius a. "
            "GEQDSK boundaries are pre-fitted frozen parameterized KernelBoundary "
            "inputs from benchmarks._common; this benchmark does not refit LCFS points."
        ),
        "args": {
            "cases": list(case_keys),
            "strategy": args.strategy,
            "metric": args.metric,
            "pareto_by": args.pareto_by,
            "max_candidates": int(args.max_candidates),
            "max_evaluations": int(args.max_evaluations),
            "nr": int(args.nr),
            "nt": int(args.nt),
            "threshold_scale": list(threshold_scales),
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        if not args.quiet_progress:
            console.print()
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)
            console.print()
    _print_summary(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
