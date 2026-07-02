#!/usr/bin/env python3
"""Continuation-policy benchmark for VEQlib using effective nfev.

This benchmark ports the remote certified-continuation sweep onto the current
top-level ``benchmarks`` package and the typed facade API.  The primary metric is
``effective_nfev``: the total residual-evaluation count reported by the native
solve, including warm-start certificates/predictors/chord attempts as well as
fallback nonlinear solves.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
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
    REPO_ROOT,
    cpu_affinity,
    float_stats,
    int_stats,
    runtime_env,
    write_json,
)
from benchmarks.veqlib_geqdsk_pareto import GeqdskConfigCase, _make_cases
from veqlib.facade import (
    KernelBoundary,
    KernelConfig,
    KernelRegistry,
    KernelSource,
    SolveResult,
    VEQlibSolver,
)
from veqlib.facade.builder import default_kernel_cache_root

DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "veqlib_continuation"
UPDATE_CHOICES = ("ip", "boundary", "source", "mixed")
DEFAULT_SPANS = (0.01,)
POLICY_CHOICES = (
    "cold-zeros",
    "cold-geometric",
    "cold",
    "warm-fixed",
    "warm-predict",
    "warm-chord",
    "warm",
)
COLD_POLICIES = frozenset({"cold-zeros", "cold-geometric", "cold"})
WARM_POLICY_INITIAL_POLICY = "cold"
DEFAULT_POLICIES = POLICY_CHOICES
SUMMARY_POLICIES = ("cold", "warm-fixed", "warm-predict", "warm-chord")
UPDATE_LABELS = {
    "ip": "C1 Ip",
    "boundary": "C2 boundary",
    "source": "C3 source",
    "mixed": "C4 mixed",
}
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")
RuntimePoint = tuple[KernelBoundary, KernelSource]


def _console() -> Console:
    return Console(highlight=False)


def _scan_offsets(*, points: int, relative_span: float) -> list[float]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return [0.0]
    lower = -0.5 * float(relative_span)
    step = float(relative_span) / float(points - 1)
    return [float(lower + step * index) for index in range(points)]


def _with_case_suffix(kernel_source: KernelSource, suffix: str) -> KernelSource:
    case_name = kernel_source.case_name or "case"
    return replace(kernel_source, case_name=f"{case_name}-{suffix}")


def _scale_ip(kernel_source: KernelSource, offset: float, *, strength: float = 1.0) -> KernelSource:
    scaled_ip = float(kernel_source.scaled_Ip)
    if not np.isfinite(scaled_ip):
        raise ValueError("ip continuation update requires finite scaled_Ip")
    return replace(kernel_source, scaled_Ip=float(scaled_ip * (1.0 + strength * offset)))


def _scaled_boundary_array(
    values: np.ndarray,
    offset: float,
    *,
    strength: float,
    sine: bool,
) -> np.ndarray:
    updated = np.array(values, dtype=np.float64, copy=True)
    for index, value in enumerate(updated):
        if sine and index == 0:
            continue
        direction = 1.0 if index % 2 == 0 else -1.0
        weight = 1.0 / float(index + 1)
        updated[index] = float(value) * (1.0 + strength * offset * direction * weight)
    return updated


def _scale_boundary(
    boundary: KernelBoundary,
    offset: float,
    *,
    strength: float = 1.0,
) -> KernelBoundary:
    return replace(
        boundary,
        c_offsets=_scaled_boundary_array(boundary.c_offsets, offset, strength=strength, sine=False),
        s_offsets=_scaled_boundary_array(boundary.s_offsets, offset, strength=strength, sine=True),
    )


def _scaled_source_array(
    values: np.ndarray,
    offset: float,
    *,
    strength: float,
    sign: float,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size <= 1:
        return np.array(values_arr, dtype=np.float64, copy=True)
    updated = np.empty_like(values_arr, dtype=np.float64)
    count = int(values_arr.size)
    for index, value in enumerate(values_arr):
        rho = float(index) / float(count - 1)
        smooth_shape = 0.55 * (2.0 * rho - 1.0) + 0.45 * math.cos(math.pi * rho)
        updated[index] = float(value) * (1.0 + strength * offset * sign * smooth_shape)
    return updated


def _scale_source(
    kernel_source: KernelSource,
    offset: float,
    *,
    strength: float = 1.0,
) -> KernelSource:
    return replace(
        kernel_source,
        scaled_heat=_scaled_source_array(
            kernel_source.scaled_heat,
            offset,
            strength=strength,
            sign=1.0,
        ),
        scaled_current=_scaled_source_array(
            kernel_source.scaled_current,
            offset,
            strength=strength,
            sign=-0.7,
        ),
    )


def _source_with_update(
    base_boundary: KernelBoundary,
    base_source: KernelSource,
    update: str,
    offset: float,
) -> RuntimePoint:
    boundary = base_boundary
    if update == "ip":
        updated = _scale_ip(base_source, offset)
    elif update == "boundary":
        updated = base_source
        boundary = _scale_boundary(base_boundary, offset)
    elif update == "source":
        updated = _scale_source(base_source, offset)
    elif update == "mixed":
        updated = _scale_ip(base_source, offset, strength=0.5)
        boundary = _scale_boundary(base_boundary, offset, strength=0.5)
        updated = _scale_source(updated, offset, strength=0.5)
    else:
        raise ValueError(f"unknown continuation update {update!r}")
    return boundary, _with_case_suffix(updated, f"{update}-{offset:+.6g}")


def _policy_runtime_config(base_config: KernelConfig, policy: str) -> KernelConfig:
    initial_policy = _initial_policy_for_policy(policy)
    return replace(base_config, initial=initial_policy, continuation=policy)


def _initial_policy_for_policy(policy: str) -> str:
    return policy if policy in COLD_POLICIES else WARM_POLICY_INITIAL_POLICY


def _run_policy_sequence_once(
    case: GeqdskConfigCase,
    runtime_points: list[RuntimePoint],
    *,
    registry: KernelRegistry,
    policy: str,
) -> dict[str, Any]:
    solver = VEQlibSolver(
        case.topology,
        recipe=case.recipe,
        registry=registry,
        solver=case.kernel_config.method,
    )
    solver.metadata()  # force artifact load outside the timed sequence
    runtime_config = _policy_runtime_config(case.kernel_config, policy)
    started = time.perf_counter_ns()
    results: list[SolveResult] = []
    try:
        for kernel_boundary, kernel_source in runtime_points:
            solver.set_kernel_runtime(
                "" if kernel_source.case_name is None else kernel_source.case_name,
                *kernel_boundary.runtime_args(),
                *kernel_source.runtime_args(),
                *runtime_config.runtime_args(x_size=case.x_size),
            )
            results.append(SolveResult.from_solve_direct(solver.solve_direct()))
    finally:
        solver.close()
    wall_ms = float(time.perf_counter_ns() - started) / 1.0e6
    return {
        "wall_ms": wall_ms,
        "internal_ms": float(sum(result.elapsed_ms for result in results)),
        "effective_nfev": int(sum(result.nfev for result in results)),
        "njev": int(sum(result.njev for result in results)),
        "jacobian_component_evaluations": int(
            sum(result.jacobian_component_evaluations for result in results)
        ),
        "success_all": bool(all(result.success for result in results)),
        "point_nfev": [int(result.nfev) for result in results],
        "point_success": [bool(result.success) for result in results],
        "point_raw_norm": [float(result.raw_norm) for result in results],
        "max_raw_norm": (
            float(max(result.raw_norm for result in results)) if results else float("nan")
        ),
    }


def _measure_policy(
    case: GeqdskConfigCase,
    runtime_points: list[RuntimePoint],
    *,
    registry: KernelRegistry,
    repeat: int,
    warmup: int,
    policy: str,
) -> dict[str, Any]:
    for _ in range(warmup):
        _run_policy_sequence_once(case, runtime_points, registry=registry, policy=policy)

    samples = [
        _run_policy_sequence_once(case, runtime_points, registry=registry, policy=policy)
        for _ in range(repeat)
    ]
    last = samples[-1]
    return {
        "policy": policy,
        "initial_policy": _initial_policy_for_policy(policy),
        "continue_policy": policy,
        "wall_ms": float_stats([float(sample["wall_ms"]) for sample in samples]),
        "internal_ms": float_stats([float(sample["internal_ms"]) for sample in samples]),
        "effective_nfev": int_stats([int(sample["effective_nfev"]) for sample in samples]),
        "njev": int_stats([int(sample["njev"]) for sample in samples]),
        "jacobian_component_evaluations": int_stats(
            [int(sample["jacobian_component_evaluations"]) for sample in samples]
        ),
        "success_all": bool(all(sample["success_all"] for sample in samples)),
        "last_point_nfev": [int(value) for value in last["point_nfev"]],
        "last_point_success": [bool(value) for value in last["point_success"]],
        "last_point_raw_norm": [float(value) for value in last["point_raw_norm"]],
        "max_raw_norm": float(max(float(sample["max_raw_norm"]) for sample in samples)),
    }


def _measure_case(
    case: GeqdskConfigCase,
    *,
    registry: KernelRegistry,
    repeat: int,
    warmup: int,
    points: int,
    relative_span: float,
    update: str,
    policies: tuple[str, ...],
) -> dict[str, Any]:
    offsets = _scan_offsets(points=points, relative_span=relative_span)
    runtime_points = [
        _source_with_update(case.kernel_boundary, case.kernel_source, update, offset)
        for offset in offsets
    ]
    build_solver = VEQlibSolver(
        case.topology,
        recipe=case.recipe,
        registry=registry,
        solver=case.kernel_config.method,
    )
    build_start = time.perf_counter_ns()
    artifact = build_solver.compile(force=False, dry_run=False)
    build_wall_ms = float(time.perf_counter_ns() - build_start) / 1.0e6
    build_solver.close()

    measurements = {
        policy: _measure_policy(
            case,
            runtime_points,
            registry=registry,
            repeat=repeat,
            warmup=warmup,
            policy=policy,
        )
        for policy in policies
    }
    success_all = bool(all(measurement["success_all"] for measurement in measurements.values()))
    return {
        "status": "passed" if success_all else "failed",
        "case": case.case_key,
        "config": case.config_label,
        "row": case.row_label,
        "x_size": case.x_size,
        "signature": case.signature,
        "update": update,
        "experiment": UPDATE_LABELS[update],
        "offsets": offsets,
        "relative_span": float(relative_span),
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "reused": bool(artifact.reused),
            "build_wall_ms": build_wall_ms,
            "build_elapsed_ms": float(artifact.metadata["build"]["elapsed_ms"]),
        },
        "policies": measurements,
    }


def _selected_cases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.case:
        return tuple(dict.fromkeys(args.case))
    return CASE_KEYS


def _selected_configs(args: argparse.Namespace) -> tuple[str, ...]:
    if args.config:
        return tuple(dict.fromkeys(args.config))
    return ("Ref",)


def _selected_updates(args: argparse.Namespace) -> tuple[str, ...]:
    if args.update:
        return tuple(dict.fromkeys(args.update))
    return UPDATE_CHOICES


def _selected_spans(args: argparse.Namespace) -> tuple[float, ...]:
    if args.span:
        return tuple(float(value) for value in args.span)
    return DEFAULT_SPANS


def _selected_policies(args: argparse.Namespace) -> tuple[str, ...]:
    if args.policy:
        return tuple(dict.fromkeys(args.policy))
    return DEFAULT_POLICIES


def _mean_nfev(policy_payload: dict[str, Any]) -> float:
    return float(policy_payload["effective_nfev"]["mean"])


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = tuple(payload["policies"])
    for row in payload["rows"]:
        policy_values = {
            policy: _mean_nfev(row["policies"][policy])
            for policy in policies
            if row["policies"][policy]["success_all"]
        }
        if policy_values:
            best_policy = min(policy_values, key=policy_values.__getitem__)
            best_nfev = policy_values[best_policy]
        else:
            best_policy = "failed"
            best_nfev = float("nan")
        cold_nfev = float("nan")
        if "cold" in row["policies"] and row["policies"]["cold"]["success_all"]:
            cold_nfev = _mean_nfev(row["policies"]["cold"])
        warm_nfev = (
            _mean_nfev(row["policies"]["warm"]) if "warm" in row["policies"] else float("nan")
        )
        comparison = {
            "experiment": row["experiment"],
            "status": str(row.get("status", "passed" if policy_values else "failed")),
            "update": row["update"],
            "span": row["relative_span"],
            "case": row["case"],
            "config": row["config"],
            "x_size": row["x_size"],
            "best": best_policy,
            "best_nfev": best_nfev,
            "vs_cold": cold_nfev / best_nfev if best_nfev > 0 else float("nan"),
            "vs_warm": warm_nfev / best_nfev if best_nfev > 0 else float("nan"),
            "success_all": bool(all(row["policies"][policy]["success_all"] for policy in policies)),
        }
        for policy in policies:
            comparison[policy] = (
                _mean_nfev(row["policies"][policy])
                if row["policies"][policy]["success_all"]
                else float("nan")
            )
        rows.append(comparison)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, *, policies: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment",
        "case",
        *SUMMARY_POLICIES,
        "best",
        "vs_cold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_summary_csv_row(row))


def _format_nfev(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _format_policy_nfev(row: dict[str, Any], policy: str) -> str:
    return _format_nfev(float(row.get(policy, float("nan"))))


def _format_vs_cold(row: dict[str, Any]) -> str:
    if not np.isfinite(float(row.get("cold", float("nan")))) or not np.isfinite(
        float(row["vs_cold"])
    ):
        return "-"
    return f"{float(row['vs_cold']):.2f}x"


def _summary_csv_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment": str(row["experiment"]),
        "case": str(row["case"]),
        **{policy: _format_policy_nfev(row, policy) for policy in SUMMARY_POLICIES},
        "best": str(row["best"]),
        "vs_cold": _format_vs_cold(row),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path, *, policies: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_header = " | ".join(SUMMARY_POLICIES)
    policy_align = " | ".join("---:" for _ in SUMMARY_POLICIES)
    lines = [
        "# VEQlib Continuation nfev Benchmark",
        "",
        "The policy columns are mean effective nfev across repeats; effective nfev includes "
        "warm-start certification/predictor/chord residual evaluations and fallback solves.",
        "",
        f"| experiment | case | {policy_header} | best | vs cold |",
        f"|---|---|{policy_align}|---|---:|",
    ]
    for row in rows:
        policy_values = " | ".join(_format_policy_nfev(row, policy) for policy in SUMMARY_POLICIES)
        lines.append(
            "| {experiment} | {case} | {values} | {best} | {vs_cold} |".format(
                experiment=row["experiment"],
                case=row["case"],
                values=policy_values,
                best=row["best"],
                vs_cold=_format_vs_cold(row),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _progress_context(console: Console, *, quiet: bool) -> Any:
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn("[dim]{task.fields[current]:<32.32}[/]"),
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
    cases: tuple[str, ...],
    configs: tuple[str, ...],
    updates: tuple[str, ...],
    spans: tuple[float, ...],
    policies: tuple[str, ...],
    repeat: int,
    warmup: int,
    points: int,
) -> None:
    console.print(Text("[config]", style="bold cyan"))
    lines = (
        f"cases: [green]{', '.join(cases)}[/]",
        f"configs: [green]{', '.join(configs)}[/]",
        f"updates: [green]{', '.join(updates)}[/]",
        f"spans: [green]{', '.join(f'{span:g}' for span in spans)}[/]",
        f"policies: [green]{', '.join(policies)}[/]",
        f"points: [green]{points}[/]",
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


def _print_summary(
    console: Console,
    rows: list[dict[str, Any]],
    *,
    policies: tuple[str, ...],
) -> None:
    table = Table(
        box=REPORT_TABLE_BOX,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("experiment", no_wrap=True)
    table.add_column("case", no_wrap=True)
    for policy in SUMMARY_POLICIES:
        table.add_column(policy, justify="right")
    table.add_column("best", no_wrap=True)
    table.add_column("vs cold", justify="right")
    for row in rows:
        table.add_row(
            str(row["experiment"]),
            str(row["case"]),
            *(_format_policy_nfev(row, policy) for policy in SUMMARY_POLICIES),
            str(row["best"]),
            _format_vs_cold(row),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--config", action="append", choices=CONFIG_LABELS)
    parser.add_argument("--update", action="append", choices=UPDATE_CHOICES)
    parser.add_argument("--span", action="append", type=float, default=None)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--policy", action="append", choices=POLICY_CHOICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.points <= 0:
        raise ValueError("--points must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    selected_cases = set(_selected_cases(args))
    selected_configs = {config.lower() for config in _selected_configs(args)}
    updates = _selected_updates(args)
    spans = _selected_spans(args)
    policies = _selected_policies(args)
    cases = _make_cases(
        build=args.build,
        selected_cases=selected_cases,
        selected_configs=selected_configs,
    )
    cache_root = args.cache_root or default_kernel_cache_root()
    registry = KernelRegistry(cache_root=cache_root, source_dir=args.source_dir.resolve())

    console = _console()
    row_plan = [(update, span, case) for update in updates for span in spans for case in cases]
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        _print_config_tree(
            console,
            cases=tuple(sorted(selected_cases)),
            configs=tuple(_selected_configs(args)),
            updates=updates,
            spans=spans,
            policies=policies,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            points=int(args.points),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with _progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "continuation",
                total=len(row_plan),
                current="-",
                phase="[cyan]run[/]",
            )
        for update, span, case in row_plan:
            current = f"{update}:{span:g}:{case.row_label}"
            if progress is not None and task_id is not None:
                progress.update(task_id, current=current, phase="[cyan]run[/]")
            row = _measure_case(
                case,
                registry=registry,
                repeat=args.repeat,
                warmup=args.warmup,
                points=args.points,
                relative_span=span,
                update=update,
                policies=policies,
            )
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=_progress_phase(row["status"]))
                progress.advance(task_id)

    payload = {
        "schema": "veqlib.continuation_nfev.v1",
        "metric": "effective_nfev",
        "metric_note": (
            "effective_nfev is native result.nfev, including initial residual probes, "
            "warm-start certificates/predictors/chord attempts, fallback nonlinear solves, "
            "and final residual checks."
        ),
        "build": str(args.build),
        "cases": sorted(selected_cases),
        "configs": list(_selected_configs(args)),
        "updates": list(updates),
        "spans": [float(span) for span in spans],
        "points": int(args.points),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "policies": list(policies),
        "warm_alias": "warm-fixed",
        "cache_root": str(cache_root),
        "source_dir": str(args.source_dir.resolve()),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "rows": rows,
    }
    comparison_rows = _comparison_rows(payload)
    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = args.output_dir / "raw_results.json"
        csv_path = args.output_dir / "summary.csv"
        md_path = args.output_dir / "summary.md"
        write_json(raw_path, payload)
        _write_csv(comparison_rows, csv_path, policies=policies)
        _write_markdown(comparison_rows, md_path, policies=policies)
        console.print()
        _print_outputs_tree(console, {"json": raw_path, "csv": csv_path, "md": md_path})
    _print_summary(console, comparison_rows, policies=policies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
