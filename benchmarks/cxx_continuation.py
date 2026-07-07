#!/usr/bin/env python3
"""Continuation-policy benchmark for Kernel handles."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
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
    REFERENCE_SOLVER_MAXFEV,
    REPO_ROOT,
    RouteBenchmarkSpec,
    continuation_points,
    cpu_affinity,
    default_kernel_cache_root,
    float_stats,
    geqdsk_kernel_case,
    int_stats,
    runtime_env,
    selected_cases,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import Kernel, KernelConfig, KernelRecipe

DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "cxx_continuation"
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
DEFAULT_POLICIES = POLICY_CHOICES
SUMMARY_POLICIES = ("cold", "warm-fixed", "warm-predict", "warm-chord")
UPDATE_LABELS = {
    "ip": "C1 Ip",
    "boundary": "C2 boundary",
    "source": "C3 source",
    "mixed": "C4 mixed",
}


def _policy_config(base: KernelConfig, policy: str) -> KernelConfig:
    if policy.startswith("cold"):
        return replace(base, initial=policy, continuation=policy)
    return replace(base, initial="cold", continuation=policy)


def _scan_offsets(*, points: int, span: float) -> list[float]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return [0.0]
    lower = -0.5 * float(span)
    step = float(span) / float(points - 1)
    return [float(lower + step * index) for index in range(points)]


def _sequence_once(points, *, recipe: KernelRecipe, config: KernelConfig) -> dict[str, Any]:
    started = time.perf_counter_ns()
    total_nfev = 0
    total_njev = 0
    total_jacobian_components = 0
    all_success = True
    point_nfev: list[int] = []
    point_success: list[bool] = []
    point_raw_norm: list[float] = []
    first = points[0]
    kernel = Kernel(topology=first.topology, recipe=recipe, config=config)
    try:
        for point in points:
            point = replace(point, config=config)
            result = kernel.solve(point.boundary, point.source, config=config)
            total_nfev += int(result.nfev)
            total_njev += int(result.njev)
            total_jacobian_components += int(result.jacobian_component_evaluations)
            all_success = all_success and bool(result.success)
            point_nfev.append(int(result.nfev))
            point_success.append(bool(result.success))
            point_raw_norm.append(float(result.raw_norm))
        elapsed_ms = float(time.perf_counter_ns() - started) / 1.0e6
        return {
            "wall_ms": elapsed_ms,
            "internal_ms": elapsed_ms,
            "effective_nfev": total_nfev,
            "njev": total_njev,
            "jacobian_component_evaluations": total_jacobian_components,
            "success_all": all_success,
            "point_nfev": point_nfev,
            "point_success": point_success,
            "point_raw_norm": point_raw_norm,
            "max_raw_norm": max(point_raw_norm) if point_raw_norm else float("nan"),
        }
    finally:
        kernel.close()


def _measure_policy_payload(
    points,
    *,
    recipe: KernelRecipe,
    base_config: KernelConfig,
    policy: str,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    config = _policy_config(base_config, policy)
    for _ in range(max(0, int(warmup))):
        _sequence_once(points, recipe=recipe, config=config)
    samples = [
        _sequence_once(points, recipe=recipe, config=config) for _ in range(max(1, int(repeat)))
    ]
    last = samples[-1]
    return {
        "policy": policy,
        "initial_policy": config.initial,
        "continue_policy": config.continuation,
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
    args: argparse.Namespace,
    case_key: str,
    config_label: str,
    update: str,
    span: float,
    policies: tuple[str, ...],
) -> dict[str, Any]:
    base = geqdsk_kernel_case(
        case_key,
        config_label,
        route_spec=RouteBenchmarkSpec("PF", "psin", "uniform", "Ip"),
        method=args.method,
        max_evaluations=args.max_evaluations,
        initial="cold",
        norm=args.norm,
    )
    points = continuation_points(base, update=update, span=span, points=args.points)
    offsets = _scan_offsets(points=args.points, span=span)
    recipe = KernelRecipe(backend="cxx", build=args.build, layout="degree")
    if args.no_run:
        return {
            "case": case_key,
            "config": config_label,
            "update": update,
            "status": "planned",
            "points": len(points),
            "x_size": base.topology.x_size,
        }
    measurements = {
        policy: _measure_policy_payload(
            points,
            recipe=recipe,
            base_config=base.config,
            policy=policy,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        for policy in policies
    }
    success_all = bool(all(measurement["success_all"] for measurement in measurements.values()))
    return {
        "status": "passed" if success_all else "failed",
        "case": case_key,
        "config": config_label,
        "row": f"{case_key}:{config_label.lower()}",
        "update": update,
        "experiment": UPDATE_LABELS[update],
        "relative_span": float(span),
        "offsets": offsets,
        "points": len(points),
        "x_size": base.topology.x_size,
        "signature": geqdsk_signature_for_continuation(case_key, config_label),
        "policies": measurements,
    }


def geqdsk_signature_for_continuation(case_key: str, config_label: str) -> dict[str, int]:
    from benchmarks._common import geqdsk_signature

    return geqdsk_signature(case_key, config_label)


def _print_summary(console, rows: list[dict[str, Any]], *, policies: tuple[str, ...]) -> None:
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
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


def _selected_configs_for_continuation(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else ("Ref",)


def _selected_spans(values: list[float] | None) -> tuple[float, ...]:
    return tuple(float(value) for value in values) if values else DEFAULT_SPANS


def _mean_nfev(policy_payload: dict[str, Any]) -> float:
    return float(policy_payload["effective_nfev"]["mean"])


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = tuple(payload["policies"])
    for row in payload["rows"]:
        if row.get("status") == "planned":
            continue
        policy_values = {
            policy: _mean_nfev(row["policies"][policy])
            for policy in policies
            if row["policies"][policy]["success_all"]
        }
        best_policy = (
            min(policy_values, key=policy_values.__getitem__) if policy_values else "failed"
        )
        best_nfev = policy_values.get(best_policy, float("nan"))
        cold_nfev = (
            _mean_nfev(row["policies"]["cold"])
            if "cold" in row["policies"] and row["policies"]["cold"]["success_all"]
            else float("nan")
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


def _format_policy_nfev(row: dict[str, Any], policy: str) -> str:
    return _format_nfev(float(row.get(policy, float("nan"))))


def _format_nfev(value: float) -> str:
    if value != value:
        return "-"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _format_vs_cold(row: dict[str, Any]) -> str:
    value = float(row.get("vs_cold", float("nan")))
    if value != value:
        return "-"
    return f"{value:.2f}x"


def _summary_csv_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment": str(row["experiment"]),
        "case": str(row["case"]),
        **{policy: _format_policy_nfev(row, policy) for policy in SUMMARY_POLICIES},
        "best": str(row["best"]),
        "vs_cold": _format_vs_cold(row),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["experiment", "case", *SUMMARY_POLICIES, "best", "vs_cold"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_summary_csv_row(row))


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_header = " | ".join(SUMMARY_POLICIES)
    policy_align = " | ".join("---:" for _ in SUMMARY_POLICIES)
    lines = [
        "# Cxx Continuation nfev Benchmark",
        "",
        "The policy columns are mean effective nfev across repeats.",
        "",
        f"| experiment | case | {policy_header} | best | vs cold |",
        f"|---|---|{policy_align}|---|---:|",
    ]
    for row in rows:
        policy_values = " | ".join(_format_policy_nfev(row, policy) for policy in SUMMARY_POLICIES)
        lines.append(
            f"| {row['experiment']} | {row['case']} | {policy_values} | "
            f"{row['best']} | {_format_vs_cold(row)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--method", default="powell")
    parser.add_argument("--norm", default="fast")
    parser.add_argument("--max-evaluations", type=int, default=REFERENCE_SOLVER_MAXFEV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.points <= 0:
        raise ValueError("--points must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = reporting_console()
    case_keys = selected_cases(args.case)
    config_labels = _selected_configs_for_continuation(args.config)
    updates = tuple(args.update) if args.update else UPDATE_CHOICES
    spans = _selected_spans(args.span)
    policies = tuple(args.policy) if args.policy else DEFAULT_POLICIES
    row_plan = [
        (case_key, config_label, update, span)
        for case_key in case_keys
        for config_label in config_labels
        for update in updates
        for span in spans
    ]
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{', '.join(case_keys)}[/]",
                f"configs: [green]{', '.join(config_labels)}[/]",
                f"updates: [green]{', '.join(updates)}[/]",
                f"spans: [green]{', '.join(f'{span:g}' for span in spans)}[/]",
                f"policies: [green]{', '.join(policies)}[/]",
                f"points: [green]{args.points}[/]",
                f"warmup: [green]{args.warmup}[/]",
                f"repeat: [green]{args.repeat}[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with progress_context(console, quiet=args.quiet_progress, width=32) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "continuation",
                total=len(row_plan),
                current="-",
                phase="[cyan]run[/]",
            )
        for case_key, config_label, update, span in row_plan:
            current = f"{update}:{span:g}:{case_key}:{config_label.lower()}"
            if progress is not None and task_id is not None:
                progress.update(task_id, current=current, phase="[cyan]run[/]")
            row = _measure_case(args, case_key, config_label, update, span, policies)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=progress_phase(row.get("status")))
                progress.advance(task_id)
    payload = {
        "schema": "veqpy.cxx.continuation_nfev.v1",
        "metric": "effective_nfev",
        "metric_note": (
            "effective_nfev is the summed Kernel SolveResult.nfev "
            "over the continuation sequence."
        ),
        "build": str(args.build),
        "cases": list(case_keys),
        "configs": list(config_labels),
        "updates": list(updates),
        "spans": [float(span) for span in spans],
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "points": int(args.points),
        "policies": list(policies),
        "warm_alias": "warm-fixed",
        "cache_root": str(args.cache_root or default_kernel_cache_root()),
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
        _write_csv(comparison_rows, csv_path)
        _write_markdown(comparison_rows, md_path)
    if not args.quiet_progress:
        if not args.no_write:
            console.print()
            print_outputs_tree(
                console,
                {"json": raw_path, "csv": csv_path, "md": md_path},
                repo_root=REPO_ROOT,
            )
    _print_summary(console, comparison_rows, policies=policies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
