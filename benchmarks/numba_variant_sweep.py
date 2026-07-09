#!/usr/bin/env python3
"""Numba Kernel.variant() construction-cost benchmark on GEQDSK Pareto cases."""

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
    CONFIG_LABELS,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REPO_ROOT,
    RouteBenchmarkSpec,
    cpu_affinity,
    geqdsk_kernel_case,
    geqdsk_signature,
    runtime_env,
    selected_cases,
    selected_configs,
    topology_profile_counts,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_speedup,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import Kernel, KernelRecipe, KernelTopology

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "numba_variant_sweep.json"


def _measure_case_sweep(
    args: argparse.Namespace,
    case_key: str,
    config_labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    route_spec = RouteBenchmarkSpec("PF", "psin", "uniform", "Ip")
    capacity_case = geqdsk_kernel_case(case_key, "Ref", route_spec=route_spec)
    target_cases = {
        config_label: geqdsk_kernel_case(case_key, config_label, route_spec=route_spec)
        for config_label in config_labels
    }
    rows = [
        _plan_row(
            case_key,
            config_label,
            capacity_case.topology,
            target_cases[config_label].topology,
        )
        for config_label in config_labels
    ]
    if args.no_run:
        return rows

    kernel = None
    try:
        kernel = Kernel(
            topology=capacity_case.topology,
            recipe=KernelRecipe(backend="numba", layout="degree"),
            config=capacity_case.config,
        )
        initial_workspace_ids = _workspace_ids(kernel)
        for row in rows:
            target_case = target_cases[str(row["config"])]
            started = time.perf_counter_ns()
            kernel.variant(**_variant_kwargs(target_case.topology))
            variant_ms = float(time.perf_counter_ns() - started) / 1.0e6
            workspace_reused = _workspace_ids(kernel) == initial_workspace_ids

            started = time.perf_counter_ns()
            new_kernel = Kernel(
                topology=target_case.topology,
                recipe=KernelRecipe(backend="numba", layout="degree"),
                config=target_case.config,
            )
            new_ms = float(time.perf_counter_ns() - started) / 1.0e6
            new_kernel.close()

            row["runtime"] = _runtime_payload(
                target_case.topology,
                variant_ms=variant_ms,
                new_ms=new_ms,
                workspace_reused=workspace_reused,
            )
    except Exception as exc:
        for row in rows:
            if row.get("runtime", {}).get("status") == "not_requested":
                row["runtime"] = {
                    "status": "failed",
                    "failure_reason": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return rows
    finally:
        if kernel is not None:
            kernel.close()
    return rows


def _plan_row(
    case_key: str,
    config_label: str,
    capacity_topology: KernelTopology,
    active_topology: KernelTopology,
) -> dict[str, Any]:
    return {
        "case": case_key,
        "config": config_label,
        "row": f"{case_key}:{config_label.lower()}",
        "signature": geqdsk_signature(case_key, config_label),
        "capacity": _topology_payload(capacity_topology),
        "active": _topology_payload(active_topology),
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


def _runtime_payload(
    topology: KernelTopology,
    *,
    variant_ms: float,
    new_ms: float,
    workspace_reused: bool,
) -> dict[str, Any]:
    ratio = variant_ms / new_ms if new_ms > 0.0 else float("nan")
    speedup = new_ms / variant_ms if variant_ms > 0.0 else float("nan")
    payload: dict[str, Any] = {
        "status": "passed" if workspace_reused else "failed",
        "x_size": int(topology.x_size),
        "variant_ms": float(variant_ms),
        "new_ms": float(new_ms),
        "variant_to_new_ratio": float(ratio),
        "new_to_variant_speedup": float(speedup),
        "workspace_reused": bool(workspace_reused),
    }
    if not workspace_reused:
        payload["failure_reason"] = "workspace_not_reused"
    return payload


def _variant_kwargs(topology: KernelTopology) -> dict[str, Any]:
    return {
        "h_count": topology.h_count,
        "v_count": topology.v_count,
        "kappa_count": topology.kappa_count,
        "psin_count": topology.psin_count,
        "F_count": topology.F_count,
        "c_counts": topology.c_counts,
        "s_counts": topology.s_counts,
    }


def _workspace_ids(kernel: Kernel) -> tuple[int, int, int, int]:
    runtime = kernel._impl._solver.runtime  # type: ignore[attr-defined]
    return (
        id(runtime.profile_workspace),
        id(runtime.geometry_workspace),
        id(runtime.source_workspace),
        id(runtime.residual_workspace),
    )


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
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("case", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("config", no_wrap=True)
    table.add_column("x", justify="right")
    table.add_column(Text("variant (ms)"), justify="right")
    table.add_column(Text("new (ms)"), justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("reuse", justify="right")
    for row in rows:
        runtime = row.get("runtime", {})
        table.add_row(
            str(row.get("case", "n/a")),
            status_cell(runtime.get("status", "n/a")),
            str(row.get("config", "n/a")),
            str(runtime.get("x_size", row.get("active", {}).get("x_size", "n/a"))),
            format_optional_float(runtime.get("variant_ms")),
            format_optional_float(runtime.get("new_ms")),
            format_optional_speedup(runtime.get("new_ms"), runtime.get("variant_ms")),
            "yes" if runtime.get("workspace_reused") is True else "no",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--config", action="append", choices=CONFIG_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    console = reporting_console()
    case_keys = selected_cases(args.case)
    config_labels = selected_configs(args.config)
    row_plan = [
        (case_key, config_label) for case_key in case_keys for config_label in config_labels
    ]
    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{len(row_plan)}[/]",
                "backend: [green]numba[/]",
                "metric: [green]variant switch vs new Kernel construction[/]",
                "capacity: [green]Ref topology per GEQDSK case[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "variant-sweep",
                total=len(row_plan),
                current="-",
                phase="[cyan]run[/]",
            )
        for case_key in case_keys:
            case_rows = _measure_case_sweep(args, case_key, config_labels)
            rows.extend(case_rows)
            if progress is not None and task_id is not None:
                for row in case_rows:
                    progress.update(
                        task_id,
                        current=str(row.get("row", "-")),
                        phase=progress_phase(row.get("runtime", {}).get("status")),
                    )
                    progress.advance(task_id)

    summary = _summary(rows)
    payload = {
        "schema": "veqpy.numba.variant_construction_geqdsk.v1",
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "timing_note": (
            "Single-pass construction benchmark. variant_ms measures Kernel.variant() "
            "on one existing Ref-capacity Numba handle; new_ms measures constructing "
            "a fresh Numba Kernel for the same active topology. Solves are excluded."
        ),
        "layout": {
            "Nr": REFERENCE_LAYOUT_NR,
            "Nt": REFERENCE_LAYOUT_NT,
            "capacity_config": "Ref",
        },
        "summary": summary,
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
