#!/usr/bin/env python3
"""Run the production certified-continuation benchmark suite."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "veqlib" / "artifact" / "continuation"
DEFAULT_MPLCONFIG = Path("/tmp/veqpy-mpl")

REAL_CASE_CHOICES = ("solovev", "chease", "efit")
CASE_CHOICES = ("PF_psin_uniform_Ip",) + REAL_CASE_CHOICES
UPDATE_CHOICES = ("ip", "boundary", "source", "mixed")
DEFAULT_SPANS = (0.0002, 0.005, 0.01, 0.05, 0.20)
PRODUCTION_POLICY = "certified-continuation"
BENCHMARK_POLICIES = ("cold", "warm-clone", PRODUCTION_POLICY)

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veqlib.benchmark_4case_compare import _make_cases  # noqa: E402
from veqpy.cpp import KernelRegistry, VEQlibSolver, solve_payload_sequence  # noqa: E402


def _quantile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    return values_sorted[int((len(values_sorted) - 1) * q)]


def _float_stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "p05": float(_quantile(values, 0.05)),
        "p95": float(_quantile(values, 0.95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _int_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def _scan_offsets(*, points: int, relative_span: float) -> list[float]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return [0.0]
    lower = -0.5 * relative_span
    step = relative_span / float(points - 1)
    return [float(lower + step * index) for index in range(points)]


def _scale_ip(payload: dict[str, Any], offset: float, *, strength: float = 1.0) -> None:
    constraints = payload["constraints"]
    base_ip = float(constraints["scaled_Ip"])
    constraints["scaled_Ip"] = float(base_ip * (1.0 + strength * offset))


def _scale_boundary(payload: dict[str, Any], offset: float, *, strength: float = 1.0) -> None:
    boundary = payload["boundary"]
    for key in ("c_offsets", "s_offsets"):
        if key not in boundary:
            continue
        updated: list[float] = []
        for index, value in enumerate(boundary[key]):
            value_f = float(value)
            if key == "s_offsets" and index == 0:
                updated.append(value_f)
                continue
            direction = 1.0 if index % 2 == 0 else -1.0
            weight = 1.0 / float(index + 1)
            updated.append(float(value_f * (1.0 + strength * offset * direction * weight)))
        boundary[key] = updated


def _scale_source(payload: dict[str, Any], offset: float, *, strength: float = 1.0) -> None:
    source = payload["source"]
    for key, sign in (("scaled_heat", 1.0), ("scaled_current", -0.7)):
        values = source[key]
        count = len(values)
        if count <= 1:
            source[key] = [float(value) for value in values]
            continue
        updated: list[float] = []
        for index, value in enumerate(values):
            rho = float(index) / float(count - 1)
            smooth_shape = 0.55 * (2.0 * rho - 1.0) + 0.45 * math.cos(math.pi * rho)
            factor = 1.0 + strength * offset * sign * smooth_shape
            updated.append(float(value) * factor)
        source[key] = updated


def _payload_with_update(base_payload: dict[str, Any], update: str, offset: float) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    if update == "ip":
        _scale_ip(payload, offset)
    elif update == "boundary":
        _scale_boundary(payload, offset)
    elif update == "source":
        _scale_source(payload, offset)
    elif update == "mixed":
        _scale_ip(payload, offset, strength=0.5)
        _scale_boundary(payload, offset, strength=0.5)
        _scale_source(payload, offset, strength=0.5)
    else:
        raise ValueError(f"unknown continuation update {update!r}")
    return payload


def _measure_policy(
    solver: VEQlibSolver,
    payloads: list[dict[str, Any]],
    *,
    repeat: int,
    warmup: int,
    policy: str,
) -> dict[str, Any]:
    first_policy = "cold"
    continuation_policy = "cold" if policy == "cold" else policy

    for _ in range(warmup):
        solve_payload_sequence(
            solver,
            payloads,
            first_policy=first_policy,
            continuation_policy=continuation_policy,
        )

    wall_ms: list[float] = []
    internal_ms: list[float] = []
    total_solver_nfev: list[int] = []
    total_raw_evals: list[int] = []
    all_success: list[bool] = []
    last_steps = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        steps = solve_payload_sequence(
            solver,
            payloads,
            first_policy=first_policy,
            continuation_policy=continuation_policy,
        )
        wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        internal_ms.append(sum(step.elapsed_ms for step in steps))
        total_solver_nfev.append(sum(step.solver_nfev for step in steps))
        total_raw_evals.append(sum(step.total_raw_residual_evaluations for step in steps))
        all_success.append(all(step.success for step in steps))
        last_steps = steps

    accepted_by = [step.accepted_by for step in last_steps]
    raw_norms = [step.raw_norm for step in last_steps]
    return {
        "policy": policy,
        "first_policy": first_policy,
        "continuation_policy": continuation_policy,
        "wall_ms": _float_stats(wall_ms),
        "internal_ms": _float_stats(internal_ms),
        "total_solver_nfev": _int_stats(total_solver_nfev),
        "total_raw_residual_evaluations": _int_stats(total_raw_evals),
        "accepted_by_counts": dict(Counter(accepted_by)),
        "last_point_accepted_by": accepted_by,
        "last_point_raw_norm": raw_norms,
        "last_point_success": [step.success for step in last_steps],
        "max_raw_norm": float(max(raw_norms)) if raw_norms else float("nan"),
        "success_all": all(all_success),
    }


def _measure_case(
    case_data: Any,
    *,
    registry: KernelRegistry,
    repeat: int,
    warmup: int,
    points: int,
    relative_span: float,
    update: str,
    policies: tuple[str, ...],
) -> dict[str, Any]:
    base_payload = json.loads(case_data.payload_json)
    offsets = _scan_offsets(points=points, relative_span=relative_span)
    payloads = [_payload_with_update(base_payload, update, offset) for offset in offsets]

    solver = VEQlibSolver(case_data.topology, registry=registry, solver="powell")
    artifact = solver.build(force=False, dry_run=False)
    measurements = {
        policy: _measure_policy(solver, payloads, repeat=repeat, warmup=warmup, policy=policy)
        for policy in policies
    }
    return {
        "case": case_data.name,
        "x_size": case_data.x_size,
        "metadata": case_data.metadata,
        "update": update,
        "offsets": offsets,
        "relative_span": float(relative_span),
        "artifact": {"artifact_id": artifact.artifact_id, "reused": bool(artifact.reused)},
        "policies": measurements,
    }


def _selected_cases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.case:
        return tuple(dict.fromkeys(args.case))
    return REAL_CASE_CHOICES


def _selected_updates(args: argparse.Namespace) -> tuple[str, ...]:
    if args.update:
        return tuple(dict.fromkeys(args.update))
    return UPDATE_CHOICES


def _accepted_summary(counts: dict[str, int]) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload["rows"]:
        cold = row["policies"]["cold"]
        warm = row["policies"]["warm-clone"]
        production = row["policies"][PRODUCTION_POLICY]
        production_wall = float(production["wall_ms"]["mean"])
        rows.append(
            {
                "update": row["update"],
                "span": row["relative_span"],
                "case": row["case"],
                "x_size": row["x_size"],
                "cold_wall_ms": cold["wall_ms"]["mean"],
                "warm_wall_ms": warm["wall_ms"]["mean"],
                "optimized_wall_ms": production_wall,
                "speedup_vs_cold": float(cold["wall_ms"]["mean"]) / production_wall,
                "speedup_vs_warm": float(warm["wall_ms"]["mean"]) / production_wall,
                "cold_solver_nfev": cold["total_solver_nfev"]["mean"],
                "warm_solver_nfev": warm["total_solver_nfev"]["mean"],
                "optimized_solver_nfev": production["total_solver_nfev"]["mean"],
                "cold_raw_eval": cold["total_raw_residual_evaluations"]["mean"],
                "warm_raw_eval": warm["total_raw_residual_evaluations"]["mean"],
                "optimized_raw_eval": production["total_raw_residual_evaluations"]["mean"],
                "optimized_accepted_by": _accepted_summary(production["accepted_by_counts"]),
                "optimized_max_raw_norm": production["max_raw_norm"],
                "success_all": (
                    cold["success_all"] and warm["success_all"] and production["success_all"]
                ),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "update",
        "span",
        "case",
        "x_size",
        "cold_wall_ms",
        "warm_wall_ms",
        "optimized_wall_ms",
        "speedup_vs_cold",
        "speedup_vs_warm",
        "cold_solver_nfev",
        "warm_solver_nfev",
        "optimized_solver_nfev",
        "cold_raw_eval",
        "warm_raw_eval",
        "optimized_raw_eval",
        "optimized_accepted_by",
        "optimized_max_raw_norm",
        "success_all",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Certified Continuation Benchmark",
        "",
        "| update | span | case | cold ms | warm ms | optimized ms | vs cold | vs warm | optimized nfev | optimized raw eval | accepted_by | max raw norm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {update} | {span:g} | {case} | {cold:.3f} | {warm:.3f} | {opt:.3f} | "
            "{cold_speed:.2f}x | {warm_speed:.2f}x | {nfev:.1f} | {raw:.1f} | "
            "{accepted} | {norm:.2e} |".format(
                update=row["update"],
                span=float(row["span"]),
                case=row["case"],
                cold=float(row["cold_wall_ms"]),
                warm=float(row["warm_wall_ms"]),
                opt=float(row["optimized_wall_ms"]),
                cold_speed=float(row["speedup_vs_cold"]),
                warm_speed=float(row["speedup_vs_warm"]),
                nfev=float(row["optimized_solver_nfev"]),
                raw=float(row["optimized_raw_eval"]),
                accepted=row["optimized_accepted_by"],
                norm=float(row["optimized_max_raw_norm"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "| update | span | case | cold ms | warm ms | optimized ms | "
        "vs cold | vs warm | optimized nfev | optimized raw eval | accepted_by | max raw norm |"
    )
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in rows:
        print(
            "| {update} | {span:g} | {case} | {cold:.3f} | {warm:.3f} | {opt:.3f} | "
            "{cold_speed:.2f}x | {warm_speed:.2f}x | {nfev:.1f} | {raw:.1f} | "
            "{accepted} | {norm:.2e} |".format(
                update=row["update"],
                span=float(row["span"]),
                case=row["case"],
                cold=float(row["cold_wall_ms"]),
                warm=float(row["warm_wall_ms"]),
                opt=float(row["optimized_wall_ms"]),
                cold_speed=float(row["speedup_vs_cold"]),
                warm_speed=float(row["speedup_vs_warm"]),
                nfev=float(row["optimized_solver_nfev"]),
                raw=float(row["optimized_raw_eval"]),
                accepted=row["optimized_accepted_by"],
                norm=float(row["optimized_max_raw_norm"]),
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_CHOICES)
    parser.add_argument("--update", action="append", choices=UPDATE_CHOICES)
    parser.add_argument("--span", action="append", type=float, default=None)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = _selected_cases(args)
    updates = _selected_updates(args)
    spans = tuple(args.span) if args.span else DEFAULT_SPANS
    policies = BENCHMARK_POLICIES

    output_dir = args.output_dir
    cache_root = args.cache_root or output_dir / "cache"
    registry = KernelRegistry(cache_root=cache_root, source_dir=REPO_ROOT / "veqlib")
    case_data = _make_cases(repeat=1, warmup=0, selected=set(cases))

    rows = []
    for update in updates:
        for span in spans:
            for case in case_data:
                print(f"measuring update={update} span={span:g} case={case.name} ...", flush=True)
                rows.append(
                    _measure_case(
                        case,
                        registry=registry,
                        repeat=args.repeat,
                        warmup=args.warmup,
                        points=args.points,
                        relative_span=span,
                        update=update,
                        policies=policies,
                    )
                )

    payload = {
        "schema": "veqlib.certified_continuation_suite.v1",
        "cases": list(cases),
        "updates": list(updates),
        "spans": [float(span) for span in spans],
        "policies": list(policies),
        "production_policy": PRODUCTION_POLICY,
        "points": int(args.points),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cache": str(cache_root),
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.json"
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison_rows = _comparison_rows(payload)
    _write_csv(comparison_rows, csv_path)
    _write_markdown(comparison_rows, md_path)

    print(f"json: {raw_path}")
    print(f"csv : {csv_path}")
    print(f"md  : {md_path}")
    print(
        f"cases={','.join(cases)} updates={','.join(updates)} "
        f"spans={','.join(f'{span:g}' for span in spans)} policy={','.join(policies)}"
    )
    _print_summary(comparison_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
