#!/usr/bin/env python3
"""Continuation-policy benchmark for VEQlib using effective nfev.

This benchmark ports the remote certified-continuation sweep onto the current
``veqlib.benchmarks`` package and the typed facade API.  The primary metric is
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from veqlib.benchmarks._common import (
    CORE_DIR,
    REPO_ROOT,
    cpu_affinity,
    float_stats,
    int_stats,
    runtime_env,
    write_json,
)
from veqlib.benchmarks.benchmark_geqdsk import GeqdskConfigCase, _make_cases
from veqlib.facade import (
    KernelInput,
    KernelRegistry,
    KernelResult,
    KernelSolve,
    VEQlibSolver,
    default_kernel_cache_root,
)

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import CASE_KEYS, CONFIG_LABELS  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "veqlib" / "benchmarks" / "results" / "continuation_nfev"
UPDATE_CHOICES = ("ip", "boundary", "source", "mixed")
DEFAULT_SPANS = (0.0002, 0.005, 0.01, 0.05, 0.20)
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
DEFAULT_POLICIES = POLICY_CHOICES
UPDATE_LABELS = {
    "ip": "C1 Ip",
    "boundary": "C2 boundary",
    "source": "C3 source",
    "mixed": "C4 mixed",
}


def _scan_offsets(*, points: int, relative_span: float) -> list[float]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return [0.0]
    lower = -0.5 * float(relative_span)
    step = float(relative_span) / float(points - 1)
    return [float(lower + step * index) for index in range(points)]


def _with_case_suffix(kernel_input: KernelInput, suffix: str) -> KernelInput:
    case_name = kernel_input.case_name or "case"
    return replace(kernel_input, case_name=f"{case_name}-{suffix}")


def _scale_ip(kernel_input: KernelInput, offset: float, *, strength: float = 1.0) -> KernelInput:
    scaled_ip = float(kernel_input.scaled_Ip)
    if not np.isfinite(scaled_ip):
        raise ValueError("ip continuation update requires finite scaled_Ip")
    return replace(kernel_input, scaled_Ip=float(scaled_ip * (1.0 + strength * offset)))


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
    kernel_input: KernelInput,
    offset: float,
    *,
    strength: float = 1.0,
) -> KernelInput:
    boundary = kernel_input.boundary
    updated_boundary = replace(
        boundary,
        c_offsets=_scaled_boundary_array(boundary.c_offsets, offset, strength=strength, sine=False),
        s_offsets=_scaled_boundary_array(boundary.s_offsets, offset, strength=strength, sine=True),
    )
    return replace(kernel_input, boundary=updated_boundary)


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
    kernel_input: KernelInput,
    offset: float,
    *,
    strength: float = 1.0,
) -> KernelInput:
    return replace(
        kernel_input,
        scaled_heat=_scaled_source_array(
            kernel_input.scaled_heat,
            offset,
            strength=strength,
            sign=1.0,
        ),
        scaled_current=_scaled_source_array(
            kernel_input.scaled_current,
            offset,
            strength=strength,
            sign=-0.7,
        ),
    )


def _input_with_update(base_input: KernelInput, update: str, offset: float) -> KernelInput:
    if update == "ip":
        updated = _scale_ip(base_input, offset)
    elif update == "boundary":
        updated = _scale_boundary(base_input, offset)
    elif update == "source":
        updated = _scale_source(base_input, offset)
    elif update == "mixed":
        updated = _scale_ip(base_input, offset, strength=0.5)
        updated = _scale_boundary(updated, offset, strength=0.5)
        updated = _scale_source(updated, offset, strength=0.5)
    else:
        raise ValueError(f"unknown continuation update {update!r}")
    return _with_case_suffix(updated, f"{update}-{offset:+.6g}")


def _policy_runtime_solve(base_solve: KernelSolve, policy: str) -> KernelSolve:
    if policy in COLD_POLICIES:
        return replace(base_solve, initial=policy, continuation=policy)
    return replace(base_solve, initial="cold", continuation=policy)


def _run_policy_sequence_once(
    case: GeqdskConfigCase,
    inputs: list[KernelInput],
    *,
    registry: KernelRegistry,
    policy: str,
) -> dict[str, Any]:
    solver = VEQlibSolver(case.topology, registry=registry, solver=case.kernel_solve.method)
    solver.metadata()  # force artifact load outside the timed sequence
    solve_policy = _policy_runtime_solve(case.kernel_solve, policy)
    started = time.perf_counter_ns()
    results: list[KernelResult] = []
    try:
        for kernel_input in inputs:
            solver.set_kernel_runtime(
                *kernel_input.runtime_args(),
                *solve_policy.runtime_args(x_size=case.x_size),
            )
            results.append(KernelResult.from_solve_direct(solver.solve_direct()))
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
    inputs: list[KernelInput],
    *,
    registry: KernelRegistry,
    repeat: int,
    warmup: int,
    policy: str,
) -> dict[str, Any]:
    for _ in range(warmup):
        _run_policy_sequence_once(case, inputs, registry=registry, policy=policy)

    samples = [
        _run_policy_sequence_once(case, inputs, registry=registry, policy=policy)
        for _ in range(repeat)
    ]
    last = samples[-1]
    return {
        "policy": policy,
        "initial_policy": policy if policy in COLD_POLICIES else "cold",
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
    inputs = [_input_with_update(case.kernel_input, update, offset) for offset in offsets]
    build_solver = VEQlibSolver(case.topology, registry=registry, solver=case.kernel_solve.method)
    build_start = time.perf_counter_ns()
    artifact = build_solver.build(force=False, dry_run=False)
    build_wall_ms = float(time.perf_counter_ns() - build_start) / 1.0e6
    build_solver.close()

    measurements = {
        policy: _measure_policy(
            case,
            inputs,
            registry=registry,
            repeat=repeat,
            warmup=warmup,
            policy=policy,
        )
        for policy in policies
    }
    return {
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
        cold_nfev = (
            _mean_nfev(row["policies"]["cold"]) if "cold" in row["policies"] else float("nan")
        )
        warm_nfev = (
            _mean_nfev(row["policies"]["warm"]) if "warm" in row["policies"] else float("nan")
        )
        comparison = {
            "experiment": row["experiment"],
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
            comparison[policy] = _mean_nfev(row["policies"][policy])
        rows.append(comparison)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, *, policies: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment",
        "update",
        "span",
        "case",
        "config",
        "x_size",
        *policies,
        "best",
        "best_nfev",
        "vs_cold",
        "vs_warm",
        "success_all",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _format_nfev(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.1f}"


def _write_markdown(rows: list[dict[str, Any]], path: Path, *, policies: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_header = " | ".join(policies)
    policy_align = " | ".join("---:" for _ in policies)
    lines = [
        "# VEQlib Continuation nfev Benchmark",
        "",
        "The policy columns are mean effective nfev across repeats; effective nfev includes "
        "warm-start certification/predictor/chord residual evaluations and fallback solves.",
        "",
        f"| experiment | span | case | config | {policy_header} | best | vs cold | vs warm |",
        f"|---|---:|---|---|{policy_align}|---|---:|---:|",
    ]
    for row in rows:
        policy_values = " | ".join(_format_nfev(float(row[policy])) for policy in policies)
        lines.append(
            "| {experiment} | {span:g} | {case} | {config} | {values} | {best} | "
            "{vs_cold:.2f}x | {vs_warm:.2f}x |".format(
                experiment=row["experiment"],
                span=float(row["span"]),
                case=row["case"],
                config=row["config"],
                values=policy_values,
                best=row["best"],
                vs_cold=float(row["vs_cold"]),
                vs_warm=float(row["vs_warm"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]], *, policies: tuple[str, ...]) -> None:
    policy_header = " | ".join(policies)
    policy_align = " | ".join("---:" for _ in policies)
    print(f"| experiment | span | case | config | {policy_header} | best | vs cold | vs warm |")
    print(f"|---|---:|---|---|{policy_align}|---|---:|---:|")
    for row in rows:
        policy_values = " | ".join(_format_nfev(float(row[policy])) for policy in policies)
        print(
            "| {experiment} | {span:g} | {case} | {config} | {values} | {best} | "
            "{vs_cold:.2f}x | {vs_warm:.2f}x |".format(
                experiment=row["experiment"],
                span=float(row["span"]),
                case=row["case"],
                config=row["config"],
                values=policy_values,
                best=row["best"],
                vs_cold=float(row["vs_cold"]),
                vs_warm=float(row["vs_warm"]),
            )
        )


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

    rows: list[dict[str, Any]] = []
    for update in updates:
        for span in spans:
            for case in cases:
                print(
                    f"[continuation] update={update} span={span:g} case={case.row_label}",
                    flush=True,
                )
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
        "warm_alias": "warm-predict",
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
        print(f"json: {raw_path}")
        print(f"csv : {csv_path}")
        print(f"md  : {md_path}")
    _print_summary(comparison_rows, policies=policies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
