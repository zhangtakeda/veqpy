#!/usr/bin/env python3
"""Four-case Ip continuation-scan benchmark for the VEQlib artifact path."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/tmp/veqlib_4case_ip_scan.json")
DEFAULT_MPLCONFIG = Path("/tmp/veqpy-mpl")
CASE_CHOICES = ("PF_psin_uniform_Ip", "solovev", "chease", "efit")

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


def _payload_with_ip(base_payload: dict[str, Any], scaled_ip: float) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["constraints"]["scaled_Ip"] = float(scaled_ip)
    return payload


def _scan_points(base_ip: float, *, points: int, relative_span: float) -> list[float]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return [float(base_ip)]
    lower = -0.5 * relative_span
    step = relative_span / float(points - 1)
    return [float(base_ip * (1.0 + lower + step * index)) for index in range(points)]


def _measure_policy(
    solver: VEQlibSolver,
    payloads: list[dict[str, Any]],
    *,
    repeat: int,
    warmup: int,
    policy: str,
) -> dict[str, Any]:
    first_policy = "cold"
    continuation_policy = "cold" if policy == "cold" else "warm-clone"

    for _ in range(warmup):
        solve_payload_sequence(
            solver,
            payloads,
            first_policy=first_policy,
            continuation_policy=continuation_policy,
        )

    wall_ms: list[float] = []
    internal_ms: list[float] = []
    total_nfev: list[int] = []
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
        total_nfev.append(sum(step.nfev for step in steps))
        all_success.append(all(step.success for step in steps))
        last_steps = steps

    return {
        "policy": policy,
        "first_policy": first_policy,
        "continuation_policy": continuation_policy,
        "wall_ms": _float_stats(wall_ms),
        "internal_ms": _float_stats(internal_ms),
        "total_nfev": _int_stats(total_nfev),
        "success_all": all(all_success),
        "last_point_nfev": [step.nfev for step in last_steps],
        "last_point_raw_norm": [step.raw_norm for step in last_steps],
    }


def _measure_case(
    case_data: Any,
    *,
    registry: KernelRegistry,
    repeat: int,
    warmup: int,
    points: int,
    relative_span: float,
) -> dict[str, Any]:
    base_payload = json.loads(case_data.payload_json)
    base_ip = float(base_payload["constraints"]["scaled_Ip"])
    scaled_ip_points = _scan_points(base_ip, points=points, relative_span=relative_span)
    payloads = [_payload_with_ip(base_payload, scaled_ip) for scaled_ip in scaled_ip_points]

    solver = VEQlibSolver(case_data.topology, registry=registry, solver="powell")
    artifact = solver.build(force=False, dry_run=False)

    cold = _measure_policy(solver, payloads, repeat=repeat, warmup=warmup, policy="cold")
    warm = _measure_policy(solver, payloads, repeat=repeat, warmup=warmup, policy="warm-clone")
    return {
        "case": case_data.name,
        "x_size": case_data.x_size,
        "metadata": case_data.metadata,
        "scaled_Ip_base": base_ip,
        "scaled_Ip_points": scaled_ip_points,
        "artifact": {"artifact_id": artifact.artifact_id, "reused": bool(artifact.reused)},
        "policies": {"cold": cold, "warm-clone": warm},
        "ratios": {
            "wall_median_warm_over_cold": float(
                warm["wall_ms"]["median"] / cold["wall_ms"]["median"]
            ),
            "internal_median_warm_over_cold": float(
                warm["internal_ms"]["median"] / cold["internal_ms"]["median"]
            ),
            "nfev_median_warm_over_cold": float(
                warm["total_nfev"]["median"] / cold["total_nfev"]["median"]
            ),
        },
    }


def _write_report(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--relative-span", type=float, default=0.20)
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--case",
        action="append",
        choices=CASE_CHOICES,
        help="Run only the selected case; repeat the option for multiple cases.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    selected = set(args.case) if args.case else None
    cases = _make_cases(repeat=1, warmup=0, selected=selected)
    cache = args.cache_root or Path(tempfile.mkdtemp(prefix="veqlib-4case-ip-scan-"))
    registry = KernelRegistry(cache_root=cache, source_dir=REPO_ROOT / "veqlib")
    rows = []
    for case_data in cases:
        print(f"measuring {case_data.name} ...", flush=True)
        rows.append(
            _measure_case(
                case_data,
                registry=registry,
                repeat=args.repeat,
                warmup=args.warmup,
                points=args.points,
                relative_span=args.relative_span,
            )
        )

    payload = {
        "schema": "veqlib.4case_ip_scan_continuation.v1",
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "points": int(args.points),
        "relative_span": float(args.relative_span),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cache": str(cache),
        "rows": rows,
    }

    if not args.no_write:
        _write_report(payload, args.output)
        print(f"json: {args.output}")

    print(
        f"repeat={args.repeat} warmup={args.warmup} points={args.points} "
        f"span={args.relative_span:.3f} affinity={payload['cpu_affinity']}"
    )
    print("| case | x_size | cold wall ms | warm wall ms | wall ratio | nfev ratio |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        cold = row["policies"]["cold"]
        warm = row["policies"]["warm-clone"]
        ratios = row["ratios"]
        print(
            f"| {row['case']} | {row['x_size']} | {cold['wall_ms']['median']:.6f} | "
            f"{warm['wall_ms']['median']:.6f} | "
            f"{ratios['wall_median_warm_over_cold']:.3f} | "
            f"{ratios['nfev_median_warm_over_cold']:.3f} |"
        )
    print("\nPoint nfev on last repeat")
    print("| case | cold | warm-clone |")
    print("|---|---|---|")
    for row in rows:
        cold = row["policies"]["cold"]
        warm = row["policies"]["warm-clone"]
        print(f"| {row['case']} | {cold['last_point_nfev']} | {warm['last_point_nfev']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
