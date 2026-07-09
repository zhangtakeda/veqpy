#!/usr/bin/env python3
"""Benchmark boundary scatter-to-coefficient fitters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from rich.table import Table
from rich.text import Text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (  # noqa: E402
    CASE_KEYS,
    CASE_REFERENCE_GFILES,
    REPO_ROOT,
    benchmark_result_path,
    cpu_affinity,
    float_stats,
    runtime_env,
    runtime_platform_payload,
    write_json,
)
from benchmarks._reporting import (  # noqa: E402
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    format_optional_speedup,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
)
from benchmarks._reporting import (  # noqa: E402
    console as reporting_console,
)
from veqpy import KernelBoundary  # noqa: E402
from veqpy.model import Geqdsk  # noqa: E402

DEFAULT_OUTPUT = benchmark_result_path("cxx_boundary_fitters")
DEFAULT_C_ORDER = 10
DEFAULT_S_ORDER = 10
DEFAULT_MAXTOL = 1.0
SUPPORTED_METHODS = ("qr", "gnqr", "least-square")
DEFAULT_METHODS = SUPPORTED_METHODS
SUPPORTED_BACKENDS = ("numpy", "numba", "cxx")
DEFAULT_BACKENDS = ("numba", "cxx")


def _load_boundary_points(case_key: str) -> tuple[np.ndarray, np.ndarray, float]:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key])
    points = np.asarray(geqdsk.boundary, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{case_key} boundary must be an (n, 2) array, got {points.shape}")
    return points[:, 0].copy(), points[:, 1].copy(), float(geqdsk.Bt0)


def _raw_boundary(
    R_boundary: np.ndarray,
    Z_boundary: np.ndarray,
    B0: float,
    *,
    c_order: int,
    s_order: int,
    maxtol: float,
    method: str | None = None,
) -> KernelBoundary:
    return KernelBoundary(
        B0=B0,
        R_boundary=R_boundary,
        Z_boundary=Z_boundary,
        c_order=c_order,
        s_order=s_order,
        fit_maxtol=maxtol,
        method=method,
    )


def _fit_once(
    boundary: KernelBoundary,
    *,
    backend: str,
    method: str,
) -> KernelBoundary:
    return boundary.fit(backend=backend, method=method)


def _measure_fit(
    backend: str,
    boundary: KernelBoundary,
    *,
    method: str,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _fit_once(
            boundary,
            backend=backend,
            method=method,
        )

    samples_ms: list[float] = []
    result: KernelBoundary | None = None
    for _ in range(repeat):
        started = perf_counter_ns()
        result = _fit_once(
            boundary,
            backend=backend,
            method=method,
        )
        samples_ms.append((perf_counter_ns() - started) / 1.0e6)

    if result is None:
        raise RuntimeError("boundary fitter did not produce a result")
    return {"timing": float_stats(samples_ms), "fit": _fit_payload(result)}


def _fit_payload(boundary: KernelBoundary) -> dict[str, Any]:
    s_offsets = np.concatenate(([0.0], np.asarray(boundary.s_offsets, dtype=np.float64)))
    return {
        "R0": float(boundary.R0),
        "Z0": float(boundary.Z0),
        "a": float(boundary.a),
        "ka": float(boundary.ka),
        "c_order": int(boundary.fit_c_order),
        "s_order": int(boundary.fit_s_order),
        "method": str(boundary.fit_method),
        "rms": float(boundary.fit_rms),
        "max_curve_error": float(boundary.fit_max_curve_error),
        "c_offsets": np.asarray(boundary.c_offsets, dtype=np.float64).tolist(),
        "s_offsets": s_offsets.tolist(),
    }


def _coefficient_vector(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(payload["R0"]),
            float(payload["Z0"]),
            float(payload["a"]),
            float(payload["ka"]),
            *[float(value) for value in payload["c_offsets"]],
            *[float(value) for value in payload["s_offsets"]],
        ],
        dtype=np.float64,
    )


def _accuracy_payload(fit: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    coeff = _coefficient_vector(fit)
    ref_coeff = _coefficient_vector(reference)
    if coeff.shape != ref_coeff.shape:
        coeff_linf = float("nan")
    else:
        coeff_linf = float(np.max(np.abs(coeff - ref_coeff)))
    return {
        "coeff_linf_vs_reference": coeff_linf,
        "rms_delta_vs_reference": float(abs(float(fit["rms"]) - float(reference["rms"]))),
        "curve_delta_vs_reference": float(
            abs(float(fit["max_curve_error"]) - float(reference["max_curve_error"]))
        ),
    }


def _measure_case(
    args: argparse.Namespace,
    case_key: str,
    method: str,
    backend: str,
) -> dict[str, Any]:
    R_boundary, Z_boundary, B0 = _load_boundary_points(case_key)
    boundary = _raw_boundary(
        R_boundary,
        Z_boundary,
        B0,
        c_order=args.c_order,
        s_order=args.s_order,
        maxtol=args.maxtol,
    )
    row = {
        "case": case_key,
        "method": method,
        "backend": backend,
        "points": int(R_boundary.size),
        "order": {"c_order": int(args.c_order), "s_order": int(args.s_order)},
        "status": "planned" if args.no_run else "not_requested",
    }
    if args.no_run:
        return row

    try:
        reference = _fit_payload(
            boundary.fit(backend="numpy", method=method)
        )
        measurement = _measure_fit(
            backend,
            boundary,
            method=method,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        accuracy = _accuracy_payload(measurement["fit"], reference)
        row.update(
            {
                "status": "passed",
                "reference_backend": "numpy",
                "timing": measurement["timing"],
                "fit": measurement["fit"],
                "accuracy": accuracy,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("status") == "passed")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    planned = sum(1 for row in rows if row.get("status") == "planned")
    return {"total": total, "passed": passed, "failed": failed, "planned": planned}


def _print_summary(console, summary: dict[str, int]) -> None:
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("summary")
    table.add_column("count", justify="right")
    for key in ("total", "passed", "failed", "planned"):
        table.add_row(key, str(summary.get(key, 0)))
    console.print(table)


def _print_timing_table(console, rows: list[dict[str, Any]]) -> None:
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("case", no_wrap=True)
    table.add_column("method", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("order", justify="right")
    table.add_column(Text("Cxx (ms)"), justify="right")
    table.add_column(Text("Numba (ms)"), justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("coeff diff", justify="right")
    table.add_column("rms diff", justify="right")
    table.add_column("curve diff", justify="right")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("case", "n/a")), str(row.get("method", "n/a")))
        grouped.setdefault(key, {})[str(row.get("backend", "n/a"))] = row

    for (case_key, method), group in grouped.items():
        cxx = group.get("cxx")
        numba = group.get("numba")
        first = next(iter(group.values()))
        order = first.get("order", {})
        cxx_ms = (cxx or {}).get("timing", {}).get("median_ms")
        numba_ms = (numba or {}).get("timing", {}).get("median_ms")
        status_values = {row.get("status") for row in group.values()}
        if "failed" in status_values:
            status = "failed"
        elif "planned" in status_values:
            status = "planned"
        elif {"cxx", "numba"}.issubset(group) and status_values == {"passed"}:
            status = "passed"
        else:
            status = "partial"
        if cxx is not None and numba is not None and cxx.get("fit") and numba.get("fit"):
            cxx_fit = cxx["fit"]
            numba_fit = numba["fit"]
            cxx_coeff = _coefficient_vector(cxx_fit)
            numba_coeff = _coefficient_vector(numba_fit)
            coeff_diff = float(np.max(np.abs(cxx_coeff - numba_coeff)))
            rms_diff = abs(float(cxx_fit["rms"]) - float(numba_fit["rms"]))
            curve_diff = abs(
                float(cxx_fit["max_curve_error"]) - float(numba_fit["max_curve_error"])
            )
        else:
            coeff_diff = None
            rms_diff = None
            curve_diff = None
        table.add_row(
            case_key,
            method,
            status_cell(status),
            f"{order.get('c_order', 'n/a')}/{order.get('s_order', 'n/a')}",
            format_optional_float(cxx_ms),
            format_optional_float(numba_ms),
            format_optional_speedup(numba_ms, cxx_ms),
            format_optional_sci(coeff_diff),
            format_optional_sci(rms_diff),
            format_optional_sci(curve_diff),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--method", action="append", choices=SUPPORTED_METHODS)
    parser.add_argument("--backend", action="append", choices=SUPPORTED_BACKENDS)
    parser.add_argument("--c-order", type=int, default=DEFAULT_C_ORDER)
    parser.add_argument("--s-order", type=int, default=DEFAULT_S_ORDER)
    parser.add_argument("--maxtol", type=float, default=DEFAULT_MAXTOL)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if args.c_order < 0 or args.s_order < 0:
        raise ValueError("--c-order and --s-order must be non-negative")

    case_keys = tuple(args.case or CASE_KEYS)
    methods = tuple(args.method or DEFAULT_METHODS)
    backends = tuple(args.backend or DEFAULT_BACKENDS)
    console = reporting_console()

    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{len(case_keys)}[/]",
                f"methods: [green]{','.join(methods)}[/]",
                f"backends: [green]{','.join(backends)}[/]",
                f"order: [green]{args.c_order}/{args.s_order}[/]",
                f"maxtol: [green]{args.maxtol:g}[/]",
                f"mode: [green]{'plan-only' if args.no_run else 'run'}[/]",
                f"warmup: [green]{args.warmup}[/]",
                f"repeat: [green]{args.repeat}[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))

    jobs = [
        (case_key, method, backend)
        for case_key in case_keys
        for method in methods
        for backend in backends
    ]
    rows: list[dict[str, Any]] = []
    with progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "boundary-fitters",
                total=len(jobs),
                current="-",
                phase="[cyan]run[/]",
            )
        for case_key, method, backend in jobs:
            label = f"{case_key}:{method}:{backend}"
            if progress is not None and task_id is not None:
                progress.update(task_id, current=label, phase="[cyan]run[/]")
            row = _measure_case(args, case_key, method, backend)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=progress_phase(row["status"]))
                progress.advance(task_id)

    summary = _summarize(rows)
    payload = {
        "schema": "veqpy.boundary.fitters.v2",
        "case_count": len(case_keys),
        "method_count": len(methods),
        "backend_count": len(backends),
        "methods": list(methods),
        "backends": list(backends),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "maxtol": float(args.maxtol),
        "order": {"c_order": int(args.c_order), "s_order": int(args.s_order)},
        "timing_note": "Wall time around one KernelBoundary.fit scatter-to-coefficient call.",
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "platform": runtime_platform_payload(),
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        if not args.quiet_progress:
            console.print()
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)

    _print_summary(console, summary)
    _print_timing_table(console, rows)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
