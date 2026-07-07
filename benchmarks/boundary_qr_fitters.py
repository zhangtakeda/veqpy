#!/usr/bin/env python3
"""Benchmark boundary scatter-to-coefficient QR fitters."""

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
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
)
from benchmarks._reporting import (  # noqa: E402
    console as reporting_console,
)
from veqpy.kernels.boundary_fit import fit_boundary_params  # noqa: E402
from veqpy.kernels.numba_kernel.boundary_fit import fit_boundary_params_numba  # noqa: E402
from veqpy.model import Geqdsk  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "boundary_qr_fitters.json"
DEFAULT_C_ORDER = 10
DEFAULT_S_ORDER = 10
DEFAULT_MAXTOL = 1.0
SUPPORTED_BACKENDS = ("numpy", "numba")


def _load_boundary_points(case_key: str) -> tuple[np.ndarray, np.ndarray]:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key])
    points = np.asarray(geqdsk.boundary, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{case_key} boundary must be an (n, 2) array, got {points.shape}")
    return points[:, 0].copy(), points[:, 1].copy()


def _fit_numpy(
    R_boundary: np.ndarray,
    Z_boundary: np.ndarray,
    *,
    c_order: int,
    s_order: int,
    maxtol: float,
) -> dict[str, Any]:
    return fit_boundary_params(
        R_boundary,
        Z_boundary,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
    )


def _fit_numba(
    R_boundary: np.ndarray,
    Z_boundary: np.ndarray,
    *,
    c_order: int,
    s_order: int,
    maxtol: float,
) -> dict[str, Any]:
    return fit_boundary_params_numba(
        R_boundary,
        Z_boundary,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
    )


def _fit_once(
    backend: str,
    R_boundary: np.ndarray,
    Z_boundary: np.ndarray,
    *,
    c_order: int,
    s_order: int,
    maxtol: float,
) -> dict[str, Any]:
    if backend == "numpy":
        return _fit_numpy(
            R_boundary,
            Z_boundary,
            c_order=c_order,
            s_order=s_order,
            maxtol=maxtol,
        )
    if backend == "numba":
        return _fit_numba(
            R_boundary,
            Z_boundary,
            c_order=c_order,
            s_order=s_order,
            maxtol=maxtol,
        )
    raise ValueError(f"unsupported boundary fitter backend {backend!r}")


def _measure_fit(
    backend: str,
    R_boundary: np.ndarray,
    Z_boundary: np.ndarray,
    *,
    c_order: int,
    s_order: int,
    maxtol: float,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _fit_once(
            backend,
            R_boundary,
            Z_boundary,
            c_order=c_order,
            s_order=s_order,
            maxtol=maxtol,
        )

    samples_ms: list[float] = []
    result: dict[str, Any] | None = None
    for _ in range(repeat):
        started = perf_counter_ns()
        result = _fit_once(
            backend,
            R_boundary,
            Z_boundary,
            c_order=c_order,
            s_order=s_order,
            maxtol=maxtol,
        )
        samples_ms.append((perf_counter_ns() - started) / 1.0e6)

    if result is None:
        raise RuntimeError("boundary fitter did not produce a result")
    return {"timing": float_stats(samples_ms), "fit": _fit_payload(result)}


def _fit_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "R0": float(result["R0"]),
        "Z0": float(result["Z0"]),
        "a": float(result["a"]),
        "ka": float(result["ka"]),
        "c_order": int(result["c_order"]),
        "s_order": int(result["s_order"]),
        "rms": float(result["rms"]),
        "max_curve_error": float(result["max_curve_error"]),
        "c_offsets": np.asarray(result["c_offsets"], dtype=np.float64).tolist(),
        "s_offsets": np.asarray(result["s_offsets"], dtype=np.float64).tolist(),
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
        "coeff_linf_vs_numpy": coeff_linf,
        "rms_delta_vs_numpy": float(abs(float(fit["rms"]) - float(reference["rms"]))),
        "curve_delta_vs_numpy": float(
            abs(float(fit["max_curve_error"]) - float(reference["max_curve_error"]))
        ),
    }


def _measure_case(args: argparse.Namespace, case_key: str, backend: str) -> dict[str, Any]:
    R_boundary, Z_boundary = _load_boundary_points(case_key)
    row = {
        "case": case_key,
        "backend": backend,
        "points": int(R_boundary.size),
        "order": {"c_order": int(args.c_order), "s_order": int(args.s_order)},
        "status": "planned" if args.no_run else "not_requested",
    }
    if args.no_run:
        return row

    try:
        reference = _fit_payload(
            _fit_numpy(
                R_boundary,
                Z_boundary,
                c_order=args.c_order,
                s_order=args.s_order,
                maxtol=args.maxtol,
            )
        )
        measurement = _measure_fit(
            backend,
            R_boundary,
            Z_boundary,
            c_order=args.c_order,
            s_order=args.s_order,
            maxtol=args.maxtol,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        accuracy = _accuracy_payload(measurement["fit"], reference)
        row.update(
            {
                "status": "passed",
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
    table.add_column("backend", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("points", justify="right")
    table.add_column("order", justify="right")
    table.add_column(Text("median (ms)"), justify="right")
    table.add_column("fit rms", justify="right")
    table.add_column("curve", justify="right")
    table.add_column("coeff diff", justify="right")
    for row in rows:
        timing = row.get("timing", {})
        fit = row.get("fit", {})
        accuracy = row.get("accuracy", {})
        order = row.get("order", {})
        table.add_row(
            str(row.get("case", "n/a")),
            str(row.get("backend", "n/a")),
            status_cell(row.get("status")),
            str(row.get("points", "n/a")),
            f"{order.get('c_order', 'n/a')}/{order.get('s_order', 'n/a')}",
            format_optional_float(timing.get("median_ms")),
            format_optional_sci(fit.get("rms")),
            format_optional_sci(fit.get("max_curve_error")),
            format_optional_sci(accuracy.get("coeff_linf_vs_numpy")),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
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
    backends = tuple(args.backend or SUPPORTED_BACKENDS)
    console = reporting_console()

    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{len(case_keys)}[/]",
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

    jobs = [(case_key, backend) for case_key in case_keys for backend in backends]
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
        for case_key, backend in jobs:
            label = f"{case_key}:{backend}"
            if progress is not None and task_id is not None:
                progress.update(task_id, current=label, phase="[cyan]run[/]")
            row = _measure_case(args, case_key, backend)
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=progress_phase(row["status"]))
                progress.advance(task_id)

    summary = _summarize(rows)
    payload = {
        "schema": "veqpy.boundary.qr_fitters.v1",
        "case_count": len(case_keys),
        "backend_count": len(backends),
        "backends": list(backends),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "maxtol": float(args.maxtol),
        "order": {"c_order": int(args.c_order), "s_order": int(args.s_order)},
        "timing_note": "Wall time around one scatter-to-coefficient boundary fit call.",
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
