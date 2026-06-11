"""Kernel-only residual timing for the benchmark route matrix.

This script reuses ``tests/benchmark.py`` to build the same 46 route/constraint
cases, solves each case once to obtain a representative packed state, then
times repeated residual writes for that fixed state.  The nonlinear solver and
SciPy are therefore setup costs only; measured samples cover the operator
fused residual path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

KERNEL_REPEAT_COUNT = 1000
KERNEL_WARMUP_COUNT = 20
BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.py"


@dataclass(frozen=True)
class KernelBenchmarkRow:
    case_name: str
    mode: str
    coordinate: str
    constraint: str
    input_kind: str
    measured_path: str
    repeat_count: int
    warmup_count: int
    state_size: int
    residual_size: int
    solve_success: bool
    function_evaluations: int
    jacobian_evaluations: int
    iterations: int
    solve_residual_norm: float
    output_norm: float
    output_max_abs: float
    repeat_max_abs_delta: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("veqpy_route_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_dir(benchmark: ModuleType) -> Path:
    return benchmark._artifact_dir()


def _solve_case_once(benchmark: ModuleType, case) -> tuple[object, object]:
    solver = benchmark.Solver(
        operator=benchmark.Operator(benchmark.TEST_GRID, case),
        config=benchmark.CONFIG,
    )
    solver.solve(
        method=benchmark.CONFIG.method,
        max_residual=benchmark.CONFIG.max_residual,
        max_evaluations=benchmark.CONFIG.max_evaluations,
        enable_verbose=False,
        enable_history=False,
    )
    result = solver.result
    if result is None:
        raise RuntimeError("kernel benchmark solve produced no result")
    return solver.operator, result


def _residual_writer(operator) -> tuple[str, object]:
    layout = getattr(operator, "layout", None)
    layout_runner = getattr(layout, "run_fused_residual_into", None)
    if callable(layout_runner):
        return "Operator.layout.run_fused_residual_into(x, out)", layout_runner

    private_runner = getattr(operator, "_residual_var_into_kernel_ready", None)
    if callable(private_runner):
        return "Operator._residual_var_into_kernel_ready(x, out)", private_runner

    public_runner = getattr(operator, "residual_var_into", None)
    if callable(public_runner):
        return "Operator.residual_var_into(x, out)", public_runner

    raise TypeError("operator does not expose a residual writer")


def _time_residual_kernel(
    operator,
    x: np.ndarray,
    *,
    repeat_count: int,
    warmup_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out = np.empty(operator.x_size, dtype=np.float64)
    measured_path, residual_writer = _residual_writer(operator)

    operator.invalidate_source_state()
    for _ in range(warmup_count):
        residual_writer(x, out)
    baseline = out.copy()

    elapsed_ms = np.empty(repeat_count, dtype=np.float64)
    for index in range(repeat_count):
        start_ns = time.perf_counter_ns()
        residual_writer(x, out)
        elapsed_ms[index] = (time.perf_counter_ns() - start_ns) * 1.0e-6

    max_abs_delta = float(np.max(np.abs(out - baseline))) if out.size else 0.0
    stats = {
        "measured_path": measured_path,
        "avg_ms": float(np.mean(elapsed_ms)),
        "median_ms": float(np.median(elapsed_ms)),
        "p95_ms": float(np.percentile(elapsed_ms, 95.0)),
        "min_ms": float(np.min(elapsed_ms)),
        "max_ms": float(np.max(elapsed_ms)),
        "std_ms": float(np.std(elapsed_ms)),
        "repeat_max_abs_delta": max_abs_delta,
    }
    return baseline, elapsed_ms, stats


def _benchmark_case(
    benchmark: ModuleType,
    spec,
    reference,
    *,
    repeat_count: int,
    warmup_count: int,
) -> KernelBenchmarkRow:
    case = benchmark._make_benchmark_case(spec, reference)
    operator, result = _solve_case_once(benchmark, case)
    x = np.ascontiguousarray(result.x, dtype=np.float64)
    benchmark_output, _, stats = _time_residual_kernel(
        operator,
        x,
        repeat_count=repeat_count,
        warmup_count=warmup_count,
    )

    return KernelBenchmarkRow(
        case_name=spec.case_name,
        mode=spec.mode,
        coordinate=spec.coordinate,
        constraint=spec.constraint,
        input_kind=spec.input_kind,
        measured_path=str(stats["measured_path"]),
        repeat_count=repeat_count,
        warmup_count=warmup_count,
        state_size=int(x.size),
        residual_size=int(benchmark_output.size),
        solve_success=bool(result.success),
        function_evaluations=int(result.function_evaluations),
        jacobian_evaluations=int(result.jacobian_evaluations),
        iterations=int(result.iterations),
        solve_residual_norm=float(result.residual_norm_final),
        output_norm=float(np.linalg.norm(benchmark_output)),
        output_max_abs=float(np.max(np.abs(benchmark_output))) if benchmark_output.size else 0.0,
        repeat_max_abs_delta=float(stats["repeat_max_abs_delta"]),
        avg_ms=float(stats["avg_ms"]),
        median_ms=float(stats["median_ms"]),
        p95_ms=float(stats["p95_ms"]),
        min_ms=float(stats["min_ms"]),
        max_ms=float(stats["max_ms"]),
        std_ms=float(stats["std_ms"]),
    )


def _write_text_report(path: Path, rows: list[KernelBenchmarkRow]) -> None:
    total_avg = sum(row.avg_ms for row in rows)
    total_median = sum(row.median_ms for row in rows)
    slowest_avg = max(rows, key=lambda row: row.avg_ms)
    slowest_p95 = max(rows, key=lambda row: row.p95_ms)
    largest_delta = max(rows, key=lambda row: row.repeat_max_abs_delta)
    failures = [row for row in rows if not row.solve_success]
    measured_paths = sorted({row.measured_path for row in rows})
    measured_path = measured_paths[0] if len(measured_paths) == 1 else "mixed"

    lines = [
        "Kernel-only residual benchmark",
        "",
        f"Measured path: {measured_path}",
        "Setup path: one Solver.solve(...) per case, excluded from timings",
        "State policy: invalidate source state once, warm up, then time steady-state repeats",
        "",
        f"case_count       : {len(rows)}",
        f"repeat_count     : {rows[0].repeat_count if rows else 0}",
        f"warmup_count     : {rows[0].warmup_count if rows else 0}",
        f"failure_count    : {len(failures)}/{len(rows)}",
        f"total_avg_ms     : {total_avg:.6f}",
        f"total_median_ms  : {total_median:.6f}",
        f"slowest_avg_case : {slowest_avg.case_name} ({slowest_avg.avg_ms:.6f} ms)",
        f"slowest_p95_case : {slowest_p95.case_name} ({slowest_p95.p95_ms:.6f} ms)",
        (
            f"largest_repeat_delta_case : {largest_delta.case_name} "
            f"({largest_delta.repeat_max_abs_delta:.6e})"
        ),
        "",
        "Case results",
        "",
    ]
    lines.append(
        "case".ljust(24)
        + " | "
        + "avg_ms".rjust(10)
        + " | "
        + "median_ms".rjust(10)
        + " | "
        + "p95_ms".rjust(10)
        + " | "
        + "std_ms".rjust(10)
        + " | "
        + "out_norm".rjust(12)
        + " | "
        + "repeat_delta".rjust(12)
        + " | "
        + "evals".rjust(5)
        + " | "
        + "ok".rjust(3)
    )
    lines.append("-" * 119)
    for row in rows:
        lines.append(
            f"{row.case_name:<24} | "
            f"{row.avg_ms:>10.6f} | "
            f"{row.median_ms:>10.6f} | "
            f"{row.p95_ms:>10.6f} | "
            f"{row.std_ms:>10.6f} | "
            f"{row.output_norm:>12.6e} | "
            f"{row.repeat_max_abs_delta:>12.6e} | "
            f"{row.function_evaluations:>5d} | "
            f"{'yes' if row.solve_success else 'no':>3}"
        )

    lines.extend(["", "Slowest avg_ms ranking", ""])
    lines.append("rank | case".ljust(32) + " | " + "avg_ms".rjust(10) + " | " + "p95_ms".rjust(10))
    lines.append("-" * 57)
    for index, row in enumerate(sorted(rows, key=lambda item: -item.avg_ms), start=1):
        lines.append(
            f"{index:>4} | {row.case_name:<24} | "
            f"{row.avg_ms:>10.6f} | {row.p95_ms:>10.6f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json_report(path: Path, rows: list[KernelBenchmarkRow]) -> None:
    measured_paths = sorted({row.measured_path for row in rows})
    payload = {
        "schema_version": 1,
        "measured_path": measured_paths[0] if len(measured_paths) == 1 else "mixed",
        "measured_paths": measured_paths,
        "setup_path": "one Solver.solve(...) per case, excluded from timings",
        "state_policy": "invalidate source state once, warm up, then time steady-state repeats",
        "case_count": len(rows),
        "repeat_count": rows[0].repeat_count if rows else 0,
        "warmup_count": rows[0].warmup_count if rows else 0,
        "total_avg_ms": sum(row.avg_ms for row in rows),
        "total_median_ms": sum(row.median_ms for row in rows),
        "cases": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_kernel_benchmark(
    *,
    repeat_count: int = KERNEL_REPEAT_COUNT,
    warmup_count: int = KERNEL_WARMUP_COUNT,
    show_progress: bool = True,
) -> tuple[list[KernelBenchmarkRow], Path]:
    benchmark = _load_benchmark_module()
    reference = benchmark._solve_reference(show_progress=show_progress)
    specs = list(benchmark._iter_benchmark_specs())
    rows: list[KernelBenchmarkRow] = []

    for index, spec in enumerate(specs, start=1):
        row = _benchmark_case(
            benchmark,
            spec,
            reference,
            repeat_count=repeat_count,
            warmup_count=warmup_count,
        )
        rows.append(row)
        if show_progress:
            print(
                f"[{index:02d}/{len(specs)}] {row.case_name}: "
                f"kernel={row.avg_ms:.6f} ms "
                f"(median={row.median_ms:.6f}, p95={row.p95_ms:.6f}) | "
                f"delta={row.repeat_max_abs_delta:.2e}"
            )

    outdir = _artifact_dir(benchmark)
    text_path = outdir / "kernel_compare.txt"
    _write_text_report(text_path, rows)
    _write_json_report(outdir / "kernel_compare.json", rows)
    return rows, text_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _run_as_script(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat-count",
        type=_positive_int,
        default=KERNEL_REPEAT_COUNT,
        help=f"timed residual repeats per case (default: {KERNEL_REPEAT_COUNT})",
    )
    parser.add_argument(
        "--warmup-count",
        type=_nonnegative_int,
        default=KERNEL_WARMUP_COUNT,
        help=f"untimed residual warmups per case (default: {KERNEL_WARMUP_COUNT})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-case progress lines",
    )
    args = parser.parse_args(argv)
    rows, text_path = run_kernel_benchmark(
        repeat_count=args.repeat_count,
        warmup_count=args.warmup_count,
        show_progress=not args.quiet,
    )
    print(text_path)
    return 0 if all(row.solve_success for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
