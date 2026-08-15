#!/usr/bin/env python3
"""Benchmark GEQDSK-backed VEQPy core solves and nonmaterializing Modules.

The formal rows deliberately measure a prepared, warmed runtime.  Artifact
builds, imports, and Numba JIT work happen before the timing loop.  The primary
measurement is ``module._kernel.solve()`` after Adapter preprocessing; the
secondary measurement is the public ``VEQ.solve(materialize=False)`` path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    BACKENDS,
    ENZYME_SKIP_REASON,
    HISTORICAL_SPEEDUP_RANGE,
    REFERENCE_SOLVER_MAXFEV,
    RUNNABLE_BACKENDS,
    benchmark_result_path,
    build_module,
    cpu_affinity,
    default_kernel_cache_root,
    geqdsk_kernel_case,
    integer_statistics,
    monotonic_interleave,
    prepare_metadata,
    runtime_env,
    selected_cases,
    selected_configs,
    statistics_payload,
    time_call,
    write_json,
)

DEFAULT_OUTPUT = benchmark_result_path("cxx_geqdsk")
VALIDATION_ATOL = 1.0e-6
STRICT_SAME_INPUT_ATOL = 1.0e-10
RELAXED_SAME_INPUT_ATOL = 1.0e-8


def _snapshot_output(output: Any) -> dict[str, Any]:
    return {
        "success": bool(output.success),
        "accepted": bool(output.success and np.isfinite(output.raw_norm)),
        "info": int(output.info),
        "nfev": int(output.nfev),
        "njev": int(output.njev),
        "raw_norm": float(output.raw_norm),
        "scaled_norm": float(output.scaled_norm),
        "elapsed_ms": float(output.elapsed_ms),
        "preprocess_ms": float(output.preprocess_ms),
        "solve_ms": float(output.solve_ms),
        "postprocess_ms": float(output.postprocess_ms),
        "alpha": np.asarray(output.alpha, dtype=np.float64).tolist(),
        "x": np.asarray(output.x, dtype=np.float64).tolist(),
        "raw": np.asarray(output.raw, dtype=np.float64).tolist(),
    }


def _snapshot_record(record: Any) -> dict[str, Any]:
    return {
        "solved": bool(record.solved),
        "accepted": bool(record.accepted),
        "materialized": bool(record.materialized),
        "residual_norm": float(record.residual_norm),
        "scaled_residual_norm": float(record.scaled_residual_norm),
        "nfev": int(record.evaluations),
        "equilibrium_is_none": record.equilibrium is None,
    }


def _empty_engine(status: str, *, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason}


def _measure_case(
    case,
    *,
    warmup: int,
    repeat: int,
    artifact_dir: Path,
    selected_backends: tuple[str, ...],
) -> dict[str, Any]:
    engines: dict[str, dict[str, Any]] = {}
    modules: dict[str, Any] = {}
    build_errors: dict[str, str] = {}

    for backend in RUNNABLE_BACKENDS:
        if backend not in selected_backends:
            engines[backend] = _empty_engine("not-selected", reason="backend filter")
            continue
        try:
            modules[backend] = build_module(case, backend, artifact_dir=artifact_dir)
            engines[backend] = {
                "status": "prepared",
                "metadata": prepare_metadata(modules[backend], backend),
            }
        except Exception as error:  # pragma: no cover - platform/compiler dependent
            message = f"{type(error).__name__}: {error}"
            build_errors[backend] = message
            engines[backend] = _empty_engine("failed", reason=message)

    engines["cxx-enzyme"] = _empty_engine("skipped", reason=ENZYME_SKIP_REASON)

    active = tuple(backend for backend in RUNNABLE_BACKENDS if backend in modules)
    try:
        for backend in active:
            module = modules[backend]
            for _ in range(warmup):
                module._adapter.fill(case.plasma)
                module._kernel.solve()
                module.solve(
                    plasma=case.plasma,
                    materialize=False,
                    verbose=False,
                    report=False,
                )

        measurements: dict[str, dict[str, Any]] = {
            backend: {
                "core_times_ms": [],
                "core_elapsed_ms": [],
                "core_preprocess_ms": [],
                "core_solve_ms": [],
                "core_postprocess_ms": [],
                "module_times_ms": [],
                "core_nfev": [],
                "module_nfev": [],
                "core_accepted": [],
                "module_accepted": [],
                "module_equilibrium_none": [],
                "last_core": None,
                "last_module": None,
            }
            for backend in active
        }
        for iteration in range(repeat):
            for backend in monotonic_interleave(active, iteration):
                module = modules[backend]
                module._adapter.fill(case.plasma)
                output, core_elapsed_ms = time_call(module._kernel.solve)
                core_snapshot = _snapshot_output(output)
                record, module_elapsed_ms = time_call(
                    lambda: module.solve(
                        plasma=case.plasma,
                        materialize=False,
                        verbose=False,
                        report=False,
                    )
                )
                module_snapshot = _snapshot_record(record)
                row = measurements[backend]
                row["core_times_ms"].append(core_elapsed_ms)
                row["core_elapsed_ms"].append(core_snapshot["elapsed_ms"])
                row["core_preprocess_ms"].append(core_snapshot["preprocess_ms"])
                row["core_solve_ms"].append(core_snapshot["solve_ms"])
                row["core_postprocess_ms"].append(core_snapshot["postprocess_ms"])
                row["module_times_ms"].append(module_elapsed_ms)
                row["core_nfev"].append(core_snapshot["nfev"])
                row["module_nfev"].append(module_snapshot["nfev"])
                row["core_accepted"].append(core_snapshot["accepted"])
                row["module_accepted"].append(module_snapshot["accepted"])
                row["module_equilibrium_none"].append(module_snapshot["equilibrium_is_none"])
                row["last_core"] = core_snapshot
                row["last_module"] = module_snapshot

        for backend in active:
            measured = measurements[backend]
            engines[backend].update(
                {
                    "status": "passed" if all(measured["core_accepted"]) else "failed",
                    "core": {
                        "wall_timing_ms": statistics_payload(measured["core_times_ms"]),
                        # Matches the historical benchmark's result.elapsed_ms
                        # comparison while retaining wall time independently.
                        "timing_ms": statistics_payload(measured["core_elapsed_ms"]),
                        "preprocess_timing_ms": statistics_payload(
                            measured["core_preprocess_ms"]
                        ),
                        "solve_timing_ms": statistics_payload(measured["core_solve_ms"]),
                        "postprocess_timing_ms": statistics_payload(
                            measured["core_postprocess_ms"]
                        ),
                        "nfev": integer_statistics(measured["core_nfev"]),
                        "accepted_all": all(measured["core_accepted"]),
                        "last": measured["last_core"],
                    },
                    "module_materialize_false": {
                        "timing_ms": statistics_payload(measured["module_times_ms"]),
                        "nfev": integer_statistics(measured["module_nfev"]),
                        "accepted_all": all(measured["module_accepted"]),
                        "equilibrium_none_all": all(measured["module_equilibrium_none"]),
                        "last": measured["last_module"],
                    },
                }
            )
            input_buffer = modules[backend]._kernel.input
            engines[backend]["runtime_input"] = {
                "x_size": int(modules[backend]._kernel.x_size),
                "source_count": int(input_buffer.source_count),
                "source_capacity": int(input_buffer.source_capacity),
                "capacity_epoch": int(input_buffer.capacity_epoch),
            }

        numba_core = measurements.get("numba", {}).get("last_core")
        if numba_core is not None:
            reference_x = np.asarray(numba_core["x"], dtype=np.float64)
            for backend in active:
                module = modules[backend]
                module._adapter.fill(case.plasma)
                same_input_raw = np.asarray(module._kernel.residual(reference_x), dtype=np.float64)
                reference_raw = np.asarray(numba_core["raw"], dtype=np.float64)
                engines[backend]["same_input_residual"] = {
                    "x_source": "last numba core solve x",
                    "raw_norm": float(np.linalg.norm(same_input_raw)),
                    "raw_max_abs_to_numba": float(np.max(np.abs(same_input_raw - reference_raw))),
                    "raw_l2_to_numba": float(np.linalg.norm(same_input_raw - reference_raw)),
                }
    except Exception as error:  # pragma: no cover - platform/compiler dependent
        message = f"{type(error).__name__}: {error}"
        for backend in active:
            if engines[backend].get("status") == "prepared":
                engines[backend] = _empty_engine("failed", reason=message)
    finally:
        for module in modules.values():
            module.close()

    numba_core = engines.get("numba", {}).get("core", {}).get("last")
    for backend in RUNNABLE_BACKENDS:
        current = engines.get(backend, {})
        last = current.get("core", {}).get("last")
        if numba_core is not None and last is not None:
            current["parity_to_numba"] = {
                "x_max_abs": float(
                    np.max(
                        np.abs(
                            np.asarray(last["x"], dtype=np.float64)
                            - np.asarray(numba_core["x"], dtype=np.float64)
                        )
                    )
                ),
                "raw_max_abs": float(
                    np.max(
                        np.abs(
                            np.asarray(last["raw"], dtype=np.float64)
                            - np.asarray(numba_core["raw"], dtype=np.float64)
                        )
                    )
                ),
            }

    row = {
        "case": case.case_key,
        "config": case.config_label,
        "label": case.label,
        "status": "failed" if build_errors else "measured",
        "geqdsk": str(case.geqdsk_path) if hasattr(case, "geqdsk_path") else str(case.case_key),
        "geqdsk_profile_count": int(case.geqdsk.P_psi.size),
        "topology": case.topology,
        "solver": case.solver,
        "signature": case.signature,
        "boundary_fit": case.boundary_fit,
        "backends": engines,
    }
    row["correctness"] = _correctness(row)
    row["status"] = "passed" if row["correctness"]["status"] == "passed" else "failed"
    return row


def _correctness(row: dict[str, Any]) -> dict[str, Any]:
    backend_checks: dict[str, dict[str, Any]] = {}
    for backend in RUNNABLE_BACKENDS:
        engine = row.get("backends", {}).get(backend, {})
        core = engine.get("core", {})
        module = engine.get("module_materialize_false", {})
        parity = engine.get("parity_to_numba", {})
        same_input = engine.get("same_input_residual", {})
        residual_atol = (
            RELAXED_SAME_INPUT_ATOL if backend == "cxx-relaxed" else STRICT_SAME_INPUT_ATOL
        )
        checks = {
            "engine_passed": engine.get("status") == "passed",
            "core_accepted_all": core.get("accepted_all") is True,
            "module_accepted_all": module.get("accepted_all") is True,
            "materialize_false_kept_equilibrium_none": module.get("equilibrium_none_all") is True,
            "solution_parity": float(parity.get("x_max_abs", float("inf"))) <= VALIDATION_ATOL,
            "solution_residual_parity": float(parity.get("raw_max_abs", float("inf")))
            <= VALIDATION_ATOL,
            "same_input_residual_parity": float(
                same_input.get("raw_max_abs_to_numba", float("inf"))
            )
            <= residual_atol,
        }
        backend_checks[backend] = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "same_input_residual_atol": residual_atol,
        }
    passed = all(item["status"] == "passed" for item in backend_checks.values())
    return {
        "status": "passed" if passed else "failed",
        "solution_atol": VALIDATION_ATOL,
        "backends": backend_checks,
    }


def _speedup(row: dict[str, Any]) -> float | None:
    if row.get("correctness", {}).get("status") != "passed":
        return None
    engines = row.get("backends", {})
    numba = engines.get("numba", {}).get("core", {}).get("timing_ms", {}).get("median_ms")
    relaxed = engines.get("cxx-relaxed", {}).get("core", {}).get("timing_ms", {}).get("median_ms")
    if numba is None or relaxed is None or relaxed <= 0.0:
        return None
    return float(numba / relaxed)


def _qualification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correctness_passed = bool(rows) and all(
        row.get("correctness", {}).get("status") == "passed" for row in rows
    )
    speedups = [value for row in rows if (value := _speedup(row)) is not None]
    complete_speedups = len(speedups) == len(rows)
    clear_advantage = complete_speedups and bool(speedups) and all(value > 1.0 for value in speedups)
    historical_scale = (
        complete_speedups
        and bool(speedups)
        and min(speedups) >= HISTORICAL_SPEEDUP_RANGE[0]
    )
    return {
        "status": (
            "passed" if correctness_passed and clear_advantage and historical_scale else "failed"
        ),
        "correctness_status": "passed" if correctness_passed else "failed",
        "reference": "historical main README reports approximately 5–11x Cxx advantage",
        "historical_range_x": list(HISTORICAL_SPEEDUP_RANGE),
        "speedups_numba_over_cxx_relaxed": speedups,
        "min_speedup_x": min(speedups) if speedups else None,
        "max_speedup_x": max(speedups) if speedups else None,
        "clear_cxx_relaxed_advantage": clear_advantage,
        "historical_scale_reached": historical_scale,
    }


def _print_table(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="VEQPy GEQDSK backend benchmark", show_lines=False)
    table.add_column("case/config", no_wrap=True)
    for backend in BACKENDS:
        table.add_column(backend, justify="right", no_wrap=True)
    table.add_column("relaxed speedup", justify="right", no_wrap=True)
    for row in rows:
        values = [row["label"]]
        engines = row.get("backends", {})
        for backend in BACKENDS:
            item = engines.get(backend, {})
            if item.get("status") == "skipped":
                values.append("skipped")
                continue
            median = item.get("core", {}).get("timing_ms", {}).get("median_ms")
            status = item.get("status", "n/a")
            values.append(f"{status} {median:.3g} ms" if median is not None else status)
        speedup = _speedup(row)
        values.append("n/a" if speedup is None else f"{speedup:.2f}x")
        table.add_row(*values)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=("solovev", "chease", "efit"))
    parser.add_argument("--config", action="append", choices=("Low", "Medium", "High", "Ref"))
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=REFERENCE_SOLVER_MAXFEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=default_kernel_cache_root())
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 5:
        raise ValueError("--warmup must be at least 5")
    if args.repeat < 100:
        raise ValueError("--repeat must be at least 100")

    cases = selected_cases(args.case)
    configs = selected_configs(args.config)
    selected_backends = tuple(args.backend) if args.backend else RUNNABLE_BACKENDS
    rows: list[dict[str, Any]] = []
    for case_key in cases:
        for config_label in configs:
            case = geqdsk_kernel_case(
                case_key,
                config_label,
                max_evaluations=args.max_evaluations,
            )
            if args.no_run:
                rows.append(
                    {
                        "case": case.case_key,
                        "config": case.config_label,
                        "label": case.label,
                        "status": "planned",
                        "geqdsk": str(case.case_key),
                        "geqdsk_profile_count": int(case.geqdsk.P_psi.size),
                        "topology": case.topology,
                        "solver": case.solver,
                        "signature": case.signature,
                        "boundary_fit": case.boundary_fit,
                        "backends": {
                            backend: _empty_engine(
                                "skipped",
                                reason="--no-run" if backend != "cxx-enzyme" else ENZYME_SKIP_REASON,
                            )
                            for backend in BACKENDS
                        },
                    }
                )
            else:
                rows.append(
                    _measure_case(
                        case,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        artifact_dir=args.artifact_dir.expanduser().resolve(),
                        selected_backends=selected_backends,
                    )
                )

    qualification = {"status": "not-run"} if args.no_run else _qualification(rows)
    payload = {
        "schema": "veqpy.cxx_geqdsk.v2",
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "cold_per_formal_solve": True,
        "timed_paths": {
            "primary": (
                "KernelOutput.elapsed_ms for prepared Module._kernel.solve after Adapter.fill; "
                "matches historical result.elapsed_ms semantics"
            ),
            "wall": "wall time around prepared Module._kernel.solve after Adapter.fill",
            "secondary": "public VEQ.solve(materialize=False)",
        },
        "backends": list(BACKENDS),
        "selected_backends": list(selected_backends),
        "artifact_dir": str(args.artifact_dir.expanduser().resolve()),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "cxx_enzyme": {"status": "skipped", "reason": ENZYME_SKIP_REASON},
        "performance_qualification": qualification,
        "rows": rows,
    }
    write_json(args.output, payload)
    console = Console()
    _print_table(console, rows)
    console.print(f"JSON: {args.output.resolve()}")
    if not args.no_run:
        console.print(f"performance qualification: {qualification['status']}")
    if not args.no_run and qualification["status"] == "failed":
        console.print("cxx-enzyme remains deferred; no enzyme implementation was started")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
