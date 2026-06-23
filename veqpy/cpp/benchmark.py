from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veqpy.topology import Topology

from .kernel_registry import KernelRegistry, SolverThreadError
from .solver import VEQlibSolver


@dataclass(frozen=True, slots=True)
class LifecycleBenchmarkConfig:
    repeat: int = 5
    warmup: int = 1
    threads: int = 2


def benchmark_kernel_lifecycle(
    topology: Topology,
    *,
    registry: KernelRegistry | None = None,
    config: LifecycleBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Measure module, workspace, and solve costs for the new VEQlib Solver path."""

    config = config or LifecycleBenchmarkConfig()
    registry = registry or KernelRegistry()

    build_start = time.perf_counter_ns()
    artifact = registry.get_or_build(topology)
    build_ms = _elapsed_ms(build_start)

    import_start = time.perf_counter_ns()
    loaded = registry.load_kernel(topology)
    cold_import_ms = _elapsed_ms(import_start)

    registry_hit_samples = []
    for _ in range(config.repeat):
        started = time.perf_counter_ns()
        registry.load_kernel(topology)
        registry_hit_samples.append(_elapsed_us(started))

    ctor_samples = []
    for _ in range(config.repeat):
        started = time.perf_counter_ns()
        loaded.module.KernelSolver()
        ctor_samples.append(_elapsed_us(started))

    solver = VEQlibSolver(topology, registry=registry)
    for _ in range(config.warmup):
        solver.warmup(1)

    first_started = time.perf_counter_ns()
    first_result = solver.solve_direct()
    first_solve_ms = _elapsed_ms(first_started)

    same_case_payload = solver.metadata_json()
    set_case_samples = []
    solve_samples = []
    result = first_result
    for _ in range(config.repeat):
        started = time.perf_counter_ns()
        solver.set_case_json(same_case_payload)
        set_case_samples.append(_elapsed_us(started))

        started = time.perf_counter_ns()
        result = solver.solve_direct()
        solve_samples.append(_elapsed_ms(started))

    thread_report = _thread_report(topology, registry=registry, threads=config.threads)
    cross_thread_guard = _cross_thread_guard_report(solver)

    return {
        "schema": "veqpy.cpp.lifecycle_benchmark.v1",
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "root_dir": str(artifact.root_dir),
            "reused": artifact.reused,
            "built": artifact.built,
        },
        "topology": topology.to_canonical_dict(),
        "metrics": {
            "build_ms": build_ms,
            "cold_import_ms": cold_import_ms,
            "warm_registry_hit_us": _stats(registry_hit_samples),
            "solver_ctor_us": _stats(ctor_samples),
            "first_solve_ms": first_solve_ms,
            "same_case_set_case_us": _stats(set_case_samples),
            "repeated_solve_ms": _stats(solve_samples),
        },
        "result": {
            "first_success": bool(first_result[1]),
            "last_success": bool(result[1]),
            "last_elapsed_ms": solver.last_elapsed_ms,
        },
        "threading": {
            "same_so_multi_thread": thread_report,
            "same_solver_cross_thread_guard": cross_thread_guard,
        },
        "legacy_veqpy_compare": {
            "status": "external_script_available",
            "script": str(Path("veqlib") / "benchmark_pf_psin_uniform_compare.py"),
            "note": "Use this existing script for full VEQPy Operator/Solver comparison while "
            "the MVP KernelSolver still wraps the benchmark PF backend.",
        },
        "case_refresh": {
            "payload_schema": "KernelSolver.metadata_json() round-trip payload",
            "scope": "same-case set_case_json() cost immediately before solve_direct()",
            "note": "Measures real same-topology runtime case refresh through nanobind without "
            "changing the numeric case values.",
        },
    }


def benchmark_kernel_lifecycle_json(
    topology: Topology,
    *,
    registry: KernelRegistry | None = None,
    config: LifecycleBenchmarkConfig | None = None,
) -> str:
    return json.dumps(
        benchmark_kernel_lifecycle(topology, registry=registry, config=config),
        indent=2,
        sort_keys=True,
    )


def _thread_report(
    topology: Topology,
    *,
    registry: KernelRegistry,
    threads: int,
) -> dict[str, Any]:
    if threads <= 0:
        return {"threads": 0, "success": True, "samples_ms": []}

    samples: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()

    def run() -> None:
        try:
            solver = registry.get_thread_solver(topology)
            started = time.perf_counter_ns()
            result = solver.solve_direct()
            elapsed = _elapsed_ms(started)
            if not result[1]:
                raise RuntimeError("thread-local solve did not report success")
            with lock:
                samples.append(elapsed)
        except BaseException as exc:  # noqa: BLE001 - benchmark records worker failures
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")

    workers = [threading.Thread(target=run) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    return {
        "threads": threads,
        "success": not errors and len(samples) == threads,
        "samples_ms": samples,
        "errors": errors,
    }


def _cross_thread_guard_report(solver: VEQlibSolver) -> dict[str, Any]:
    errors: list[str] = []

    def run() -> None:
        try:
            solver.metadata()
        except SolverThreadError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    return {"raised": bool(errors), "error": errors[0] if errors else None}


def _stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"repeat_count": 0, "samples": [], "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "repeat_count": len(samples),
        "samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _elapsed_ms(started_ns: int) -> float:
    return float(time.perf_counter_ns() - started_ns) / 1.0e6


def _elapsed_us(started_ns: int) -> float:
    return float(time.perf_counter_ns() - started_ns) / 1.0e3
