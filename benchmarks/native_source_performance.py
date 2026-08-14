"""Measure uniform versus runtime-explicit source costs after Numba fusion."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmarks._common import (
    ROUTE_BENCHMARK_MODES,
    RouteBenchmarkSpec,
    route_kernel_case,
)
from veqpy import Kernel, KernelRecipe, KernelSource
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE


def _case(route: str, coordinate: str, mode: str, sample_count: int):
    reference = route_kernel_case(
        RouteBenchmarkSpec(route, coordinate, "uniform", "ip"),
        nr=24,
        nt=12,
        sample_count=201,
        pj2_f_count=6 if coordinate == "psin" and route in {"PJ2", "PJ3"} else 0,
    )
    reference_nodes = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    nodes = parameter if mode == "uniform" else parameter**1.25
    pressure = PchipInterpolator(reference_nodes, reference.source.pressure_profile)(nodes)
    driver = PchipInterpolator(reference_nodes, reference.source.driver_profile)(nodes)
    topology = replace(
        reference.topology,
        nodes=mode,
        sample_count=sample_count if mode == "uniform" else None,
        key=None,
    )
    source = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: pressure},
        **{reference.source.driver_name: driver},
        source_nodes=nodes if mode == "explicit" else None,
        Ip=reference.source.Ip,
        beta=reference.source.beta,
    )
    return replace(reference, topology=topology, source=source)


def _measure(case, *, warmup: int, repeat: int, batch: int) -> dict[str, float | int]:
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        x0 = np.zeros(case.topology.x_size, dtype=np.float64)
        kernel.solve(case.boundary, case.source, x0=x0)
        result = kernel.solve(case.boundary, case.source, x0=x0)
        if not result.success:
            raise RuntimeError(f"solve failed with raw_norm={result.raw_norm:.6e}")
        runtime = kernel._impl._solver.runtime
        out = np.empty(case.topology.x_size, dtype=np.float64)
        for _ in range(warmup):
            runtime.residual_into_for_current_case(out, result.x)
        timings = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            for _ in range(batch):
                runtime.residual_into_for_current_case(out, result.x)
            timings.append((time.perf_counter_ns() - started) * 1.0e-6 / batch)
        return {
            "hot_residual_ms": median(timings),
            "solve_ms": float(result.elapsed_ms),
            "nfev": int(result.nfev),
            "raw_norm": float(result.raw_norm),
        }
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for coordinate in ("r", "psin", "rho"):
        for route in ROUTE_BENCHMARK_MODES:
            modes = {
                mode: _measure(
                    _case(route, coordinate, mode, args.sample_count),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    batch=args.batch,
                )
                for mode in ("uniform", "explicit")
            }
            rows.append(
                {
                    "route": route,
                    "coordinate": coordinate,
                    "uniform": modes["uniform"],
                    "explicit": modes["explicit"],
                    "hot_ratio": (
                        modes["explicit"]["hot_residual_ms"] / modes["uniform"]["hot_residual_ms"]
                    ),
                }
            )
    payload = {
        "Nr": 24,
        "Nt": 12,
        "sample_count": args.sample_count,
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
