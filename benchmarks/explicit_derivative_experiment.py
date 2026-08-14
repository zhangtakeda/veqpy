"""Rollback-ready experiment for r/explicit value-and-derivative materialization."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from benchmarks._common import RouteBenchmarkSpec, route_kernel_case
from veqpy import Kernel, KernelRecipe, KernelSource
from veqpy.numerics import make_calculus

COARSE_NODES = np.array(
    [
        0.0,
        0.03,
        0.08,
        0.15,
        0.26,
        0.40,
        0.55,
        0.68,
        0.78,
        0.85,
        0.90,
        0.94,
        0.965,
        0.974,
        0.982,
        0.993,
        1.0,
    ],
    dtype=np.float64,
)
PROFILE_NAMES = ("psin", "F", "P", "q", "jtor")


def _dense_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(0.0, 1.0, 2001, dtype=np.float64)
    edge_feature = np.exp(-(((r - 0.92) / 0.03) ** 2))
    minus_pressure_r = 2.4e4 * r * (1.0 + 0.6 * edge_feature)
    pressure_drop = cumulative_trapezoid(minus_pressure_r, r, initial=0.0)
    pressure = 6.0e3 + pressure_drop[-1] - pressure_drop

    current_r = 2.0 * r * (1.0 + 0.6 * edge_feature)
    current = cumulative_trapezoid(current_r, r, initial=0.0)
    current /= current[-1]
    return r, pressure, current


def _case(nodes: np.ndarray):
    base = route_kernel_case(
        RouteBenchmarkSpec("PI", "r", "uniform", "ip"),
        nr=24,
        nt=12,
        sample_count=101,
    )
    dense_r, dense_pressure, dense_current = _dense_profiles()
    pressure = PchipInterpolator(dense_r, dense_pressure)(nodes)
    itor = base.source.Ip * PchipInterpolator(dense_r, dense_current)(nodes)
    topology = replace(base.topology, nodes="explicit", sample_count=None, key=None)
    source = KernelSource(
        p=pressure,
        itor=itor,
        source_nodes=nodes,
        Ip=base.source.Ip,
    )
    return replace(base, topology=topology, source=source)


def _relative_error(current: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-14)
    return float(np.max(np.abs(current - reference)) / scale)


def _measure(
    nodes: np.ndarray,
    *,
    legacy_derivative: bool,
    warmup: int,
    repeat: int,
    batch: int,
):
    case = _case(nodes)
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        residual = np.empty(case.topology.x_size, dtype=np.float64)
        zero = np.zeros(case.topology.x_size, dtype=np.float64)
        kernel.residual_into(residual, zero, case.boundary, case.source)
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        grid = runtime.plan.grid_workspace
        bind_timings = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            runtime.set_case(case.boundary, replace(case.source))
            bind_timings.append((time.perf_counter_ns() - started) * 1.0e-6)
        runtime.set_case(case.boundary, case.source)
        if legacy_derivative:
            # Reconstruct the old two-step behavior in the already-bound
            # workspace: native global D -> PCHIP for pressure, and PCHIP
            # values -> Gauss spectral D for cumulative current.
            legacy_native_pprime = make_calculus(nodes, scheme="spectral")[1] @ (
                case.source.pressure_profile * (4.0e-7 * np.pi)
            )
            workspace.materialized_pprime_input[:] = PchipInterpolator(
                nodes,
                legacy_native_pprime,
            )(grid.r)
            workspace.materialized_driver_derivative[:] = (
                grid.differentiator @ workspace.materialized_driver_input
            )

        result = kernel.solve(case.boundary, case.source)
        equilibrium = kernel.build_equilibrium(result.x)
        for _ in range(warmup):
            for _ in range(batch):
                runtime.residual_into_for_current_case(residual, result.x)
        timings = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            for _ in range(batch):
                runtime.residual_into_for_current_case(residual, result.x)
            timings.append((time.perf_counter_ns() - started) * 1.0e-6 / batch)
        return {
            "result": result,
            "equilibrium": equilibrium,
            "pprime": workspace.materialized_pprime_input.copy(),
            "itor_r": workspace.materialized_driver_derivative.copy(),
            "median_equivalent_case_bind_ms": median(bind_timings),
            "median_hot_residual_ms": median(timings),
        }
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dense_nodes = np.linspace(0.0, 1.0, 801, dtype=np.float64)
    reference = _measure(
        dense_nodes,
        legacy_derivative=False,
        warmup=args.warmup,
        repeat=args.repeat,
        batch=args.batch,
    )
    current = _measure(
        COARSE_NODES,
        legacy_derivative=False,
        warmup=args.warmup,
        repeat=args.repeat,
        batch=args.batch,
    )
    legacy = _measure(
        COARSE_NODES,
        legacy_derivative=True,
        warmup=args.warmup,
        repeat=args.repeat,
        batch=args.batch,
    )
    payload = {
        "case": "PI/r/explicit with p and edge-localized current/pressure gradients",
        "Nr": 24,
        "Nt": 12,
        "coarse_source_count": int(COARSE_NODES.size),
        "reference_source_count": int(dense_nodes.size),
        "modes": {},
    }
    for name, row in (("same_pchip", current), ("legacy_two_step", legacy)):
        result = row["result"]
        equilibrium = row["equilibrium"]
        payload["modes"][name] = {
            "success": bool(result.success),
            "raw_norm": float(result.raw_norm),
            "nfev": int(result.nfev),
            "median_equivalent_case_bind_ms": row["median_equivalent_case_bind_ms"],
            "median_hot_residual_ms": row["median_hot_residual_ms"],
            "pprime_error": _relative_error(row["pprime"], reference["pprime"]),
            "itor_r_error": _relative_error(row["itor_r"], reference["itor_r"]),
            "profile_errors": {
                profile: _relative_error(
                    np.asarray(getattr(equilibrium, profile)),
                    np.asarray(getattr(reference["equilibrium"], profile)),
                )
                for profile in PROFILE_NAMES
            },
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
