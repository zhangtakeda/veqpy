from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmarks._common import RouteBenchmarkSpec, route_kernel_case
from veqpy import Kernel, KernelRecipe, KernelSource
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE
from veqpy.numerics import build_explicit_source_interpolation_coefficients

EXPLICIT_NODES = np.array(
    [
        0.0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.78,
        0.84,
        0.88,
        0.91,
        0.935,
        0.955,
        0.972,
        0.986,
        1.0,
    ],
    dtype=np.float64,
)
PROFILE_NAMES = ("psin", "F", "P", "q", "jtor")


def _case(kind: str):
    reference = route_kernel_case(
        RouteBenchmarkSpec("PJ1", "rho", "uniform", "ip"),
        nr=24,
        nt=12,
        sample_count=801,
    )
    dense_nodes = np.linspace(0.0, 1.0, 801, dtype=np.float64)
    pedestal = np.exp(-(((dense_nodes - 0.92) / 0.025) ** 2))
    dense_pressure = reference.source.pressure_profile * (1.0 + pedestal)
    dense_driver = reference.source.driver_profile * (1.0 + 0.3 * pedestal)
    if kind == "reference":
        nodes = dense_nodes
        node_mode = "uniform"
    elif kind == "uniform":
        nodes = np.linspace(0.0, 1.0, EXPLICIT_NODES.size, dtype=np.float64)
        node_mode = "uniform"
    elif kind == "explicit":
        nodes = EXPLICIT_NODES
        node_mode = "explicit"
    else:
        raise ValueError(kind)
    pressure = PchipInterpolator(dense_nodes, dense_pressure)(nodes)
    driver = PchipInterpolator(dense_nodes, dense_driver)(nodes)
    topology = replace(
        reference.topology,
        nodes=node_mode,
        sample_count=None if node_mode == "explicit" else nodes.size,
        key=None,
    )
    source = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[reference.topology.coordinate]: pressure},
        jtor=driver,
        source_nodes=nodes if node_mode == "explicit" else None,
        Ip=reference.source.Ip,
    )
    return (
        replace(reference, topology=topology, source=source),
        dense_nodes,
        dense_pressure,
        dense_driver,
    )


def _relative_error(current: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-14)
    return float(np.max(np.abs(current - reference)) / scale)


def _measure(kind: str, *, warmup: int, repeat: int, batch: int):
    case, dense_nodes, dense_pressure, dense_driver = _case(kind)
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        if not result.success:
            raise RuntimeError(f"{kind} solve failed: raw_norm={result.raw_norm:.6e}")
        runtime = kernel._impl._solver.runtime
        residual_out = np.empty(case.topology.x_size, dtype=np.float64)
        for _ in range(warmup):
            for _ in range(batch):
                runtime.residual_into_for_current_case(residual_out, result.x)
        residual_ms = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            for _ in range(batch):
                runtime.residual_into_for_current_case(residual_out, result.x)
            residual_ms.append((time.perf_counter_ns() - started) * 1.0e-6 / batch)
        case_bind_ms = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            runtime.set_case(case.boundary, case.source)
            case_bind_ms.append((time.perf_counter_ns() - started) * 1.0e-6)
        source_clones = (replace(case.source), replace(case.source))
        equivalent_case_bind_ms = []
        for index in range(repeat):
            started = time.perf_counter_ns()
            runtime.set_case(case.boundary, source_clones[index % 2])
            equivalent_case_bind_ms.append((time.perf_counter_ns() - started) * 1.0e-6)
        runtime.set_case(case.boundary, case.source)
        public_residual_ms = []
        for _ in range(repeat):
            started = time.perf_counter_ns()
            kernel.residual_into(residual_out, result.x, case.boundary, case.source)
            public_residual_ms.append((time.perf_counter_ns() - started) * 1.0e-6)
        solve_ms = []
        for _ in range(max(3, repeat // 10)):
            timed = kernel.solve(case.boundary, case.source)
            if not timed.success:
                raise RuntimeError(f"{kind} repeated solve failed")
            solve_ms.append(float(timed.elapsed_ms))
        axis = (
            case.source.source_nodes
            if case.source.source_nodes is not None
            else np.linspace(0.0, 1.0, case.topology.sample_count)
        )
        pressure_dense = PchipInterpolator(axis, case.source.pressure_profile)(dense_nodes)
        driver_dense = PchipInterpolator(axis, case.source.driver_profile)(dense_nodes)
        return {
            "result": result,
            "equilibrium": kernel.build_equilibrium(),
            "source_errors": {
                "pprime": _relative_error(pressure_dense, dense_pressure),
                "jtor": _relative_error(driver_dense, dense_driver),
            },
            "median_hot_residual_ms": median(residual_ms),
            "median_cached_case_bind_ms": median(case_bind_ms),
            "median_equivalent_case_bind_ms": median(equivalent_case_bind_ms),
            "median_public_residual_ms": median(public_residual_ms),
            "median_solve_ms": median(solve_ms),
            "nfev": int(result.nfev),
            "fixed_point_iterations": int(
                kernel._impl._solver.runtime.source_workspace.rho_state[0]
            ),
        }
    finally:
        kernel.close()


def _fixed_node_topology_upper_bound(*, repeat: int) -> list[dict[str, float | int]]:
    """Measure the maximum setup saving available from topology-owned nodes.

    PCHIP coefficients remain value-dependent, so the fixed-node emulation can
    remove only runtime axis validation/copying; both sides still construct the
    two value-dependent coefficient tables.
    """
    rows = []
    for count in (17, 137, 2001):
        parameter = np.linspace(0.0, 1.0, count, dtype=np.float64)
        nodes = 1.0 - (1.0 - parameter) ** 2
        pressure = -np.exp(-nodes) * (1.0 + 0.2 * np.sin(8.0 * nodes))
        driver = 1.5 + 0.3 * nodes * nodes
        dynamic_ms = []
        fixed_ms = []
        for _ in range(max(100, repeat * 10)):
            started = time.perf_counter_ns()
            runtime_nodes = np.asarray(nodes, dtype=np.float64)
            if (
                runtime_nodes.ndim != 1
                or not np.all(np.isfinite(runtime_nodes))
                or np.any(np.diff(runtime_nodes) <= 0.0)
                or runtime_nodes[0] != 0.0
                or runtime_nodes[-1] != 1.0
            ):
                raise RuntimeError("invalid benchmark nodes")
            runtime_nodes = np.array(runtime_nodes, copy=True)
            build_explicit_source_interpolation_coefficients(runtime_nodes, pressure)
            build_explicit_source_interpolation_coefficients(runtime_nodes, driver)
            dynamic_ms.append((time.perf_counter_ns() - started) * 1.0e-6)

            started = time.perf_counter_ns()
            build_explicit_source_interpolation_coefficients(nodes, pressure)
            build_explicit_source_interpolation_coefficients(nodes, driver)
            fixed_ms.append((time.perf_counter_ns() - started) * 1.0e-6)
        dynamic_median = median(dynamic_ms)
        fixed_median = median(fixed_ms)
        rows.append(
            {
                "sample_count": count,
                "runtime_nodes_ms": dynamic_median,
                "topology_nodes_upper_bound_ms": fixed_median,
                "speedup": dynamic_median / fixed_median,
                "saving_fraction": 1.0 - fixed_median / dynamic_median,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    measured = {
        kind: _measure(kind, warmup=args.warmup, repeat=args.repeat, batch=args.batch)
        for kind in ("reference", "uniform", "explicit")
    }
    reference_equilibrium = measured["reference"]["equilibrium"]
    payload = {
        "case": "PJ1/rho/ip with an edge-localized source feature",
        "Nr": 24,
        "Nt": 12,
        "coarse_sample_count": int(EXPLICIT_NODES.size),
        "reference_sample_count": 801,
        "fixed_node_topology_upper_bound": _fixed_node_topology_upper_bound(repeat=args.repeat),
        "modes": {},
    }
    for kind, row in measured.items():
        equilibrium = row.pop("equilibrium")
        result = row.pop("result")
        profile_errors = {
            name: _relative_error(
                np.asarray(getattr(equilibrium, name)),
                np.asarray(getattr(reference_equilibrium, name)),
            )
            for name in PROFILE_NAMES
        }
        payload["modes"][kind] = {
            **row,
            "success": bool(result.success),
            "raw_norm": float(result.raw_norm),
            "profile_errors_vs_reference": profile_errors,
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
