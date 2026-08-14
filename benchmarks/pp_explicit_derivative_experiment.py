"""Compare PP r/explicit derivative ownership against its legacy two-step map."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks._common import (
    ROUTE_BENCHMARK_CONSTRAINTS,
    RouteBenchmarkSpec,
    route_kernel_case,
)
from veqpy import Kernel, KernelRecipe, KernelSource
from veqpy.kernels.numba_kernel import numba_operator

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
DENSE_NODES = np.linspace(0.0, 1.0, 801, dtype=np.float64)
PROFILE_NAMES = ("psin", "psin_r", "psin_rr", "F", "P", "q", "jtor")


def _source_profiles(r: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray]:
    if profile == "edge_feature":
        edge_feature = np.exp(-(((r - 0.92) / 0.03) ** 2))
    elif profile == "smooth":
        edge_feature = np.zeros_like(r)
    else:
        raise ValueError(profile)
    pprime = -2.4e4 * r * (1.0 + 0.25 * r * r + 0.35 * edge_feature)
    psi_r = r * (1.0 + 0.22 * r * r + 0.35 * edge_feature)
    return pprime, psi_r


def _case(nodes: np.ndarray, constraint: str, profile: str):
    base = route_kernel_case(
        RouteBenchmarkSpec("PP", "r", "uniform", constraint),
        nr=24,
        nt=12,
        sample_count=101,
    )
    pprime, psi_r = _source_profiles(nodes, profile)
    topology = replace(base.topology, nodes="explicit", sample_count=None, key=None)
    source = KernelSource(
        P_r=pprime,
        p0=6.0e3,
        psi_r=psi_r,
        source_nodes=nodes,
        Ip=base.source.Ip,
        beta=base.source.beta,
    )
    return replace(base, topology=topology, source=source)


@contextmanager
def _legacy_pp_derivative_binding(enabled: bool):
    if not enabled:
        yield
        return
    original = numba_operator._bind_r_explicit_derivative_source_eval_runner

    def legacy_binder(
        *,
        source_eval_binding,
        f_profile_fields,
        driver_derivative,
        source_kernel,
    ):
        del driver_derivative, source_kernel
        return numba_operator._bind_source_eval_runner_for_fused_backend(
            source_eval_binding=source_eval_binding,
            f_profile_fields=f_profile_fields,
        )

    numba_operator._bind_r_explicit_derivative_source_eval_runner = legacy_binder
    try:
        yield
    finally:
        numba_operator._bind_r_explicit_derivative_source_eval_runner = original


def _measure(
    nodes: np.ndarray,
    constraint: str,
    profile: str,
    *,
    legacy: bool,
    warmup: int,
    repeat: int,
    batch: int,
):
    case = _case(nodes, constraint, profile)
    with _legacy_pp_derivative_binding(legacy):
        kernel = Kernel(
            topology=case.topology,
            recipe=KernelRecipe(backend="numba"),
            config=case.config,
        )
        try:
            result = kernel.solve(case.boundary, case.source)
            equilibrium = kernel.build_equilibrium(result.x)
            runtime = kernel._impl._solver.runtime
            residual = np.empty(case.topology.x_size, dtype=np.float64)
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
                "median_hot_residual_ms": median(timings),
            }
        finally:
            kernel.close()


def _relative_error(current: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-14)
    return float(np.max(np.abs(current - reference)) / scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "case": "PP/r/explicit with edge-localized pressure and psi_r gradients",
        "Nr": 24,
        "Nt": 12,
        "coarse_source_count": int(COARSE_NODES.size),
        "reference_source_count": int(DENSE_NODES.size),
        "profiles": {},
    }
    for profile in ("edge_feature", "smooth"):
        constraints = {}
        for constraint in ROUTE_BENCHMARK_CONSTRAINTS["PP"]:
            reference = _measure(
                DENSE_NODES,
                constraint,
                profile,
                legacy=False,
                warmup=args.warmup,
                repeat=args.repeat,
                batch=args.batch,
            )
            rows = {}
            for name, legacy in (("same_pchip", False), ("legacy_two_step", True)):
                measured = _measure(
                    COARSE_NODES,
                    constraint,
                    profile,
                    legacy=legacy,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    batch=args.batch,
                )
                result = measured["result"]
                equilibrium = measured["equilibrium"]
                rows[name] = {
                    "success": bool(result.success),
                    "raw_norm": float(result.raw_norm),
                    "nfev": int(result.nfev),
                    "median_hot_residual_ms": measured["median_hot_residual_ms"],
                    "profile_errors": {
                        name: _relative_error(
                            np.asarray(getattr(equilibrium, name)),
                            np.asarray(getattr(reference["equilibrium"], name)),
                        )
                        for name in PROFILE_NAMES
                    },
                }
            constraints[constraint] = rows
        payload["profiles"][profile] = constraints

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
