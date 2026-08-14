"""Compare warmed r and native ``rho = sqrt(Phi_N)`` Numba residual costs."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks._common import (
    ROUTE_BENCHMARK_MODES,
    RouteBenchmarkSpec,
    route_kernel_case,
    runtime_env,
    runtime_platform_payload,
    write_json,
)
from veqpy import Kernel, KernelRecipe


def _hot_residual_ms(
    kernel: Kernel,
    x: np.ndarray,
    *,
    warmup: int,
    repeats: int,
    batches: int,
) -> float:
    out = np.empty(kernel.x_size, dtype=np.float64)
    runtime = kernel._impl._solver.runtime
    for _ in range(warmup):
        runtime.residual_into_for_current_case(out, x)
    timings = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            runtime.residual_into_for_current_case(out, x)
        timings.append((time.perf_counter_ns() - start) * 1.0e-6 / repeats)
    return float(statistics.median(timings))


def _run_coordinate(
    route: str,
    coordinate: str,
    nodes: str,
    *,
    nr: int,
    nt: int,
    sample_count: int,
    warmup: int,
    repeats: int,
    batches: int,
    solve_warmup: int,
    solve_repeats: int,
) -> tuple[dict[str, float | int | bool], dict[str, np.ndarray]]:
    case = route_kernel_case(
        RouteBenchmarkSpec(route, coordinate, nodes, "ip"),
        nr=nr,
        nt=nt,
        sample_count=sample_count,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        equilibrium = kernel.build_equilibrium()
        for _ in range(solve_warmup):
            result = kernel.solve(case.boundary, case.source)
        solve_timings = []
        solve_nfev = []
        for _ in range(solve_repeats):
            result = kernel.solve(case.boundary, case.source)
            solve_timings.append(float(result.elapsed_ms))
            solve_nfev.append(int(result.nfev))
        row: dict[str, float | int | bool] = {
            "success": bool(result.success),
            "raw_norm": float(result.raw_norm),
            "nfev": int(result.nfev),
            "solve_elapsed_ms": float(result.elapsed_ms),
            "solve_median_ms": float(statistics.median(solve_timings)),
            "solve_nfev_median": float(statistics.median(solve_nfev)),
            "hot_residual_ms": _hot_residual_ms(
                kernel,
                result.x,
                warmup=warmup,
                repeats=repeats,
                batches=batches,
            ),
        }
        if coordinate == "rho":
            state = kernel._impl._solver.runtime.source_workspace.rho_state
            row.update(
                local_iterations=int(state[0]),
                local_defect=float(state[1]),
            )
        profiles = {
            "psin": np.asarray(equilibrium.psin, dtype=np.float64),
            "psi_r": np.asarray(equilibrium.alpha2 * equilibrium.psin_r, dtype=np.float64),
            "F": np.asarray(equilibrium.F, dtype=np.float64),
            "P": np.asarray(equilibrium.P, dtype=np.float64),
            "q": np.asarray(equilibrium.q, dtype=np.float64),
            "jtor": np.asarray(equilibrium.jtor, dtype=np.float64),
        }
        return row, profiles
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--solve-warmup", type=int, default=5)
    parser.add_argument("--solve-repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for nodes in ("uniform", "grid"):
        for route in ROUTE_BENCHMARK_MODES:
            r, r_profiles = _run_coordinate(
                route,
                "r",
                nodes,
                nr=args.nr,
                nt=args.nt,
                sample_count=args.sample_count,
                warmup=args.warmup,
                repeats=args.repeats,
                batches=args.batches,
                solve_warmup=args.solve_warmup,
                solve_repeats=args.solve_repeats,
            )
            rho, rho_profiles = _run_coordinate(
                route,
                "rho",
                nodes,
                nr=args.nr,
                nt=args.nt,
                sample_count=args.sample_count,
                warmup=args.warmup,
                repeats=args.repeats,
                batches=args.batches,
                solve_warmup=args.solve_warmup,
                solve_repeats=args.solve_repeats,
            )
            ratio = float(rho["hot_residual_ms"]) / float(r["hot_residual_ms"])
            solve_ratio = float(rho["solve_median_ms"]) / float(r["solve_median_ms"])
            profile_rel_max = {}
            for name, reference in r_profiles.items():
                scale = max(float(np.max(np.abs(reference))), 1.0e-14)
                profile_rel_max[name] = float(
                    np.max(np.abs(rho_profiles[name] - reference)) / scale
                )
            rows.append(
                {
                    "route": route,
                    "nodes": nodes,
                    "r": r,
                    "rho": rho,
                    "hot_residual_ratio": ratio,
                    "solve_time_ratio": solve_ratio,
                    "profile_rel_max_vs_r": profile_rel_max,
                }
            )
            print(
                f"{route:>3}/{nodes:<7} residual={ratio:.2f}x "
                f"solve={solve_ratio:.2f}x "
                f"({r['solve_median_ms']:.4f}->{rho['solve_median_ms']:.4f} ms)"
            )

    payload = {
        "settings": {
            "nr": args.nr,
            "nt": args.nt,
            "sample_count": args.sample_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batches": args.batches,
            "solve_warmup": args.solve_warmup,
            "solve_repeats": args.solve_repeats,
        },
        "platform": runtime_platform_payload(),
        "environment": runtime_env(),
        "rows": rows,
    }
    if args.output is not None:
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
