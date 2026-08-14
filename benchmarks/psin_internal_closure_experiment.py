"""Qualify a candidate source-local ``psin`` coordinate closure.

This experiment intentionally lives outside the production route registry.  It
holds a converged geometric-r equilibrium and its outer unknown vector fixed,
expresses the same source profiles in normalized poloidal flux, and applies the
candidate Picard map

``(psin, psin_r) -> source remap -> r source stage -> (psin_new, psin_r_new)``.

The scan answers whether an internal source-coordinate loop can safely replace
the existing profile-owned ``psin`` unknowns.  It does not alter Kernel ABI or
production route ownership.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmarks._common import (
    RouteBenchmarkSpec,
    build_route_mode_inputs,
    route_kernel_case,
    solve_numba_case,
)
from veqpy import KernelSource
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE, source_driver_for

ROUTES = ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ")
CONSTRAINTS = ("none", "ip", "beta", "both")
PF_CONSTRAINTS = ("none", "ip", "beta")


@dataclass(frozen=True, slots=True)
class ClosureResult:
    route: str
    constraint: str
    relaxation: float
    converged: bool
    iterations: int
    defect: float
    value_defect: float
    derivative_defect: float
    elapsed_ms: float
    error: str | None


def _interp(axis: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(PchipInterpolator(axis, values, extrapolate=True)(query), dtype=np.float64)


def _source_on_psin(
    route: str,
    constraint: str,
    equilibrium: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    psin_r = np.asarray(equilibrium.psin_r, dtype=np.float64)
    profiles = {
        "setup_Pn_r": np.asarray(equilibrium.Pn_r, dtype=np.float64) / (4.0e-7 * np.pi),
        "P_r": np.asarray(equilibrium.P_r, dtype=np.float64),
        "FFn_r": np.asarray(equilibrium.FFn_r, dtype=np.float64),
        "FF_r": np.asarray(equilibrium.FF_r, dtype=np.float64),
        "psin_r": psin_r,
        "psi_r": np.asarray(equilibrium.alpha2 * equilibrium.psin_r, dtype=np.float64),
        "Itor": np.asarray(equilibrium.Itor, dtype=np.float64),
        "jtor": np.asarray(equilibrium.jtor, dtype=np.float64),
        "jpara": np.asarray(equilibrium.jpara, dtype=np.float64),
        "jtotal": np.asarray(equilibrium.jtotal, dtype=np.float64),
        "qn": np.asarray(equilibrium.q, dtype=np.float64) * 0.1,
        "q": np.asarray(equilibrium.q, dtype=np.float64),
    }
    p_r, driver_r = build_route_mode_inputs(route, "r", constraint, profiles)
    p_psin = np.asarray(p_r, dtype=np.float64) / psin_r
    driver_psin = np.asarray(driver_r, dtype=np.float64)
    if route == "PF":
        driver_psin = driver_psin / psin_r

    axis = np.concatenate(([0.0], psin, [1.0]))
    p_values = np.concatenate(([p_psin[0]], p_psin, [p_psin[-1]]))
    driver_values = np.concatenate(([driver_psin[0]], driver_psin, [driver_psin[-1]]))
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("reference psin coordinate is not strictly monotone")
    return axis, p_values, driver_values


def _runtime_source(case: Any, p_r: np.ndarray, driver_r: np.ndarray) -> KernelSource:
    kwargs: dict[str, Any] = {
        PRESSURE_DERIVATIVE_BY_COORDINATE[case.topology.coordinate]: p_r,
        source_driver_for(case.topology.route, case.topology.coordinate): driver_r,
    }
    if case.topology.source_uses_ip_constraint:
        kwargs["Ip"] = case.source.Ip
    if case.topology.source_uses_beta_constraint:
        kwargs["beta"] = case.source.beta
    return KernelSource(**kwargs)


def run_case(
    route: str,
    constraint: str,
    *,
    relaxation: float,
    tolerance: float,
    max_iterations: int,
    nr: int,
    nt: int,
) -> ClosureResult:
    case = route_kernel_case(
        RouteBenchmarkSpec(route, "r", "grid", constraint),
        nr=nr,
        nt=nt,
        pj2_f_count=0,
    )
    base, kernel = solve_numba_case(case)
    if not base.success:
        kernel.close()
        return ClosureResult(
            route,
            constraint,
            relaxation,
            False,
            0,
            np.inf,
            np.inf,
            np.inf,
            0.0,
            f"base solve failed: raw_norm={base.raw_norm:.3e}",
        )
    start = time.perf_counter_ns()
    try:
        reference = kernel.build_equilibrium(base.x)
        axis, p_psin, driver_psin = _source_on_psin(route, constraint, reference)
        r = np.asarray(reference.r, dtype=np.float64)
        query = r * r
        query_r = 2.0 * r
        value_defect = np.inf
        derivative_defect = np.inf
        for iteration in range(1, max_iterations + 1):
            mapped_p = _interp(axis, p_psin, query)
            mapped_driver = _interp(axis, driver_psin, query)
            p_r = mapped_p * query_r
            driver_r = mapped_driver * query_r if route == "PF" else mapped_driver
            source = _runtime_source(case, p_r, driver_r)
            kernel.residual(base.x, case.boundary, source)
            candidate = kernel.build_equilibrium(base.x)
            next_query = np.asarray(candidate.psin, dtype=np.float64)
            next_query_r = np.asarray(candidate.psin_r, dtype=np.float64)
            value_defect = float(np.max(np.abs(next_query - query)))
            derivative_defect = float(
                np.max(np.abs(next_query_r - query_r) / (1.0 + np.abs(next_query_r)))
            )
            defect = max(value_defect, derivative_defect)
            if defect <= tolerance:
                elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
                return ClosureResult(
                    route,
                    constraint,
                    relaxation,
                    True,
                    iteration,
                    defect,
                    value_defect,
                    derivative_defect,
                    elapsed_ms,
                    None,
                )
            query += relaxation * (next_query - query)
            query_r += relaxation * (next_query_r - query_r)
            if (
                np.any(~np.isfinite(query))
                or np.any(~np.isfinite(query_r))
                or np.any(query <= 0.0)
                or np.any(query >= 1.0)
                or np.any(query_r <= 0.0)
                or np.any(np.diff(query) <= 0.0)
            ):
                raise ValueError("iterate left the monotone psin domain")
        elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
        return ClosureResult(
            route,
            constraint,
            relaxation,
            False,
            max_iterations,
            max(value_defect, derivative_defect),
            value_defect,
            derivative_defect,
            elapsed_ms,
            "iteration cap reached",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
        return ClosureResult(
            route,
            constraint,
            relaxation,
            False,
            0,
            np.inf,
            np.inf,
            np.inf,
            elapsed_ms,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--relaxation", type=float, action="append")
    args = parser.parse_args()
    relaxations = args.relaxation or [1.0, 0.5, 0.2]
    results: list[ClosureResult] = []
    for route in ROUTES:
        constraints = PF_CONSTRAINTS if route == "PF" else CONSTRAINTS
        for constraint in constraints:
            for relaxation in relaxations:
                result = run_case(
                    route,
                    constraint,
                    relaxation=float(relaxation),
                    tolerance=float(args.tolerance),
                    max_iterations=int(args.max_iterations),
                    nr=int(args.nr),
                    nt=int(args.nt),
                )
                results.append(result)
                print(
                    f"{route:>3} {constraint:>4} w={relaxation:.2f} "
                    f"converged={result.converged!s:<5} iter={result.iterations:2d} "
                    f"defect={result.defect:.3e} elapsed={result.elapsed_ms:.3f} ms"
                )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
