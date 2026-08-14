"""Qualify local ``rho = sqrt(Phi_N)`` source-coordinate fixed points.

The experiment deliberately uses the existing geometric-r source routes.  A
converged r case supplies one fixed geometry and source profiles expressed in
``s = rho = sqrt(Phi_N)``.  The loop then repeatedly

1. maps the s-coordinate source samples onto the fixed operator r grid,
2. applies the exact derivative chain rule for P_rho (and PF FF_rho),
3. evaluates the existing r source closure at the fixed packed geometry, and
4. rebuilds s(r) from the resulting F and geometry.

No Grad--Shafranov optimization occurs inside the loop.  The measured defect is
therefore the source-stage coordinate closure that a native rho route must
solve during every residual evaluation.
"""

from __future__ import annotations

import argparse
import json
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
class FixedPointResult:
    route: str
    constraint: str
    initial_power: float
    relaxation: float
    converged: bool
    iterations: int
    defect: float
    value_defect: float
    derivative_defect: float
    error: str | None


def _interp(axis: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(PchipInterpolator(axis, values, extrapolate=True)(query), dtype=np.float64)


def _source_on_s(
    route: str,
    constraint: str,
    equilibrium: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(equilibrium.rho, dtype=np.float64)
    s_r = np.asarray(equilibrium.rho_r, dtype=np.float64)
    profiles = {
        "setup_Pn_r": np.asarray(equilibrium.Pn_r, dtype=np.float64) / (4.0e-7 * np.pi),
        "P_r": np.asarray(equilibrium.P_r, dtype=np.float64),
        "FFn_r": np.asarray(equilibrium.FFn_r, dtype=np.float64),
        "FF_r": np.asarray(equilibrium.FF_r, dtype=np.float64),
        "psin_r": np.asarray(equilibrium.psin_r, dtype=np.float64),
        "psi_r": np.asarray(equilibrium.alpha2 * equilibrium.psin_r, dtype=np.float64),
        "Itor": np.asarray(equilibrium.Itor, dtype=np.float64),
        "jtor": np.asarray(equilibrium.jtor, dtype=np.float64),
        "jpara": np.asarray(equilibrium.jpara, dtype=np.float64),
        "jtotal": np.asarray(equilibrium.jtotal, dtype=np.float64),
        "qn": np.asarray(equilibrium.q, dtype=np.float64) * 0.1,
        "q": np.asarray(equilibrium.q, dtype=np.float64),
    }
    p_r, driver_r = build_route_mode_inputs(route, "r", constraint, profiles)
    p_s = np.asarray(p_r, dtype=np.float64) / s_r
    driver_s = np.asarray(driver_r, dtype=np.float64)
    if route == "PF":
        driver_s = driver_s / s_r

    # The native grid is open.  Anchor interpolation with analytic coordinate
    # endpoints and one-sided profile extrapolation without altering nodal data.
    s_axis = np.concatenate(([0.0], s, [1.0]))
    p_axis = np.concatenate(([p_s[0]], p_s, [p_s[-1]]))
    d_axis = np.concatenate(([driver_s[0]], driver_s, [driver_s[-1]]))
    if np.any(np.diff(s_axis) <= 0.0):
        raise ValueError("reference rho coordinate is not strictly monotone")
    return s_axis, p_axis, d_axis


def _runtime_source(
    *,
    case: Any,
    p_r: np.ndarray,
    driver_r: np.ndarray,
) -> KernelSource:
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
    initial_power: float,
    relaxation: float,
    tolerance: float,
    max_iterations: int,
) -> FixedPointResult:
    spec = RouteBenchmarkSpec(route, "r", "grid", constraint)
    case = route_kernel_case(spec, nr=32, nt=16, pj2_f_count=0)
    base, kernel = solve_numba_case(case)
    if not base.success:
        kernel.close()
        return FixedPointResult(
            route,
            constraint,
            initial_power,
            relaxation,
            False,
            0,
            np.inf,
            np.inf,
            np.inf,
            f"base solve failed: raw_norm={base.raw_norm:.3e}",
        )
    try:
        reference = kernel.build_equilibrium(base.x)
        source_s, p_s, driver_s = _source_on_s(route, constraint, reference)
        r = np.asarray(reference.r, dtype=np.float64)
        query = r**initial_power
        query_r = initial_power * r ** (initial_power - 1.0)
        value_defect = np.inf
        derivative_defect = np.inf
        for iteration in range(1, max_iterations + 1):
            mapped_p = _interp(source_s, p_s, query)
            mapped_driver = _interp(source_s, driver_s, query)
            p_r = mapped_p * query_r
            driver_r = mapped_driver * query_r if route == "PF" else mapped_driver
            source = _runtime_source(case=case, p_r=p_r, driver_r=driver_r)
            kernel.residual(base.x, case.boundary, source)
            candidate = kernel.build_equilibrium(base.x)
            next_query = np.asarray(candidate.rho, dtype=np.float64)
            next_query_r = np.asarray(candidate.rho_r, dtype=np.float64)
            value_defect = float(np.max(np.abs(next_query - query)))
            derivative_defect = float(
                np.max(np.abs(next_query_r - query_r) / (1.0 + np.abs(next_query_r)))
            )
            defect = max(value_defect, derivative_defect)
            if defect <= tolerance:
                return FixedPointResult(
                    route,
                    constraint,
                    initial_power,
                    relaxation,
                    True,
                    iteration,
                    defect,
                    value_defect,
                    derivative_defect,
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
                return FixedPointResult(
                    route,
                    constraint,
                    initial_power,
                    relaxation,
                    False,
                    iteration,
                    defect,
                    value_defect,
                    derivative_defect,
                    "iterate left the monotone coordinate domain",
                )
        return FixedPointResult(
            route,
            constraint,
            initial_power,
            relaxation,
            False,
            max_iterations,
            max(value_defect, derivative_defect),
            value_defect,
            derivative_defect,
            "iteration cap reached",
        )
    except Exception as exc:  # qualification records failures instead of aborting the scan
        return FixedPointResult(
            route,
            constraint,
            initial_power,
            relaxation,
            False,
            0,
            np.inf,
            np.inf,
            np.inf,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--initial-power", type=float, action="append")
    parser.add_argument("--relaxation", type=float, action="append")
    args = parser.parse_args()
    initial_powers = args.initial_power or [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    relaxations = args.relaxation or [1.0, 0.7, 0.5]
    results: list[FixedPointResult] = []
    for route in ROUTES:
        constraints = PF_CONSTRAINTS if route == "PF" else CONSTRAINTS
        for constraint in constraints:
            for initial_power in initial_powers:
                for relaxation in relaxations:
                    result = run_case(
                        route,
                        constraint,
                        initial_power=float(initial_power),
                        relaxation=float(relaxation),
                        tolerance=float(args.tolerance),
                        max_iterations=int(args.max_iterations),
                    )
                    results.append(result)
                    print(
                        f"{route:>3} {constraint:>4} p={initial_power:.2f} "
                        f"w={relaxation:.2f} converged={result.converged!s:<5} "
                        f"iter={result.iterations:2d} defect={result.defect:.3e}"
                    )
    payload = [asdict(result) for result in results]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
