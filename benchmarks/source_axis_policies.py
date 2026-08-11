#!/usr/bin/env python3
"""Characterize magnetic-axis policies for rho-coordinate source routes.

The benchmark starts every route from profiles derived from one converged PF
equilibrium.  It then applies controlled, axis-local perturbations without
changing the public source contract.  This separates three questions:

1. Does the nonlinear solve remain finite and converge?
2. Does a route preserve the profile that is authoritative for that route?
3. Does an axis policy improve or damage the resulting physical equilibrium?

``fix_rho`` is changed through the private Numba runtime only for this
diagnostic.  It is deliberately not a public Kernel option.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    MU0,
    REPO_ROOT,
    RouteBenchmarkSpec,
    benchmark_route_case_diagnostics,
    extract_shape_x,
    route_kernel_case,
    synthetic_route_reference,
)
from veqpy import Kernel, KernelRecipe, KernelSource

ROUTES = ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ")
NODES = ("grid", "uniform")
PERTURBATIONS = (
    "smooth",
    "pressure-regular",
    "pressure-irregular",
    "driver-regular",
    "driver-irregular",
    "driver-invalid-coordinate",
)
DRIVER_FIELDS = {
    "PF": "FFn_r",
    "PP": "psin_r",
    "PI": "Itor",
    "PJ1": "jtor",
    "PJ2": "jpara",
    "PJ3": "jtotal",
    "PQ": "q",
}


def _source_rho(case) -> np.ndarray:
    if case.topology.nodes == "grid":
        from veqpy.numerics import make_quadrature

        rho, _ = make_quadrature(case.topology.Nr, scheme=case.topology.quadrature)
        return np.asarray(rho, dtype=np.float64)
    return np.linspace(0.0, 1.0, case.topology.sample_count, dtype=np.float64)


def _axis_envelope(rho: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-((rho / width) ** 4))


def _perturb_source(
    source: KernelSource,
    *,
    route: str,
    rho: np.ndarray,
    kind: str,
    amplitude: float,
    width: float,
) -> KernelSource:
    if kind == "smooth":
        return source

    pressure = np.asarray(source.pressure_profile, dtype=np.float64).copy()
    driver = np.asarray(source.driver_profile, dtype=np.float64).copy()
    envelope = _axis_envelope(rho, width)
    pressure_scale = max(float(np.max(np.abs(pressure))), 1.0)
    driver_scale = max(float(np.max(np.abs(driver))), 1.0)

    if kind == "pressure-regular":
        pressure *= 1.0 + amplitude * envelope
    elif kind == "pressure-irregular":
        # p_r must vanish linearly at a smooth magnetic axis.  This finite
        # additive bump intentionally violates that parity while remaining a
        # perfectly evaluable profile on the interior Gauss grid.
        pressure += amplitude * pressure_scale * envelope
    elif kind == "driver-regular":
        driver *= 1.0 + amplitude * envelope
    elif kind == "driver-irregular":
        if route in {"PF", "PP", "PI"}:
            # FF_r and psi_r are odd in rho; Itor starts as rho**2.  An
            # additive even bump violates each route's smooth-axis limit.
            driver += amplitude * driver_scale * envelope
        else:
            # Local current density and q are even in rho.  Add an odd local
            # component so their axis derivative no longer vanishes.
            driver += amplitude * driver_scale * (rho / width) * envelope
    elif kind == "driver-invalid-coordinate":
        # Force the route driver through zero near the magnetic axis.  This is
        # not a valid smooth flux coordinate; it tests whether a protection is
        # structural and whether the resulting failure is reported explicitly.
        driver -= 2.0 * driver_scale * envelope
    else:
        raise ValueError(f"unsupported perturbation {kind!r}")

    kwargs: dict[str, Any] = {
        source.pressure_name: pressure,
        source.driver_name: driver,
        "Ip": source.Ip,
        "beta": source.beta,
        "case_name": source.case_name,
    }
    if source.pressure_name == "pprime":
        kwargs["p0"] = source.p0
    return KernelSource(**kwargs)


def _best_scale_error(target: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float:
    target_values = np.asarray(target, dtype=np.float64)[mask]
    actual_values = np.asarray(actual, dtype=np.float64)[mask]
    if target_values.size == 0:
        return 0.0
    denominator = float(np.dot(target_values, target_values))
    if denominator <= 1.0e-30:
        return float(np.linalg.norm(actual_values))
    scale = float(np.dot(target_values, actual_values) / denominator)
    reference = scale * target_values
    norm = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(actual_values - reference) / norm)


def _authority_diagnostics(kernel: Kernel, equilibrium, route: str) -> dict[str, float]:
    runtime = kernel._impl._solver.runtime
    pressure_target = runtime.source_workspace.materialized_pprime_input.copy()
    driver_target = runtime.source_workspace.materialized_driver_input.copy()
    pressure_actual = np.asarray(equilibrium.P_r, dtype=np.float64)
    driver_actual = np.asarray(getattr(equilibrium, DRIVER_FIELDS[route]), dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    all_nodes = np.ones(rho.shape, dtype=np.bool_)
    axis_nodes = rho < 0.05
    outer_nodes = ~axis_nodes
    diagnostics = {
        "pressure_shape_rel_l2": _best_scale_error(pressure_target, pressure_actual, all_nodes),
        "pressure_axis_shape_rel_l2": _best_scale_error(
            pressure_target, pressure_actual, axis_nodes
        ),
        "pressure_outer_shape_rel_l2": _best_scale_error(
            pressure_target, pressure_actual, outer_nodes
        ),
        "driver_shape_rel_l2": _best_scale_error(driver_target, driver_actual, all_nodes),
        "driver_axis_shape_rel_l2": _best_scale_error(driver_target, driver_actual, axis_nodes),
        "driver_outer_shape_rel_l2": _best_scale_error(driver_target, driver_actual, outer_nodes),
        "psin_r_min": float(np.min(equilibrium.psin_r)),
        "psin_min_step": float(np.min(np.diff(equilibrium.psin))),
    }
    if route == "PJ1":
        source_plan = runtime.plan.source_plan
        if np.isfinite(source_plan.scaled_Ip):
            current_integral = float(
                np.dot(driver_target * equilibrium.S_r, equilibrium.grid.weights)
            )
            target = driver_target * source_plan.scaled_Ip / current_integral
        else:
            target = driver_target
        actual = MU0 * driver_actual
        delta = actual - target
        scale = max(float(np.linalg.norm(target)), 1.0e-30)
        point_scale = max(float(np.max(np.abs(target))), 1.0e-30)
        diagnostics["pj1_current_rel_l2"] = float(np.linalg.norm(delta) / scale)
        diagnostics["pj1_current_axis_rel_max"] = float(
            np.max(np.abs(delta[axis_nodes])) / point_scale
        )
    if route == "PI":
        actual_itor = MU0 * np.asarray(equilibrium.Itor, dtype=np.float64)
        differentiated = equilibrium.grid.differentiator @ actual_itor
        implied = (
            -equilibrium.Ln_r * equilibrium.FFn_psin
            - equilibrium.V_r * equilibrium.Pn_psin / (4.0 * np.pi**2)
        ) * (2.0 * np.pi * equilibrium.alpha1)
        delta = implied - differentiated
        scale = max(float(np.linalg.norm(differentiated)), 1.0e-30)
        point_scale = max(float(np.max(np.abs(differentiated))), 1.0e-30)
        diagnostics["pi_current_derivative_rel_l2"] = float(np.linalg.norm(delta) / scale)
        diagnostics["pi_current_derivative_axis_rel_max"] = float(
            np.max(np.abs(delta[axis_nodes])) / point_scale
        )
    return diagnostics


def _run_case(
    *,
    route: str,
    nodes: str,
    nr: int,
    perturbation: str,
    amplitude: float,
    width: float,
    fix_rho: float,
) -> dict[str, Any]:
    spec = RouteBenchmarkSpec(route, "rho", nodes, "ip")
    case = route_kernel_case(spec, nr=nr)
    source = _perturb_source(
        case.source,
        route=route,
        rho=_source_rho(case),
        kind=perturbation,
        amplitude=amplitude,
        width=width,
    )
    case = replace(case, source=source)
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba", layout="degree"),
        config=case.config,
    )
    # Research-only switch.  The public runtime contract remains unchanged.
    kernel._impl._solver.runtime.fix_rho = float(fix_rho)
    row: dict[str, Any] = {
        "route": route,
        "coordinate": "rho",
        "nodes": nodes,
        "constraint": "ip",
        "Nr": int(nr),
        "perturbation": perturbation,
        "fix_rho": float(fix_rho),
    }
    try:
        result = kernel.solve(case.boundary, case.source)
        equilibrium = kernel.build_equilibrium()
        row.update(
            {
                "solve_success": bool(result.success),
                "finite": bool(
                    np.all(np.isfinite(result.raw)) and np.all(np.isfinite(equilibrium.psin_r))
                ),
                "raw_norm": float(result.raw_norm),
                "nfev": int(result.nfev),
                "elapsed_ms": float(result.elapsed_ms),
                "reference": benchmark_route_case_diagnostics(
                    synthetic_route_reference(),
                    equilibrium,
                    extract_shape_x(case.topology, result.x),
                ),
                "authority": _authority_diagnostics(kernel, equilibrium, route),
            }
        )
    except Exception as exc:
        row.update(
            {
                "solve_success": False,
                "finite": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        kernel.close()
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", action="append", choices=ROUTES)
    parser.add_argument("--nodes", action="append", choices=NODES)
    parser.add_argument("--nr", action="append", type=int)
    parser.add_argument("--perturbation", action="append", choices=PERTURBATIONS)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--width", type=float, default=0.05)
    parser.add_argument("--fix-rho", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/source_axis_policy/axis_policy.json",
    )
    args = parser.parse_args(argv)

    routes = tuple(args.route or ROUTES)
    nodes = tuple(args.nodes or NODES)
    radial_counts = tuple(args.nr or (32,))
    perturbations = tuple(args.perturbation or PERTURBATIONS)
    rows = [
        _run_case(
            route=route,
            nodes=node_kind,
            nr=nr,
            perturbation=perturbation,
            amplitude=args.amplitude,
            width=args.width,
            fix_rho=args.fix_rho,
        )
        for nr in radial_counts
        for node_kind in nodes
        for route in routes
        for perturbation in perturbations
    ]
    payload = {
        "schema": "veqpy.source-axis-policy.v1",
        "configuration": {
            "routes": list(routes),
            "nodes": list(nodes),
            "Nr": list(radial_counts),
            "perturbations": list(perturbations),
            "amplitude": float(args.amplitude),
            "width": float(args.width),
            "fix_rho": float(args.fix_rho),
        },
        "summary": {
            "case_count": len(rows),
            "solve_success": sum(bool(row["solve_success"]) for row in rows),
            "finite": sum(bool(row["finite"]) for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
