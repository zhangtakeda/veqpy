"""Scan the single-layer PJ2/PJ3 rho, u, C fixed-point domain."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmarks._common import RouteBenchmarkSpec, route_kernel_case
from veqpy import Kernel, KernelRecipe
from veqpy.kernels.numba_kernel import numba_source
from veqpy.kernels.numba_kernel.workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_V_R,
)


@dataclass(frozen=True, slots=True)
class JointResult:
    route: str
    nodes: str
    constraint: str
    coordinate_power: float
    u_scale: float
    current_scale: float
    converged: bool
    iterations: int
    defect: float
    coordinate_defect: float
    physics_defect: float
    error: str | None


def _interpolate(plan, workspace, query, out_p, out_driver) -> None:
    if plan.is_grid_nodes:
        numba_source._global_barycentric_interpolate_pair(
            out_p,
            out_driver,
            plan.scaled_pprime,
            plan.scaled_driver,
            workspace.source_coordinate_nodes,
            workspace.source_coordinate_weights,
            query,
        )
    elif plan.uses_barycentric_interpolation:
        numba_source._local_barycentric_interpolate_pair(
            out_p,
            out_driver,
            plan.scaled_pprime,
            plan.scaled_driver,
            query,
            workspace.barycentric_weights,
        )
    else:
        numba_source._uniform_spline_interpolate_pair(
            out_p,
            out_driver,
            workspace.pprime_spline_coeff,
            workspace.driver_spline_coeff,
            query,
        )


def _scan_initial_state(
    kernel: Kernel,
    result,
    boundary,
    source,
    *,
    coordinate_power: float,
    u_scale: float,
    current_scale: float,
) -> JointResult:
    runtime = kernel._impl._solver.runtime
    runtime.residual_for_current_case(result.x)
    equilibrium = kernel.build_equilibrium(result.x)
    plan = runtime.plan.source_plan
    workspace = runtime.source_workspace
    grid = runtime.plan.grid_workspace
    radial = runtime.geometry_workspace.radial_fields
    r = grid.r
    nr = r.size
    query = r**coordinate_power
    query_r = coordinate_power * r ** (coordinate_power - 1.0)
    next_query = np.empty(nr)
    next_query_r = np.empty(nr)
    sampled_p = np.empty(nr)
    sampled_driver = np.empty(nr)
    p_r = np.empty(nr)
    driver_r = np.empty(nr)
    edge_f = float(boundary.R0 * boundary.B0)
    reference_u = 2.0 * np.log(np.asarray(equilibrium.F) / edge_f)
    reference_C = (
        radial[GEOMETRY_RADIAL_KN] * float(equilibrium.alpha2) * np.asarray(equilibrium.psin_r)
    )
    u = u_scale * reference_u
    C = current_scale * reference_C
    next_u = np.empty(nr)
    next_C = np.empty(nr)
    F = np.empty(nr)
    last_coordinate = np.inf
    last_physics = np.inf
    try:
        for iteration in range(1, numba_source.RHO_FIXED_POINT_MAX_ITER + 1):
            _interpolate(plan, workspace, query, sampled_p, sampled_driver)
            numba_source._prepare_rho_r_inputs(
                p_r,
                driver_r,
                sampled_p,
                sampled_driver,
                query_r,
                False,
            )
            pressure_multiplier = numba_source._pj23_joint_pressure_multiplier(
                p_r,
                float(plan.beta),
                float(boundary.B0),
                float(plan.scaled_p0),
                radial[GEOMETRY_RADIAL_V_R],
                grid.accumulator,
                grid.weights,
                workspace.array_scratch[2],
            )
            last_physics = float(
                numba_source._pj23_joint_fixed_point_map_with_scratch(
                    next_u,
                    next_C,
                    F,
                    u,
                    C,
                    p_r,
                    driver_r,
                    pressure_multiplier,
                    float(boundary.B0),
                    edge_f,
                    radial[GEOMETRY_RADIAL_KN],
                    radial[GEOMETRY_RADIAL_LN_R],
                    radial[GEOMETRY_RADIAL_V_R],
                    grid.weights,
                    grid.accumulator,
                    float(plan.scaled_Ip),
                    plan.route == "PJ3",
                    workspace.array_scratch,
                )
            )
            invalid = numba_source._update_rho_from_u(
                next_query,
                next_query_r,
                F,
                next_u,
                edge_f,
                radial[GEOMETRY_RADIAL_LN_R],
                grid.accumulator,
                grid.weights,
            )
            if invalid:
                raise ValueError(f"non-physical coordinate code {invalid}")
            _, value_defect, derivative_defect = numba_source._rho_fixed_point_defect(
                query,
                query_r,
                next_query,
                next_query_r,
            )
            last_coordinate = max(float(value_defect), float(derivative_defect))
            defect = max(last_coordinate, last_physics)
            if defect <= numba_source.RHO_FIXED_POINT_MAX_RESIDUAL:
                return JointResult(
                    plan.route,
                    plan.nodes,
                    kernel.topology.constraint,
                    coordinate_power,
                    u_scale,
                    current_scale,
                    True,
                    iteration,
                    defect,
                    last_coordinate,
                    last_physics,
                    None,
                )
            np.copyto(query, next_query)
            np.copyto(query_r, next_query_r)
            np.copyto(u, next_u)
            np.copyto(C, next_C)
        return JointResult(
            plan.route,
            plan.nodes,
            kernel.topology.constraint,
            coordinate_power,
            u_scale,
            current_scale,
            False,
            numba_source.RHO_FIXED_POINT_MAX_ITER,
            max(last_coordinate, last_physics),
            last_coordinate,
            last_physics,
            "iteration cap reached",
        )
    except Exception as exc:
        return JointResult(
            plan.route,
            plan.nodes,
            kernel.topology.constraint,
            coordinate_power,
            u_scale,
            current_scale,
            False,
            0,
            np.inf,
            last_coordinate,
            last_physics,
            f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    powers = (0.25, 0.5, 1.0, 1.5, 2.0)
    state_scales = (0.0, 0.5, 1.5)
    rows = []
    for nodes in ("uniform", "grid"):
        for route in ("PJ2", "PJ3"):
            for constraint in ("none", "ip", "beta", "both"):
                case = route_kernel_case(
                    RouteBenchmarkSpec(route, "rho", nodes, constraint),
                    nr=32,
                    nt=16,
                    sample_count=51,
                )
                kernel = Kernel(
                    topology=case.topology,
                    recipe=KernelRecipe(backend="numba"),
                    config=case.config,
                )
                try:
                    result = kernel.solve(case.boundary, case.source)
                    if not result.success:
                        raise RuntimeError(f"base solve failed: {result.raw_norm:.3e}")
                    for power in powers:
                        for u_scale in state_scales:
                            for current_scale in state_scales:
                                rows.append(
                                    _scan_initial_state(
                                        kernel,
                                        result,
                                        case.boundary,
                                        case.source,
                                        coordinate_power=power,
                                        u_scale=u_scale,
                                        current_scale=current_scale,
                                    )
                                )
                finally:
                    kernel.close()
    passed = sum(row.converged for row in rows)
    print(f"converged={passed}/{len(rows)}")
    if passed:
        iterations = [row.iterations for row in rows if row.converged]
        print(f"iteration_range={min(iterations)}--{max(iterations)}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(row) for row in rows], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
