"""
Module: layout.source_binding

Role:
- Bind source stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from source planning and runtime memory refresh.

Notes:
- This module binds preallocated arrays and engine callables; it does not allocate memory.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veqpy.engine.numba_source import (
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
    materialize_profile_owned_psin_source,
    resolve_source_inputs,
    update_fixed_point_psin_query,
)


def build_bound_source_stage_runner(
    *,
    plan,
    case,
    source_workspace,
    profile_workspace,
    residual_workspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable:
    """Bind the source stage runner selected by the source route key."""
    route_key = tuple(plan.source_execution.route_key)
    if route_key == ("PJ2", "psin", "uniform"):
        return _build_pj2_psin_uniform_source_stage_runner(
            plan=plan,
            case=case,
            source_workspace=source_workspace,
            profile_workspace=profile_workspace,
            residual_workspace=residual_workspace,
            source_eval_runner=source_eval_runner,
        )
    return _build_source_stage_runner_shared(
        plan=plan,
        case=case,
        source_workspace=source_workspace,
        profile_workspace=profile_workspace,
        residual_workspace=residual_workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )


def _build_source_stage_runner_shared(
    *,
    plan,
    case,
    source_workspace,
    profile_workspace,
    residual_workspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable:
    source_plan = plan.source_plan
    source_execution = plan.source_execution
    root_fields = residual_workspace.root_fields
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    materialized_heat_input = source_workspace.materialized_heat_input
    materialized_current_input = source_workspace.materialized_current_input
    source_target_root_fields = source_workspace.target_root_fields
    case_R0 = float(case.R0)

    if source_execution.requires_optimized_psin_profile:
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_workspace.psin_query
            source_parameter_query = source_workspace.parameter_query
            psin_profile_fields = profile_workspace.fields_for("psin")
            heat_input = source_plan.heat_input
            current_input = source_plan.current_input
            parameterization_code = source_plan.parameterization_code
            grid_workspace = plan.grid_workspace
            n_axis_fix = int(np.searchsorted(grid_workspace.rho, fix_rho))

            def runner() -> tuple[float, float]:
                if psin_profile_fields.size == 0:
                    raise RuntimeError("psin_profile runtime fields are not initialized")
                materialize_profile_owned_psin_source(
                    psin,
                    psin_r,
                    psin_rr,
                    source_psin_query,
                    source_parameter_query,
                    materialized_heat_input,
                    materialized_current_input,
                    psin_profile_fields,
                    heat_input,
                    current_input,
                    source_workspace.heat_spline_coeff,
                    source_workspace.current_spline_coeff,
                    parameterization_code,
                    grid_workspace.rho,
                    grid_workspace.differentiator,
                    grid_workspace.accumulator,
                    n_axis_fix,
                    source_workspace.barycentric_weights,
                    source_plan.uses_barycentric_interpolation,
                )
                return source_eval_runner(
                    source_target_root_fields,
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
                    case_R0,
                )

            return runner

        source_psin_query = source_workspace.psin_query
        psin_profile_u = profile_workspace.values_for("psin")
        psin_profile_fields = profile_workspace.fields_for("psin")

        def runner() -> tuple[float, float]:
            if psin_profile_fields.size == 0:
                raise RuntimeError("psin_profile runtime fields are not initialized")
            np.copyto(psin, psin_profile_u)
            np.copyto(psin_r, psin_profile_fields[1])
            np.copyto(psin_rr, psin_profile_fields[2])
            np.copyto(source_psin_query, psin)
            if source_plan.parameterization == "identity":
                np.copyto(source_workspace.parameter_query, source_psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_workspace.parameter_query, source_psin_query)
                np.maximum(
                    source_workspace.parameter_query,
                    0.0,
                    out=source_workspace.parameter_query,
                )
                np.sqrt(source_workspace.parameter_query, out=source_workspace.parameter_query)
            else:
                raise ValueError(
                    f"Unsupported source parameterization {source_plan.parameterization!r}"
                )
            resolve_source_inputs(
                source_workspace.materialized_heat_input,
                source_workspace.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
                source_workspace.heat_spline_coeff,
                source_workspace.current_spline_coeff,
                source_workspace.parameter_query,
                source_plan.uses_barycentric_interpolation,
            )
            return source_eval_runner(
                source_target_root_fields,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                case_R0,
            )

        return runner

    def runner() -> tuple[float, float]:
        return source_eval_runner(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            case_R0,
        )

    return runner


def _build_pj2_psin_uniform_source_stage_runner(
    *,
    plan,
    case,
    source_workspace,
    profile_workspace,
    residual_workspace,
    source_eval_runner: Callable,
) -> Callable[[], tuple[float, float]]:
    source_plan = plan.source_plan
    target_root_fields = source_workspace.target_root_fields
    psin_profile_u = profile_workspace.values_for("psin")
    root_fields = residual_workspace.root_fields
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    case_R0 = float(case.R0)

    def runner() -> tuple[float, float]:
        if source_workspace.psin_query[0] < 0.0:
            np.copyto(source_workspace.psin_query, np.asarray(psin_profile_u, dtype=np.float64))
            if source_workspace.psin_query.ndim != 1 or source_workspace.psin_query.size < 2:
                raise ValueError(
                    f"Expected psin query to be 1D with at least two points, "
                    f"got {source_workspace.psin_query.shape}"
                )
            offset = float(source_workspace.psin_query[0])
            scale = float(source_workspace.psin_query[-1] - offset)
            if abs(scale) < 1e-12:
                raise ValueError("psin query does not span a valid normalized flux interval")
            source_workspace.psin_query -= offset
            source_workspace.psin_query /= scale
            source_workspace.psin_query[0] = 0.0
            source_workspace.psin_query[-1] = 1.0
        alpha1 = float("nan")
        alpha2 = float("nan")
        for _ in range(PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER):
            if source_plan.parameterization == "identity":
                np.copyto(source_workspace.parameter_query, source_workspace.psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_workspace.parameter_query, source_workspace.psin_query)
                np.maximum(
                    source_workspace.parameter_query, 0.0, out=source_workspace.parameter_query
                )
                np.sqrt(source_workspace.parameter_query, out=source_workspace.parameter_query)
            else:
                raise ValueError(
                    f"Unsupported source parameterization {source_plan.parameterization!r}"
                )
            resolve_source_inputs(
                source_workspace.materialized_heat_input,
                source_workspace.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
                source_workspace.heat_spline_coeff,
                source_workspace.current_spline_coeff,
                source_workspace.parameter_query,
                source_plan.uses_barycentric_interpolation,
            )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                FFn_psin,
                Pn_psin,
                source_workspace.materialized_heat_input,
                source_workspace.materialized_current_input,
                case_R0,
            )
            if update_fixed_point_psin_query(
                source_workspace.psin_query,
                target_root_fields[0],
                PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
            ):
                break
        np.copyto(source_workspace.psin_query, target_root_fields[0])
        np.copyto(psin, target_root_fields[0])
        np.copyto(psin_r, target_root_fields[1])
        np.copyto(psin_rr, target_root_fields[2])
        return alpha1, alpha2

    return runner
