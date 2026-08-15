"""
Module: veqpy.kernels.numba_kernel.source_binding

Role:
- Bind source stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from source planning and runtime memory refresh.

Notes:
- This module binds preallocated arrays and backend callables; it does not allocate memory.
- Numerical kernels remain private to this backend package.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit

from . import numba_source
from .workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_V_R,
)


def _retained_source_interpolation_code(source_plan) -> int:
    if source_plan.is_explicit_nodes:
        return numba_source.RETAINED_SOURCE_EXPLICIT_BARYCENTRIC
    if source_plan.is_grid_nodes:
        return numba_source.RETAINED_SOURCE_GRID_BARYCENTRIC
    if source_plan.uses_barycentric_interpolation:
        return numba_source.RETAINED_SOURCE_LOCAL_BARYCENTRIC
    return numba_source.RETAINED_SOURCE_UNIFORM_SPLINE


# These two dispatchers accept another Numba dispatcher as a route argument.
# Numba can compile that signature in memory, but cannot safely serialize the
# dispatcher type into its disk cache (``underlying object has vanished`` on a
# later process).  The global dispatcher still reuses each compiled overload
# for the rest of the process, so this changes cold-start persistence only.
@njit(nogil=True)
def _evaluate_source_kernel_impl(
    source_kernel,
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    f_profile_fields: np.ndarray,
    scaled_p0: float,
    scaled_Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
    pressure_state: np.ndarray,
    pressure_derivative_work: np.ndarray,
    driver_derivative_work: np.ndarray,
    scale_driver_by_alpha2: bool,
) -> tuple[float, float]:
    if coordinate_code != numba_source.PSIN_COORDINATE:
        alpha1, alpha2 = source_kernel(
            root_fields,
            FFn_psin,
            Pn_psin,
            pprime_input,
            driver_input,
            coordinate_code,
            R0,
            B0,
            weights,
            differentiator,
            accumulator,
            grid_radial_fields,
            n_axis_fix,
            radial_fields,
            surface_fields,
            f_profile_fields,
            scaled_p0,
            scaled_Ip,
            beta,
            array_scratch,
            matrix_scratch,
        )
        alpha1 = numba_source.finalize_pressure_normalization(
            FFn_psin,
            Pn_psin,
            pprime_input,
            coordinate_code,
            scaled_p0,
            beta,
            alpha1,
            alpha2,
            root_fields[1],
            accumulator,
            weights,
            array_scratch[0],
            array_scratch[1],
            pressure_state,
        )
        return alpha1, alpha2

    # Existing psin route kernels consume P_psi and, for PF, FF_psi. Public
    # inputs are P_psin and FF_psin, so alpha2 and the effective derivatives
    # form one small scalar fixed point. Keep it local and allocation-free.
    alpha2_guess = 1.0
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(numba_source.PSIN_DERIVATIVE_FIXED_POINT_MAX_ITER):
        if not np.isfinite(alpha2_guess) or abs(alpha2_guess) <= 1.0e-14:
            break
        derivative_scale = abs(alpha2_guess) if scale_driver_by_alpha2 else alpha2_guess
        for i in range(pprime_input.shape[0]):
            pressure_derivative_work[i] = pprime_input[i] / derivative_scale
            if scale_driver_by_alpha2:
                driver_derivative_work[i] = driver_input[i] / derivative_scale
        effective_driver = driver_derivative_work if scale_driver_by_alpha2 else driver_input
        alpha1, alpha2 = source_kernel(
            root_fields,
            FFn_psin,
            Pn_psin,
            pressure_derivative_work,
            effective_driver,
            coordinate_code,
            R0,
            B0,
            weights,
            differentiator,
            accumulator,
            grid_radial_fields,
            n_axis_fix,
            radial_fields,
            surface_fields,
            f_profile_fields,
            scaled_p0,
            scaled_Ip,
            beta,
            array_scratch,
            matrix_scratch,
        )
        alpha1 = numba_source.finalize_pressure_normalization(
            FFn_psin,
            Pn_psin,
            pressure_derivative_work,
            coordinate_code,
            scaled_p0,
            beta,
            alpha1,
            alpha2,
            root_fields[1],
            accumulator,
            weights,
            array_scratch[0],
            array_scratch[1],
            pressure_state,
        )
        next_scale = abs(alpha2) if scale_driver_by_alpha2 else alpha2
        defect = abs(next_scale - alpha2_guess) / max(abs(next_scale), 1.0e-14)
        if defect <= numba_source.PSIN_DERIVATIVE_FIXED_POINT_MAX_RESIDUAL:
            return alpha1, alpha2
        # Scaling both PF derivatives gives the null-constraint map the
        # homogeneous form g(a)~C/a. Direct Picard then has derivative -1 and
        # can enter an exact two-cycle; equal-weight averaging removes that
        # mode without introducing a runtime tuning parameter. Iterate only
        # the magnitude because an Ip constraint independently owns the sign.
        if scale_driver_by_alpha2:
            # psin is oriented from axis to edge regardless of the physical
            # flux direction. Close its scale magnitude here; Ip retains
            # ownership of the signed alpha2 gauge returned by the route.
            alpha2_guess = 0.5 * (alpha2_guess + abs(alpha2))
        else:
            alpha2_guess = alpha2
    raise ValueError("psin source-derivative scalar closure did not reach tolerance")


@njit(nogil=True)
def _run_rho_source_closure_impl(
    source_kernel,
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    query: np.ndarray,
    query_next: np.ndarray,
    query_r: np.ndarray,
    query_r_next: np.ndarray,
    sampled_pprime: np.ndarray,
    sampled_driver: np.ndarray,
    pprime_r: np.ndarray,
    driver_r: np.ndarray,
    f_profile: np.ndarray,
    f2_profile: np.ndarray,
    state: np.ndarray,
    r: np.ndarray,
    Ln_r: np.ndarray,
    source_values0: np.ndarray,
    source_values1: np.ndarray,
    source_nodes: np.ndarray,
    source_weights: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    local_weights: np.ndarray,
    interpolation_code: int,
    differentiate_pressure: bool,
    transform_driver_derivative: bool,
    edge_f: float,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    f_profile_fields: np.ndarray,
    scaled_p0: float,
    scaled_Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
    pressure_state: np.ndarray,
    max_iterations: int,
    max_residual: float,
) -> tuple[float, float]:
    # Seed the source-coordinate fixed point from the current geometry and the
    # exactly known edge field.  This is the toroidal-flux coordinate obtained
    # from F(r)=F_edge, so it is a better deterministic approximation than
    # s=r without borrowing state from an earlier residual evaluation.
    for i in range(f2_profile.shape[0]):
        f2_profile[i] = 0.0
    seed_invalid = numba_source._update_rho_from_u(
        query,
        query_r,
        f_profile,
        f2_profile,
        edge_f,
        Ln_r,
        accumulator,
        weights,
    )
    # Nonlinear outer solvers can visit non-physical trial geometry.  Preserve
    # the old convergence domain by falling back to the always-defined neutral
    # coordinate instead of rejecting such a trial at the initial-guess stage.
    if seed_invalid != 0:
        for i in range(query.shape[0]):
            query[i] = r[i]
            query_r[i] = 1.0
    state.fill(0.0)
    for iteration in range(1, max_iterations + 1):
        numba_source._interpolate_retained_source_pair_impl(
            sampled_pprime,
            sampled_driver,
            source_values0,
            source_values1,
            source_nodes,
            source_weights,
            coeff0,
            coeff1,
            local_weights,
            query,
            interpolation_code,
            differentiate_pressure,
        )
        numba_source._prepare_rho_r_inputs(
            pprime_r,
            driver_r,
            sampled_pprime,
            sampled_driver,
            query_r,
            transform_driver_derivative,
        )
        alpha1, alpha2 = _evaluate_source_kernel_impl(
            source_kernel,
            root_fields,
            FFn_psin,
            Pn_psin,
            pprime_r,
            driver_r,
            numba_source.R_COORDINATE,
            R0,
            B0,
            weights,
            differentiator,
            accumulator,
            grid_radial_fields,
            n_axis_fix,
            radial_fields,
            surface_fields,
            f_profile_fields,
            scaled_p0,
            scaled_Ip,
            beta,
            array_scratch,
            matrix_scratch,
            pressure_state,
            sampled_pprime,
            sampled_driver,
            False,
        )
        invalid = numba_source._update_rho_from_source(
            query_next,
            query_r_next,
            f_profile,
            f2_profile,
            FFn_psin,
            root_fields[1],
            alpha1,
            alpha2,
            edge_f,
            Ln_r,
            accumulator,
            weights,
        )
        if invalid != 0:
            raise ValueError("rho source closure produced a non-physical coordinate")
        defect, value_defect, derivative_defect = numba_source._rho_fixed_point_defect(
            query,
            query_r,
            query_next,
            query_r_next,
        )
        state[0] = iteration
        state[1] = defect
        state[2] = value_defect
        state[3] = derivative_defect
        state[4] = 0.0
        if defect <= max_residual:
            return alpha1, alpha2
        for i in range(query.shape[0]):
            query[i] = query_next[i]
            query_r[i] = query_r_next[i]
    raise ValueError("rho source closure did not reach tolerance within iteration limit")


@njit(cache=True, nogil=True)
def _run_pj23_rho_joint_closure_impl(
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    query: np.ndarray,
    query_next: np.ndarray,
    query_r: np.ndarray,
    query_r_next: np.ndarray,
    sampled_pprime: np.ndarray,
    sampled_driver: np.ndarray,
    pprime_r: np.ndarray,
    driver_r: np.ndarray,
    f_profile: np.ndarray,
    u: np.ndarray,
    u_next: np.ndarray,
    current: np.ndarray,
    current_next: np.ndarray,
    state: np.ndarray,
    source_values0: np.ndarray,
    source_values1: np.ndarray,
    source_nodes: np.ndarray,
    source_weights: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    local_weights: np.ndarray,
    interpolation_code: int,
    differentiate_pressure: bool,
    array_scratch: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    differentiator: np.ndarray,
    F_fields: np.ndarray,
    edge_f: float,
    use_jtotal_semantics: bool,
    scaled_p0: float,
    scaled_Ip: float,
    beta: float,
    R0: float,
    B0: float,
    pressure_state: np.ndarray,
    max_iterations: int,
    max_residual: float,
) -> tuple[float, float]:
    for i in range(query.shape[0]):
        u[i] = 0.0
        current[i] = 0.0
    seed_invalid = numba_source._update_rho_from_u(
        query,
        query_r,
        f_profile,
        u,
        edge_f,
        Ln_r,
        accumulator,
        weights,
    )
    if seed_invalid != 0:
        for i in range(query.shape[0]):
            query[i] = r[i]
            query_r[i] = 1.0
    state.fill(0.0)
    for iteration in range(1, max_iterations + 1):
        numba_source._interpolate_retained_source_pair_impl(
            sampled_pprime,
            sampled_driver,
            source_values0,
            source_values1,
            source_nodes,
            source_weights,
            coeff0,
            coeff1,
            local_weights,
            query,
            interpolation_code,
            differentiate_pressure,
        )
        numba_source._prepare_rho_r_inputs(
            pprime_r,
            driver_r,
            sampled_pprime,
            sampled_driver,
            query_r,
            False,
        )
        pressure_multiplier = numba_source._pj23_joint_pressure_multiplier(
            pprime_r,
            beta,
            B0,
            scaled_p0,
            V_r,
            accumulator,
            weights,
            array_scratch[2],
        )
        strict_defect = numba_source._pj23_joint_fixed_point_map_with_scratch(
            u_next,
            current_next,
            F_fields[0],
            u,
            current,
            pprime_r,
            driver_r,
            pressure_multiplier,
            B0,
            edge_f,
            Kn,
            Ln_r,
            V_r,
            weights,
            accumulator,
            scaled_Ip,
            use_jtotal_semantics,
            array_scratch,
        )
        invalid = numba_source._update_rho_from_u(
            query_next,
            query_r_next,
            f_profile,
            u_next,
            edge_f,
            Ln_r,
            accumulator,
            weights,
        )
        if invalid != 0:
            raise ValueError("joint rho closure produced a non-physical coordinate")
        _, value_defect, derivative_defect = numba_source._rho_fixed_point_defect(
            query,
            query_r,
            query_next,
            query_r_next,
        )
        defect = max(value_defect, derivative_defect, strict_defect)
        state[0] = iteration
        state[1] = defect
        state[2] = value_defect
        state[3] = derivative_defect
        state[4] = strict_defect
        if defect <= max_residual:
            # The just-evaluated source, F, and physics right-hand side all
            # belong to the current state, whose map defect passed the gate.
            # Publish that state directly, as the strict r closure does,
            # instead of advancing once and recomputing the complete map only
            # to refresh scratch arrays. Keep ``query_next`` as the published
            # coordinate for diagnostics.
            for i in range(query.shape[0]):
                query_next[i] = query[i]
                query_r_next[i] = query_r[i]
            return numba_source._publish_pj23_joint_state(
                root_fields,
                FFn_psin,
                Pn_psin,
                F_fields,
                u,
                current,
                pprime_r,
                pressure_multiplier,
                scaled_p0,
                beta,
                R0,
                B0,
                Kn,
                weights,
                differentiator,
                accumulator,
                array_scratch,
                pressure_state,
            )
        for i in range(query.shape[0]):
            query[i] = query_next[i]
            query_r[i] = query_r_next[i]
            u[i] = u_next[i]
            current[i] = current_next[i]
    raise ValueError("joint rho closure did not reach tolerance within iteration limit")


def _interpolate_retained_source(
    *,
    source_plan,
    source_workspace,
    query: np.ndarray,
    out_pprime: np.ndarray,
    out_driver: np.ndarray,
) -> None:
    """Evaluate the native source representation at one changing coordinate."""
    if source_plan.is_explicit_nodes:
        values0 = (
            source_plan.scaled_pressure
            if source_plan.scaled_pressure is not None
            else source_plan.scaled_pprime
        )
        numba_source._explicit_local_barycentric_interpolate_pair_with_derivatives(
            out_pprime,
            out_driver,
            out_pprime,
            out_driver,
            values0,
            source_plan.scaled_driver,
            source_workspace.source_coordinate_nodes,
            source_workspace.source_coordinate_weights,
            query,
            source_plan.scaled_pressure is not None,
            False,
        )
    elif source_plan.is_grid_nodes:
        numba_source._global_barycentric_interpolate_pair(
            out_pprime,
            out_driver,
            source_plan.scaled_pprime,
            source_plan.scaled_driver,
            source_workspace.source_coordinate_nodes,
            source_workspace.source_coordinate_weights,
            query,
        )
    elif source_plan.uses_barycentric_interpolation:
        numba_source._local_barycentric_interpolate_pair(
            out_pprime,
            out_driver,
            source_plan.scaled_pprime,
            source_plan.scaled_driver,
            query,
            source_workspace.barycentric_weights,
        )
    else:
        numba_source._uniform_spline_interpolate_pair(
            out_pprime,
            out_driver,
            source_workspace.pprime_spline_coeff,
            source_workspace.driver_spline_coeff,
            query,
        )


def build_bound_source_stage_runner(
    *,
    plan,
    case,
    grid_workspace,
    source_workspace,
    profile_workspace,
    geometry_workspace,
    residual_workspace,
    fix_r: float,
    source_eval_runner: Callable,
) -> Callable:
    """Bind the source stage runner selected by the source route key."""
    route_key = tuple(plan.source_execution.route_key)
    if plan.source_execution.requires_rho_closure:
        if route_key[0] in {"PJ2", "PJ3"}:
            return _build_pj23_rho_joint_source_stage_runner(
                plan=plan,
                case=case,
                grid_workspace=grid_workspace,
                source_workspace=source_workspace,
                profile_workspace=profile_workspace,
                geometry_workspace=geometry_workspace,
                residual_workspace=residual_workspace,
            )
        return _build_rho_source_stage_runner(
            plan=plan,
            case=case,
            grid_workspace=grid_workspace,
            source_workspace=source_workspace,
            profile_workspace=profile_workspace,
            geometry_workspace=geometry_workspace,
            residual_workspace=residual_workspace,
            fix_r=fix_r,
            source_eval_runner=source_eval_runner,
        )
    if (
        route_key[0] in {"PJ2", "PJ3"}
        and route_key[1] == "psin"
        and route_key[2] in {"uniform", "explicit"}
    ):
        # This route is stateful because source samples are queried in the psin
        # produced by the same route.  It needs a fixed-point wrapper around the
        # normal source kernel instead of the shared single-pass runner.
        return _build_pj2_psin_uniform_source_stage_runner(
            plan=plan,
            case=case,
            grid_workspace=grid_workspace,
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
        fix_r=fix_r,
        source_eval_runner=source_eval_runner,
    )


def _build_rho_source_stage_runner(
    *,
    plan,
    case,
    grid_workspace,
    source_workspace,
    profile_workspace,
    geometry_workspace,
    residual_workspace,
    fix_r: float,
    source_eval_runner: Callable,
) -> Callable[[], tuple[float, float]]:
    """Bind deterministic local source-coordinate closure for sqrt(Phi_N)."""
    source_plan = plan.source_plan
    root_fields = residual_workspace.root_fields
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    query = source_workspace.rho_query
    query_next = source_workspace.rho_query_next
    query_r = source_workspace.rho_derivative
    query_r_next = source_workspace.rho_derivative_next
    sampled_pprime = source_workspace.materialized_pprime_input
    sampled_driver = source_workspace.materialized_driver_input
    pprime_r = source_workspace.rho_pprime
    driver_r = source_workspace.rho_driver
    f_profile = source_workspace.rho_f
    f2_profile = source_workspace.rho_f2
    state = source_workspace.rho_state
    r = grid_workspace.r
    Ln_r = geometry_workspace.radial_fields[GEOMETRY_RADIAL_LN_R]
    route_is_pf = source_plan.route == "PF"
    case_R0 = float(case.R0)
    edge_f = float(case.R0 * case.B0)
    interpolation_code = _retained_source_interpolation_code(source_plan)
    source_kernel = source_plan.kernel
    weights = grid_workspace.weights
    differentiator = grid_workspace.differentiator
    accumulator = grid_workspace.accumulator
    grid_radial_fields = grid_workspace.radial_fields
    radial_fields = geometry_workspace.radial_fields
    surface_fields = geometry_workspace.surface_fields
    f_profile_fields = profile_workspace.fields_for("F")
    n_axis_fix = int(np.searchsorted(r, fix_r))
    case_B0 = float(case.B0)
    scaled_p0 = float(source_plan.scaled_p0)
    scaled_Ip = float(source_plan.scaled_Ip)
    beta = float(source_plan.beta)
    coefficient0 = (
        source_workspace.pressure_spline_coeff
        if source_plan.scaled_pressure is not None
        else source_workspace.pprime_spline_coeff
    )
    differentiate_pressure = source_plan.scaled_pressure is not None

    def runner() -> tuple[float, float]:
        return _run_rho_source_closure_impl(
            source_kernel,
            root_fields,
            FFn_psin,
            Pn_psin,
            query,
            query_next,
            query_r,
            query_r_next,
            sampled_pprime,
            sampled_driver,
            pprime_r,
            driver_r,
            f_profile,
            f2_profile,
            state,
            r,
            Ln_r,
            source_plan.scaled_pprime,
            source_plan.scaled_driver,
            source_workspace.source_coordinate_nodes,
            source_workspace.source_coordinate_weights,
            coefficient0,
            source_workspace.driver_spline_coeff,
            source_workspace.barycentric_weights,
            interpolation_code,
            differentiate_pressure,
            route_is_pf,
            edge_f,
            case_R0,
            case_B0,
            weights,
            differentiator,
            accumulator,
            grid_radial_fields,
            n_axis_fix,
            radial_fields,
            surface_fields,
            f_profile_fields,
            scaled_p0,
            scaled_Ip,
            beta,
            source_workspace.array_scratch,
            source_workspace.matrix_scratch,
            source_workspace.pressure_state,
            int(numba_source.RHO_FIXED_POINT_MAX_ITER),
            float(numba_source.RHO_FIXED_POINT_MAX_RESIDUAL),
        )

    return runner


def _build_pj23_rho_joint_source_stage_runner(
    *,
    plan,
    case,
    grid_workspace,
    source_workspace,
    profile_workspace,
    geometry_workspace,
    residual_workspace,
) -> Callable[[], tuple[float, float]]:
    """Bind one joint ``(sqrt(Phi_N), u, C)`` fixed-point map for PJ2/PJ3."""
    source_plan = plan.source_plan
    root_fields = residual_workspace.root_fields
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    query = source_workspace.rho_query
    query_next = source_workspace.rho_query_next
    query_r = source_workspace.rho_derivative
    query_r_next = source_workspace.rho_derivative_next
    sampled_pprime = source_workspace.materialized_pprime_input
    sampled_driver = source_workspace.materialized_driver_input
    pprime_r = source_workspace.rho_pprime
    driver_r = source_workspace.rho_driver
    f_profile = source_workspace.rho_f
    u = source_workspace.rho_u
    u_next = source_workspace.rho_u_next
    current = source_workspace.rho_current
    current_next = source_workspace.rho_current_next
    state = source_workspace.rho_state
    array_scratch = source_workspace.array_scratch
    radial_fields = geometry_workspace.radial_fields
    V_r = radial_fields[GEOMETRY_RADIAL_V_R]
    Kn = radial_fields[GEOMETRY_RADIAL_KN]
    Ln_r = radial_fields[GEOMETRY_RADIAL_LN_R]
    r = grid_workspace.r
    weights = grid_workspace.weights
    accumulator = grid_workspace.accumulator
    differentiator = grid_workspace.differentiator
    F_fields = profile_workspace.fields_for("F")
    edge_f = float(case.R0 * case.B0)
    use_jtotal_semantics = source_plan.route == "PJ3"
    scaled_p0 = float(source_plan.scaled_p0)
    scaled_Ip = float(source_plan.scaled_Ip)
    beta = float(source_plan.beta)
    case_R0 = float(case.R0)
    case_B0 = float(case.B0)
    interpolation_code = _retained_source_interpolation_code(source_plan)
    coefficient0 = (
        source_workspace.pressure_spline_coeff
        if source_plan.scaled_pressure is not None
        else source_workspace.pprime_spline_coeff
    )
    differentiate_pressure = source_plan.scaled_pressure is not None

    def runner() -> tuple[float, float]:
        return _run_pj23_rho_joint_closure_impl(
            root_fields,
            FFn_psin,
            Pn_psin,
            query,
            query_next,
            query_r,
            query_r_next,
            sampled_pprime,
            sampled_driver,
            pprime_r,
            driver_r,
            f_profile,
            u,
            u_next,
            current,
            current_next,
            state,
            source_plan.scaled_pprime,
            source_plan.scaled_driver,
            source_workspace.source_coordinate_nodes,
            source_workspace.source_coordinate_weights,
            coefficient0,
            source_workspace.driver_spline_coeff,
            source_workspace.barycentric_weights,
            interpolation_code,
            differentiate_pressure,
            array_scratch,
            V_r,
            Kn,
            Ln_r,
            r,
            weights,
            accumulator,
            differentiator,
            F_fields,
            edge_f,
            use_jtotal_semantics,
            scaled_p0,
            scaled_Ip,
            beta,
            case_R0,
            case_B0,
            source_workspace.pressure_state,
            int(numba_source.RHO_FIXED_POINT_MAX_ITER),
            float(numba_source.RHO_FIXED_POINT_MAX_RESIDUAL),
        )

    return runner


def _build_source_stage_runner_shared(
    *,
    plan,
    case,
    source_workspace,
    profile_workspace,
    residual_workspace,
    fix_r: float,
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
    materialized_pprime_input = source_workspace.materialized_pprime_input
    materialized_driver_input = source_workspace.materialized_driver_input
    source_target_root_fields = source_workspace.target_root_fields
    case_R0 = float(case.R0)

    if source_execution.requires_optimized_psin_profile:
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_workspace.psin_query
            source_parameter_query = source_workspace.parameter_query
            psin_profile_fields = profile_workspace.fields_for("psin")
            pprime_input = source_plan.scaled_pprime
            driver_input = source_plan.scaled_driver
            parameterization_code = source_plan.parameterization_code
            grid_workspace = plan.grid_workspace
            n_axis_fix = int(np.searchsorted(grid_workspace.r, fix_r))

            def runner() -> tuple[float, float]:
                if psin_profile_fields.size == 0:
                    raise RuntimeError("psin_profile runtime fields are not initialized")
                # Optimized psin owns the root fields here.  Materialization also
                # remaps pprime/driver using the just-built psin coordinate.
                if source_plan.is_explicit_nodes:
                    numba_source._materialize_profile_owned_psin_fields_impl(
                        psin,
                        psin_r,
                        psin_rr,
                        source_psin_query,
                        psin_profile_fields,
                        grid_workspace.radial_fields,
                        grid_workspace.differentiator,
                        grid_workspace.accumulator,
                        grid_workspace.weights,
                        int(n_axis_fix),
                    )
                    _interpolate_retained_source(
                        source_plan=source_plan,
                        source_workspace=source_workspace,
                        query=source_psin_query,
                        out_pprime=materialized_pprime_input,
                        out_driver=materialized_driver_input,
                    )
                else:
                    numba_source._materialize_profile_owned_psin_source_impl(
                        psin,
                        psin_r,
                        psin_rr,
                        source_psin_query,
                        source_parameter_query,
                        materialized_pprime_input,
                        materialized_driver_input,
                        psin_profile_fields,
                        pprime_input,
                        driver_input,
                        source_workspace.pprime_spline_coeff,
                        source_workspace.driver_spline_coeff,
                        int(parameterization_code),
                        grid_workspace.radial_fields,
                        grid_workspace.differentiator,
                        grid_workspace.accumulator,
                        grid_workspace.weights,
                        int(n_axis_fix),
                        source_workspace.barycentric_weights,
                        bool(source_plan.uses_barycentric_interpolation),
                    )
                return source_eval_runner(
                    source_target_root_fields,
                    FFn_psin,
                    Pn_psin,
                    materialized_pprime_input,
                    materialized_driver_input,
                    case_R0,
                )

            return runner

        source_psin_query = source_workspace.psin_query
        psin_profile_u = profile_workspace.values_for("psin")
        psin_profile_fields = profile_workspace.fields_for("psin")

        def runner() -> tuple[float, float]:
            if psin_profile_fields.size == 0:
                raise RuntimeError("psin_profile runtime fields are not initialized")
            # Grid-node or already-materialized psin routes can copy optimized
            # psin fields directly, then remap only the source inputs.
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
            if source_plan.is_explicit_nodes:
                _interpolate_retained_source(
                    source_plan=source_plan,
                    source_workspace=source_workspace,
                    query=source_workspace.parameter_query,
                    out_pprime=source_workspace.materialized_pprime_input,
                    out_driver=source_workspace.materialized_driver_input,
                )
            else:
                numba_source._resolve_source_inputs_prepared(
                    source_workspace.materialized_pprime_input,
                    source_workspace.materialized_driver_input,
                    source_plan.scaled_pprime,
                    source_plan.scaled_driver,
                    source_plan.coordinate_code,
                    source_plan.source_sample_count,
                    source_workspace.barycentric_weights,
                    source_workspace.fixed_remap_matrix,
                    source_workspace.pprime_spline_coeff,
                    source_workspace.driver_spline_coeff,
                    source_workspace.parameter_query,
                    source_plan.uses_barycentric_interpolation,
                )
            return source_eval_runner(
                source_target_root_fields,
                FFn_psin,
                Pn_psin,
                materialized_pprime_input,
                materialized_driver_input,
                case_R0,
            )

        return runner

    def runner() -> tuple[float, float]:
        # Source-owned psin routes write root_fields themselves; source inputs
        # were already materialized by source_runtime refresh.
        return source_eval_runner(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_pprime_input,
            materialized_driver_input,
            case_R0,
        )

    return runner


def _build_pj2_psin_uniform_source_stage_runner(
    *,
    plan,
    case,
    grid_workspace,
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
    axis_weights = grid_workspace.axis_interpolation_weights
    edge_weights = grid_workspace.edge_interpolation_weights

    def runner() -> tuple[float, float]:
        if source_workspace.psin_query[0] < 0.0:
            # Seed from the optimized psin profile and normalize to [0, 1]; the
            # fixed-point loop then evolves this query in source-owned psin space.
            np.copyto(source_workspace.psin_query, psin_profile_u)
            if source_workspace.psin_query.ndim != 1 or source_workspace.psin_query.size < 2:
                raise ValueError(
                    f"Expected psin query to be 1D with at least two points, "
                    f"got {source_workspace.psin_query.shape}"
                )
            offset = float(np.dot(axis_weights, source_workspace.psin_query))
            scale = float(np.dot(edge_weights, source_workspace.psin_query) - offset)
            if abs(scale) < 1e-12:
                raise ValueError("psin query does not span a valid normalized flux interval")
            source_workspace.psin_query -= offset
            source_workspace.psin_query /= scale
        alpha1 = float("nan")
        alpha2 = float("nan")
        converged = False
        for _ in range(numba_source.PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER):
            # Each iteration remaps source samples using the previous psin query,
            # runs the source kernel, then tests whether the produced psin moved.
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
            if source_plan.is_explicit_nodes:
                _interpolate_retained_source(
                    source_plan=source_plan,
                    source_workspace=source_workspace,
                    query=source_workspace.parameter_query,
                    out_pprime=source_workspace.materialized_pprime_input,
                    out_driver=source_workspace.materialized_driver_input,
                )
            else:
                numba_source._resolve_source_inputs_prepared(
                    source_workspace.materialized_pprime_input,
                    source_workspace.materialized_driver_input,
                    source_plan.scaled_pprime,
                    source_plan.scaled_driver,
                    source_plan.coordinate_code,
                    source_plan.source_sample_count,
                    source_workspace.barycentric_weights,
                    source_workspace.fixed_remap_matrix,
                    source_workspace.pprime_spline_coeff,
                    source_workspace.driver_spline_coeff,
                    source_workspace.parameter_query,
                    source_plan.uses_barycentric_interpolation,
                )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                FFn_psin,
                Pn_psin,
                source_workspace.materialized_pprime_input,
                source_workspace.materialized_driver_input,
                case_R0,
            )
            if bool(
                numba_source._update_fixed_point_psin_query_impl(
                    source_workspace.psin_query,
                    target_root_fields[0],
                    float(numba_source.PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL),
                )
            ):
                converged = True
                break
        if not converged:
            raise ValueError(
                "PJ2/PJ3 sampled-psin source closure did not reach "
                f"{numba_source.PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL:.1e} "
                f"within {numba_source.PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER} iterations"
            )
        np.copyto(source_workspace.psin_query, target_root_fields[0])
        np.copyto(psin, target_root_fields[0])
        np.copyto(psin_r, target_root_fields[1])
        np.copyto(psin_rr, target_root_fields[2])
        return alpha1, alpha2

    return runner
