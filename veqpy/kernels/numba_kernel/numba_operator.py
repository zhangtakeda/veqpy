"""
Module: engine.numba_operator

Role:
- Provide the fused x -> residual hot runner.
- Chain Stage A/B/C/D for common routes into a single engine binding entrypoint.

Public API:
- bind_fused_residual_runner
- bind_fused_residual_runner_into

Notes:
- Only common routes are covered here.
- PJ2-psin-uniform fixed-point psin is handled locally inside that route.
- Fused residual binding has three ownership modes:
  single-pass source-owned psin, profile-owned psin, and PJ2 fixed-point psin.
  Each mode refreshes profile/geometry first, then writes root fields, alpha_state,
  residual surface fields, and finally the packed residual.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

import veqpy.kernels.numba_kernel.backend_abi as backend_abi
from veqpy.kernels.numba_kernel.geometry_stage import update_geometry_hot_auto
from veqpy.kernels.numba_kernel.numba_residual import (
    run_residual_blocks_packed_precomputed_auto,
    update_residual_compact,
)
from veqpy.kernels.numba_kernel.numba_source import (
    PJ2_PSIN_UNIFORM_BARYCENTRIC_ORDER_CAP,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
    _local_barycentric_interpolate_pair,
    _materialize_profile_owned_psin_source_impl,
    _uniform_spline_interpolate_pair,
    _update_fixed_point_psin_query_and_local_barycentric_inputs_impl,
    _update_fixed_point_psin_query_and_spline_uniform_inputs_impl,
    _update_fourier_family_fields_impl,
    _update_pj2_from_psin_uniform_inputs_with_scratch,
    uniform_barycentric_weights,
)
from veqpy.kernels.numba_kernel.profile_stage import update_profiles_packed_bulk
from veqpy.kernels.numba_kernel.workspace.grid_workspace import GridWorkspace
from veqpy.numerics import build_uniform_source_interpolation_coefficients

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.workspace import (
        GeometryWorkspace,
        ProfileWorkspace,
        ResidualWorkspace,
        SourceWorkspace,
    )


def bind_source_eval_runner(
    *,
    source_plan: Any,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    B0: float,
    fix_rho: float,
) -> Callable:
    """Bind a Python callable around the fused-backend source evaluator."""
    return _bind_source_eval_runner_for_fused_backend(
        source_eval_binding=backend_abi.build_fused_source_eval_abi(
            source_plan=source_plan,
            grid_workspace=grid_workspace,
            geometry_workspace=geometry_workspace,
            source_workspace=source_workspace,
            B0=B0,
            fix_rho=fix_rho,
        ),
        f_profile_fields=profile_workspace.fields_for("F"),
    )


def _normalize_psin_query(out: np.ndarray, source: np.ndarray) -> None:
    np.copyto(out, source)
    offset = float(out[0])
    scale = float(out[-1] - offset)
    if abs(scale) < 1.0e-12:
        raise ValueError("psin query does not span a valid normalized flux interval")
    out -= offset
    out /= scale
    out[0] = 0.0
    out[-1] = 1.0


def _refresh_hot_runtime(
    x: np.ndarray,
    *,
    hot_runtime_binding: backend_abi.FusedHotRuntimeABI,
) -> None:
    # Stage A/B are fused here: packed coefficients refresh profile fields,
    # profile-derived Fourier families are rebuilt, then geometry is updated
    # before any source or residual kernel reads the workspaces.
    update_profiles_packed_bulk(
        hot_runtime_binding.profile_fields,
        hot_runtime_binding.profile_rp_fields,
        hot_runtime_binding.profile_env_fields,
        hot_runtime_binding.active_profile_ids,
        hot_runtime_binding.grid_radial_fields,
        hot_runtime_binding.grid_k_max,
        hot_runtime_binding.grid_l_max,
        hot_runtime_binding.active_offsets,
        hot_runtime_binding.active_scales,
        hot_runtime_binding.active_amplitude_powers,
        x,
        hot_runtime_binding.active_coeff_index_rows,
        hot_runtime_binding.active_lengths,
    )
    _update_fourier_family_fields_impl(
        hot_runtime_binding.c_family_fields,
        hot_runtime_binding.s_family_fields,
        hot_runtime_binding.c_family_base_fields,
        hot_runtime_binding.s_family_base_fields,
        hot_runtime_binding.profile_fields,
        hot_runtime_binding.c_family_source_profile_ids,
        hot_runtime_binding.s_family_source_profile_ids,
        hot_runtime_binding.c_active_order,
        hot_runtime_binding.s_active_order,
    )
    update_geometry_hot_auto(
        hot_runtime_binding.geometry_surface_fields,
        hot_runtime_binding.geometry_radial_fields,
        hot_runtime_binding.a,
        hot_runtime_binding.R0,
        hot_runtime_binding.Z0,
        hot_runtime_binding.grid_radial_fields,
        hot_runtime_binding.grid_poloidal_fields,
        hot_runtime_binding.h_fields,
        hot_runtime_binding.v_fields,
        hot_runtime_binding.k_fields,
        hot_runtime_binding.c_family_fields,
        hot_runtime_binding.s_family_fields,
        hot_runtime_binding.c_active_order,
        hot_runtime_binding.s_active_order,
    )


def _pack_residual_output_into(
    out: np.ndarray,
    *,
    residual_pack_binding: backend_abi.FusedResidualPackABI,
) -> None:
    # Packing is separate from residual-surface refresh so fused, staged, and
    # collocation paths can share the same compact G/G*grad(psin) workspace.
    out.fill(0.0)
    run_residual_blocks_packed_precomputed_auto(
        out,
        residual_pack_binding.residual_pack_scratch,
        residual_pack_binding.residual_pack_scratch_rows,
        residual_pack_binding.active_residual_block_codes,
        residual_pack_binding.active_residual_block_orders,
        residual_pack_binding.active_residual_block_radial_powers,
        residual_pack_binding.active_coeff_index_rows,
        residual_pack_binding.active_lengths,
        residual_pack_binding.residual_surface_fields,
        residual_pack_binding.grid_radial_fields,
        residual_pack_binding.grid_poloidal_fields,
        residual_pack_binding.grid_k_max,
        residual_pack_binding.grid_l_max,
        residual_pack_binding.weights,
        residual_pack_binding.a,
        residual_pack_binding.R0,
        residual_pack_binding.B0,
    )


@njit(cache=True, nogil=True)
def _run_pj2_psin_uniform_spline_with_scratch_impl(
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    heat_spline_coeff: np.ndarray,
    current_spline_coeff: np.ndarray,
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
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    # Materialize the first source sample on the previous psin query, then let
    # each fixed-point pass update psin and remap heat/current onto the new query.
    _uniform_spline_interpolate_pair(
        materialized_heat_input,
        materialized_current_input,
        heat_spline_coeff,
        current_spline_coeff,
        source_psin_query,
    )
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER):
        alpha1, alpha2 = _update_pj2_from_psin_uniform_inputs_with_scratch(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
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
            Ip,
            beta,
            array_scratch,
            matrix_scratch,
        )
        if _update_fixed_point_psin_query_and_spline_uniform_inputs_impl(
            source_psin_query,
            psin,
            PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
            materialized_heat_input,
            materialized_current_input,
            heat_input,
            current_input,
            heat_spline_coeff,
            current_spline_coeff,
        ):
            break
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _run_pj2_psin_uniform_barycentric_with_scratch_impl(
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    barycentric_weights: np.ndarray,
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
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    # Same fixed-point contract as the spline path, but interpolation uses a
    # small local barycentric stencil to avoid building dense remap matrices.
    _local_barycentric_interpolate_pair(
        materialized_heat_input,
        materialized_current_input,
        heat_input,
        current_input,
        source_psin_query,
        barycentric_weights,
    )
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER):
        alpha1, alpha2 = _update_pj2_from_psin_uniform_inputs_with_scratch(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
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
            Ip,
            beta,
            array_scratch,
            matrix_scratch,
        )
        if _update_fixed_point_psin_query_and_local_barycentric_inputs_impl(
            source_psin_query,
            psin,
            PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
            materialized_heat_input,
            materialized_current_input,
            heat_input,
            current_input,
            barycentric_weights,
        ):
            break
    return alpha1, alpha2


def bind_fused_residual_runner(
    *,
    source_plan: Any,
    source_execution: backend_abi.SourceExecutionABI,
    grid_workspace: GridWorkspace,
    residual_binding_layout: Any,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    fix_rho: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Bind fused residual execution that returns a copied output vector."""
    runner_into = bind_fused_residual_runner_into(
        source_plan=source_plan,
        source_execution=source_execution,
        grid_workspace=grid_workspace,
        residual_binding_layout=residual_binding_layout,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        alpha_state=alpha_state,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
        B0=B0,
        fix_rho=fix_rho,
    )
    packed_residual = residual_workspace.packed_residual

    def runner(x: np.ndarray) -> np.ndarray:
        runner_into(x, packed_residual)
        return packed_residual.copy()

    return runner


def bind_fused_residual_runner_into(
    *,
    source_plan: Any,
    source_execution: backend_abi.SourceExecutionABI,
    grid_workspace: GridWorkspace,
    residual_binding_layout: Any,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    fix_rho: float,
) -> Callable[[np.ndarray, np.ndarray], None]:
    """Bind fused residual execution into a caller-provided output vector."""
    route_key = tuple(source_execution.route_key)
    if route_key != source_plan.route_key:
        # SourcePlan owns user-facing route semantics; SourceExecutionABI owns
        # runtime workspace requirements.  A mismatch here would bind a valid
        # kernel against the wrong ownership contract.
        raise ValueError(
            f"Source execution ABI route mismatch: plan={source_plan.route_key!r}, "
            f"binding={route_key!r}"
        )

    hot_runtime_binding = backend_abi.build_fused_hot_runtime_abi(
        grid_workspace=grid_workspace,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )
    residual_pack_binding = backend_abi.build_fused_residual_pack_abi(
        grid_workspace=grid_workspace,
        residual_binding_layout=residual_binding_layout,
        profile_workspace=profile_workspace,
        residual_workspace=residual_workspace,
        a=a,
        R0=R0,
        B0=B0,
    )

    if route_key == ("PJ2", "psin", "uniform"):
        # PJ2/psin/uniform is the only fused route whose source query is itself
        # part of the solution, so it needs a dedicated fixed-point runner.
        return _bind_pj2_psin_uniform_residual_runner_core(
            source_plan=source_plan,
            grid_workspace=grid_workspace,
            profile_workspace=profile_workspace,
            geometry_workspace=geometry_workspace,
            source_workspace=source_workspace,
            residual_workspace=residual_workspace,
            hot_runtime_binding=hot_runtime_binding,
            residual_pack_binding=residual_pack_binding,
            alpha_state=alpha_state,
            R0=R0,
            B0=B0,
            fix_rho=fix_rho,
        )

    source_eval_runner = bind_source_eval_runner(
        source_plan=source_plan,
        grid_workspace=grid_workspace,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        B0=B0,
        fix_rho=fix_rho,
    )
    if source_execution.requires_optimized_psin_profile:
        # These routes read psin from the packed profile, materialize source
        # inputs against that profile, then evaluate source fields into a
        # separate target root buffer so the optimized psin rows remain intact.
        return _bind_profile_owned_psin_residual_runner_core(
            source_plan=source_plan,
            source_execution=source_execution,
            grid_workspace=grid_workspace,
            profile_workspace=profile_workspace,
            geometry_workspace=geometry_workspace,
            source_workspace=source_workspace,
            residual_workspace=residual_workspace,
            source_eval_runner=source_eval_runner,
            hot_runtime_binding=hot_runtime_binding,
            residual_pack_binding=residual_pack_binding,
            alpha_state=alpha_state,
            R0=R0,
            fix_rho=fix_rho,
        )

    return _bind_single_pass_residual_runner_core(
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        source_eval_runner=source_eval_runner,
        hot_runtime_binding=hot_runtime_binding,
        residual_pack_binding=residual_pack_binding,
        alpha_state=alpha_state,
        R0=R0,
    )


def _bind_single_pass_residual_runner_core(
    *,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    source_eval_runner: Callable,
    hot_runtime_binding: backend_abi.FusedHotRuntimeABI,
    residual_pack_binding: backend_abi.FusedResidualPackABI,
    alpha_state: np.ndarray,
    R0: float,
) -> Callable[[np.ndarray, np.ndarray], None]:
    surface_fields = geometry_workspace.surface_fields
    residual_surface_fields = residual_workspace.surface_fields
    root_fields = residual_workspace.root_fields
    materialized_heat_input = source_workspace.materialized_heat_input
    materialized_current_input = source_workspace.materialized_current_input
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray, out: np.ndarray) -> None:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        # Single-pass routes let the source kernel own psin/root_fields directly.
        # Source inputs were materialized by source_runtime before this runner.
        alpha1, alpha2 = source_eval_runner(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            R0,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        # alpha_state mirrors the returned scales for staged callers and
        # snapshots; residual_compact consumes the local values immediately.
        update_residual_compact(
            residual_surface_fields,
            alpha1,
            alpha2,
            root_fields,
            surface_fields,
        )
        _pack_residual_output_into(out, residual_pack_binding=residual_pack_binding)

    return runner


def _bind_profile_owned_psin_residual_runner_core(
    *,
    source_plan: Any,
    source_execution: backend_abi.SourceExecutionABI,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    source_eval_runner: Callable,
    hot_runtime_binding: backend_abi.FusedHotRuntimeABI,
    residual_pack_binding: backend_abi.FusedResidualPackABI,
    alpha_state: np.ndarray,
    R0: float,
    fix_rho: float,
) -> Callable[[np.ndarray, np.ndarray], None]:
    surface_fields = geometry_workspace.surface_fields
    residual_surface_fields = residual_workspace.surface_fields
    n_axis_fix = int(np.searchsorted(grid_workspace.rho, fix_rho))
    root_fields = residual_workspace.root_fields
    profile_owned_psin_binding = backend_abi.build_profile_owned_psin_source_abi(
        source_plan=source_plan,
        source_execution=source_execution,
        grid_workspace=grid_workspace,
        profile_workspace=profile_workspace,
        source_workspace=source_workspace,
    )
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray, out: np.ndarray) -> None:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        # Profile-owned psin routes copy psin/psin_r/psin_rr from the active
        # optimized profile, then evaluate heat/current samples at that psin
        # coordinate.  The source kernel writes only source derivatives/scales.
        _materialize_profile_owned_psin_source_impl(
            psin,
            psin_r,
            psin_rr,
            profile_owned_psin_binding.source_psin_query,
            profile_owned_psin_binding.source_parameter_query,
            profile_owned_psin_binding.materialized_heat_input,
            profile_owned_psin_binding.materialized_current_input,
            profile_owned_psin_binding.psin_profile_fields,
            profile_owned_psin_binding.scaled_heat,
            profile_owned_psin_binding.scaled_current,
            profile_owned_psin_binding.heat_spline_coeff,
            profile_owned_psin_binding.current_spline_coeff,
            profile_owned_psin_binding.parameterization_code,
            profile_owned_psin_binding.grid_radial_fields,
            profile_owned_psin_binding.differentiator,
            profile_owned_psin_binding.accumulator,
            n_axis_fix,
            profile_owned_psin_binding.barycentric_weights,
            profile_owned_psin_binding.use_barycentric,
        )
        alpha1, alpha2 = source_eval_runner(
            profile_owned_psin_binding.source_target_root_fields,
            FFn_psin,
            Pn_psin,
            profile_owned_psin_binding.materialized_heat_input,
            profile_owned_psin_binding.materialized_current_input,
            R0,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        # residual_compact reads root_fields, not source_target_root_fields.
        # That preserves the optimized psin rows while still using route-produced
        # FFn/Pn and alpha scales.
        update_residual_compact(
            residual_surface_fields,
            alpha1,
            alpha2,
            root_fields,
            surface_fields,
        )
        _pack_residual_output_into(out, residual_pack_binding=residual_pack_binding)

    return runner


def _bind_pj2_psin_uniform_residual_runner_core(
    *,
    source_plan: Any,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    hot_runtime_binding: backend_abi.FusedHotRuntimeABI,
    residual_pack_binding: backend_abi.FusedResidualPackABI,
    alpha_state: np.ndarray,
    R0: float,
    B0: float,
    fix_rho: float,
) -> Callable[[np.ndarray, np.ndarray], None]:
    surface_fields = geometry_workspace.surface_fields
    radial_fields = geometry_workspace.radial_fields
    residual_surface_fields = residual_workspace.surface_fields
    rho = grid_workspace.rho
    grid_radial_fields = grid_workspace.radial_fields
    weights = grid_workspace.weights
    differentiator = grid_workspace.differentiator
    accumulator = grid_workspace.accumulator
    n_axis_fix = int(np.searchsorted(rho, fix_rho))
    root_fields = residual_workspace.root_fields

    source_psin_query = source_workspace.psin_query
    materialized_heat_input = source_workspace.materialized_heat_input
    materialized_current_input = source_workspace.materialized_current_input
    array_scratch = source_workspace.array_scratch
    matrix_scratch = source_workspace.matrix_scratch
    f_profile_fields = profile_workspace.fields_for("F")
    psin_profile_u = profile_workspace.values_for("psin")
    heat_input = source_plan.scaled_heat
    current_input = source_plan.scaled_current
    heat_spline_coeff = build_uniform_source_interpolation_coefficients(
        heat_input,
        kind=source_plan.interpolation_kind,
    )
    current_spline_coeff = build_uniform_source_interpolation_coefficients(
        current_input,
        kind=source_plan.interpolation_kind,
    )
    coordinate_code = int(source_plan.coordinate_code)
    Ip = float(source_plan.scaled_Ip)
    beta = float(source_plan.beta)
    has_Ip = bool(np.isfinite(Ip))
    use_local_barycentric = bool(source_plan.uses_barycentric_interpolation)
    # PJ2 fixed-point can use allocation-free local barycentric updates in the
    # Ip-constrained path.  Other cases use spline coefficients already prepared
    # for uniform source samples.
    barycentric_weights = uniform_barycentric_weights(
        min(
            PJ2_PSIN_UNIFORM_BARYCENTRIC_ORDER_CAP,
            int(source_plan.source_sample_count),
        )
    )

    psin = root_fields[0]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray, out: np.ndarray) -> None:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        if source_psin_query[0] < 0.0:
            # ``invalidate_source_state`` marks the query with -1.  The first
            # evaluation after an x0 reset re-seeds it from the optimized psin
            # profile before the fixed-point loop starts.
            _normalize_psin_query(source_psin_query, psin_profile_u)
        if has_Ip and use_local_barycentric:
            # The barycentric helper updates heat/current inside the fixed-point
            # loop without allocating a fresh remap matrix for each psin query.
            alpha1, alpha2 = _run_pj2_psin_uniform_barycentric_with_scratch_impl(
                source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                heat_input,
                current_input,
                barycentric_weights,
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
                Ip,
                beta,
                array_scratch,
                matrix_scratch,
            )
        else:
            # Spline coefficients are reused across the loop; only query values
            # and materialized source arrays change between iterations.
            alpha1, alpha2 = _run_pj2_psin_uniform_spline_with_scratch_impl(
                source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                heat_input,
                current_input,
                heat_spline_coeff,
                current_spline_coeff,
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
                Ip,
                beta,
                array_scratch,
                matrix_scratch,
            )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        # The fixed-point source kernel publishes its last root_fields even when
        # the iteration stops by cap, so residual packing remains self-consistent.
        update_residual_compact(
            residual_surface_fields,
            alpha1,
            alpha2,
            root_fields,
            surface_fields,
        )
        _pack_residual_output_into(out, residual_pack_binding=residual_pack_binding)

    return runner


def _bind_source_eval_runner_for_fused_backend(
    *,
    source_eval_binding: backend_abi.FusedSourceEvalABI,
    f_profile_fields: np.ndarray,
) -> Callable:
    def runner(
        out_root_fields: np.ndarray,
        out_FFn_psin: np.ndarray,
        out_Pn_psin: np.ndarray,
        heat_input: np.ndarray,
        current_input: np.ndarray,
        R0: float,
    ) -> tuple[float, float]:
        return source_eval_binding.source_kernel(
            out_root_fields,
            out_FFn_psin,
            out_Pn_psin,
            heat_input,
            current_input,
            source_eval_binding.coordinate_code,
            R0,
            source_eval_binding.B0,
            source_eval_binding.weights,
            source_eval_binding.differentiator,
            source_eval_binding.accumulator,
            source_eval_binding.grid_radial_fields,
            source_eval_binding.n_axis_fix,
            source_eval_binding.radial_fields,
            source_eval_binding.surface_fields,
            f_profile_fields,
            source_eval_binding.scaled_Ip,
            source_eval_binding.beta,
            source_eval_binding.array_scratch,
            source_eval_binding.matrix_scratch,
        )

    return runner
