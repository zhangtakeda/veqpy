"""
Module: layout.binding

Role:
- Own Python closure wiring for executable operator layouts.
- Bind hot-path callables against refreshed plan and workspace objects.

Public API:
- build_operator_layout

Notes:
- Workspace objects own memory; layout objects own executable stage callables.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine import numba_operator, numba_profile
from veqpy.layout.geometry_binding import build_geometry_stage_runner
from veqpy.layout.profile_binding import build_profile_stage_runner
from veqpy.layout.residual_binding import (
    build_collocation_runner_into,
    build_fused_residual_runner,
    build_fused_residual_runner_into,
    build_residual_full_stage_runner,
    build_residual_full_stage_runner_into,
)
from veqpy.layout.source_binding import build_bound_source_stage_runner

if TYPE_CHECKING:
    from veqpy.operator.build_plan import OperatorBuildPlan, ResidualBindingLayout
    from veqpy.operator.operator_case import OperatorCase
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace

from .runtime import OperatorLayout


def build_operator_layout(
    *,
    plan: OperatorBuildPlan,
    case: OperatorCase,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    grid_workspace: GridWorkspace,
    residual_binding_layout: ResidualBindingLayout,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
) -> OperatorLayout:
    """Bind a full executable ``OperatorLayout`` from refreshed runtime state."""

    alpha_state = source_workspace.alpha_state
    profile_stage_runner = _build_profile_stage_runner(
        plan=plan, profile_workspace=profile_workspace
    )
    profile_postprocess_runner = _build_profile_postprocess_runner(
        profile_workspace=profile_workspace,
    )
    geometry_stage_runner = _build_geometry_stage_runner(
        plan=plan,
        case=case,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
    )
    source_eval_runner = numba_operator.bind_source_eval_runner(
        source_plan=plan.source_plan,
        grid_workspace=grid_workspace,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        B0=case.B0,
        fix_rho=fix_rho,
    )
    raw_source_stage_runner = _build_bound_source_stage_runner(
        plan=plan,
        case=case,
        source_workspace=source_workspace,
        profile_workspace=profile_workspace,
        residual_workspace=residual_workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )
    source_stage_runner = _bind_alpha_tracking_source_runner(
        raw_source_stage_runner, alpha_state=alpha_state
    )
    residual_full_stage_runner_into = build_residual_full_stage_runner_into(
        plan=plan,
        case=case,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        residual_workspace=residual_workspace,
        alpha_state=alpha_state,
    )
    residual_full_stage_runner = build_residual_full_stage_runner(
        plan=plan,
        runner_into=residual_full_stage_runner_into,
    )
    fused_residual_runner_into = build_fused_residual_runner_into(
        plan=plan,
        case=case,
        grid_workspace=grid_workspace,
        residual_binding_layout=residual_binding_layout,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        alpha_state=alpha_state,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        fix_rho=fix_rho,
        psin_profile_fields_available=psin_profile_fields_available,
        profile_stage_runner=profile_stage_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_stage_runner=source_stage_runner,
        residual_full_stage_runner_into=residual_full_stage_runner_into,
    )
    fused_residual_runner = build_fused_residual_runner(
        plan=plan,
        runner_into=fused_residual_runner_into,
    )
    collocation_runner_into = build_collocation_runner_into(
        geometry_workspace=geometry_workspace,
        residual_workspace=residual_workspace,
        profile_stage_runner=profile_stage_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_stage_runner=source_stage_runner,
        alpha_state=alpha_state,
    )
    return OperatorLayout.from_callables(
        profile_stage_runner=profile_stage_runner,
        profile_postprocess_runner=profile_postprocess_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_eval_runner=source_eval_runner,
        source_stage_runner=source_stage_runner,
        residual_full_stage_runner_into=residual_full_stage_runner_into,
        residual_full_stage_runner=residual_full_stage_runner,
        fused_residual_runner_into=fused_residual_runner_into,
        fused_residual_runner=fused_residual_runner,
        collocation_runner_into=collocation_runner_into,
    )


def _build_profile_stage_runner(
    *,
    plan: OperatorBuildPlan,
    profile_workspace: ProfileWorkspace,
) -> Callable[[np.ndarray], None]:
    return build_profile_stage_runner(
        active_profile_ids=plan.active_profile_ids,
        profile_fields=profile_workspace.profile_fields,
        profile_rp_fields=profile_workspace.profile_rp_fields,
        profile_env_fields=profile_workspace.profile_env_fields,
        T=plan.grid_workspace.T,
        T_r=plan.grid_workspace.T_r,
        T_rr=plan.grid_workspace.T_rr,
        active_offsets=profile_workspace.active_offsets,
        active_scales=profile_workspace.active_scales,
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        update_profiles_packed_bulk=numba_profile.update_profiles_packed_bulk,
    )


def _build_profile_postprocess_runner(
    *,
    profile_workspace: ProfileWorkspace,
) -> Callable[[], None]:
    eps = 1.0e-10

    f_fields = profile_workspace.fields_for("F")

    def runner() -> None:
        numba_operator.convert_f_squared_fields_to_f(f_fields, eps=eps)

    return runner


def _build_geometry_stage_runner(
    *,
    plan: OperatorBuildPlan,
    case: OperatorCase,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    c_effective_order: int,
    s_effective_order: int,
) -> Callable[[], None]:
    return build_geometry_stage_runner(
        c_family_fields=profile_workspace.c_family_fields,
        s_family_fields=profile_workspace.s_family_fields,
        c_family_base_fields=profile_workspace.c_family_base_fields,
        s_family_base_fields=profile_workspace.s_family_base_fields,
        profile_fields=profile_workspace.profile_fields,
        c_family_source_profile_ids=profile_workspace.c_family_source_profile_ids,
        s_family_source_profile_ids=profile_workspace.s_family_source_profile_ids,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        h_fields=profile_workspace.fields_for("h"),
        v_fields=profile_workspace.fields_for("v"),
        k_fields=profile_workspace.fields_for("k"),
        a=case.a,
        R0=case.R0,
        Z0=case.Z0,
        surface_fields=geometry_workspace.surface_fields,
        radial_fields=geometry_workspace.radial_fields,
        rho=plan.grid_workspace.rho,
        theta=plan.grid_workspace.theta,
        cos_mtheta=plan.grid_workspace.cos_mtheta,
        sin_mtheta=plan.grid_workspace.sin_mtheta,
        m_cos_mtheta=plan.grid_workspace.m_cos_mtheta,
        m_sin_mtheta=plan.grid_workspace.m_sin_mtheta,
        m2_cos_mtheta=plan.grid_workspace.m2_cos_mtheta,
        m2_sin_mtheta=plan.grid_workspace.m2_sin_mtheta,
    )


def _build_bound_source_stage_runner(
    *,
    plan: OperatorBuildPlan,
    case: OperatorCase,
    source_workspace: SourceWorkspace,
    profile_workspace: ProfileWorkspace,
    residual_workspace: ResidualWorkspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable[[], tuple[float, float]]:
    return build_bound_source_stage_runner(
        plan=plan,
        case=case,
        source_workspace=source_workspace,
        profile_workspace=profile_workspace,
        residual_workspace=residual_workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )


def _bind_alpha_tracking_source_runner(
    runner: Callable[[], tuple[float, float]], *, alpha_state: np.ndarray
) -> Callable[[], tuple[float, float]]:
    """Return a source runner that writes source scale factors into workspace memory."""

    def tracked_runner() -> tuple[float, float]:
        alpha1, alpha2 = runner()
        alpha_state[0] = float(alpha1)
        alpha_state[1] = float(alpha2)
        return float(alpha1), float(alpha2)

    return tracked_runner
