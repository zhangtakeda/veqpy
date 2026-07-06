"""
Module: layout.binding

Role:
- Own Python closure wiring for executable Kernel layouts.
- Bind hot-path callables against refreshed plan and workspace objects.

Public API:
- build_kernel_layout

Notes:
- Workspace objects own memory; layout objects own executable stage callables.
- Numerical kernels remain private to this backend package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import veqpy.kernels.numba_kernel.profile_stage as numba_profile

from . import numba_operator
from .geometry_binding import build_geometry_stage_runner
from .profile_binding import build_profile_stage_runner
from .residual_binding import (
    build_collocation_runner_into,
    build_fused_residual_runner_into,
    build_residual_full_stage_runner_into,
)
from .source_binding import build_bound_source_stage_runner

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.workspace import (
        GeometryWorkspace,
        ProfileWorkspace,
        ResidualWorkspace,
        SourceWorkspace,
    )
    from veqpy.kernels.numba_kernel.workspace.grid_workspace import GridWorkspace

from .layout import KernelLayout


def build_kernel_layout(
    *,
    plan: Any,
    case: object,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    grid_workspace: GridWorkspace,
    residual_binding_layout: Any,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
) -> KernelLayout:
    """Bind a full executable ``KernelLayout`` from refreshed runtime state."""

    alpha_state = source_workspace.alpha_state
    # Build stage closures in dependency order.  Later fused/collocation runners
    # reuse the same closures so staged and fused paths share workspace semantics.
    profile_stage_runner = build_profile_stage_runner(
        active_profile_ids=plan.active_profile_ids,
        profile_fields=profile_workspace.profile_fields,
        profile_rp_fields=profile_workspace.profile_rp_fields,
        profile_env_fields=profile_workspace.profile_env_fields,
        grid_radial_fields=plan.grid_workspace.radial_fields,
        grid_k_max=int(plan.grid_workspace.K_max),
        grid_l_max=int(plan.grid_workspace.L_max),
        active_offsets=profile_workspace.active_offsets,
        active_scales=profile_workspace.active_scales,
        active_amplitude_powers=profile_workspace.active_amplitude_powers,
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        update_profiles_packed_bulk=numba_profile.update_profiles_packed_bulk,
    )

    geometry_stage_runner = build_geometry_stage_runner(
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
        grid_radial_fields=plan.grid_workspace.radial_fields,
        grid_poloidal_fields=plan.grid_workspace.poloidal_fields,
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
    raw_source_stage_runner = build_bound_source_stage_runner(
        plan=plan,
        case=case,
        source_workspace=source_workspace,
        profile_workspace=profile_workspace,
        residual_workspace=residual_workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )

    def source_stage_runner() -> tuple[float, float]:
        alpha1, alpha2 = raw_source_stage_runner()
        alpha_state[0] = float(alpha1)
        alpha_state[1] = float(alpha2)
        return float(alpha1), float(alpha2)

    # Residual closures read alpha_state, so alpha tracking must wrap the source
    # runner before residual and collocation runners are bound.
    residual_full_stage_runner_into = build_residual_full_stage_runner_into(
        plan=plan,
        case=case,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        residual_workspace=residual_workspace,
        alpha_state=alpha_state,
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
    # Collocation shares the Stage A/B/C refresh chain but writes a pointwise
    # objective instead of the packed Galerkin residual.
    collocation_runner_into = build_collocation_runner_into(
        geometry_workspace=geometry_workspace,
        residual_workspace=residual_workspace,
        profile_stage_runner=profile_stage_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_stage_runner=source_stage_runner,
        alpha_state=alpha_state,
    )
    return KernelLayout.from_callables(
        profile_stage_runner=profile_stage_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_stage_runner=source_stage_runner,
        residual_full_stage_runner_into=residual_full_stage_runner_into,
        fused_residual_runner_into=fused_residual_runner_into,
        collocation_runner_into=collocation_runner_into,
    )
