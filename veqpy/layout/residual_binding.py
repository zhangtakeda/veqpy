"""
Module: layout.residual_binding

Role:
- Bind residual and collocation stage callables from refreshed runtime state.
- Keep residual closure wiring separate from the top-level operator layout factory.

Notes:
- Packed residual semantics remain owned by ``veqpy.operator.packed_layout`` and
  ``veqpy.operator.build_plan``.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine import numba_operator, numba_residual

if TYPE_CHECKING:
    from veqpy.model.problem import Problem
    from veqpy.operator.build_plan import OperatorBuildPlan, ResidualBindingLayout
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace


def build_residual_full_stage_runner_into(
    *,
    plan: OperatorBuildPlan,
    case: Problem,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    residual_workspace: ResidualWorkspace,
    alpha_state: np.ndarray,
) -> Callable[[np.ndarray], None]:
    """Bind staged residual assembly into a caller-provided output vector."""
    root_fields = residual_workspace.root_fields
    surface_fields = geometry_workspace.surface_fields
    residual_surface_fields = residual_workspace.surface_fields
    residual_pack_scratch = residual_workspace.pack_scratch
    residual_pack_scratch_rows = residual_workspace.pack_scratch_rows
    grid_radial_fields = plan.grid_workspace.radial_fields
    grid_poloidal_fields = plan.grid_workspace.poloidal_fields
    grid_k_max = int(plan.grid_workspace.K_max)
    grid_l_max = int(plan.grid_workspace.L_max)
    weights = plan.grid_workspace.weights
    a = case.a
    R0 = case.R0
    B0 = case.B0

    def runner(out: np.ndarray) -> None:
        # Residual compact fields depend only on current geometry/root fields and
        # alpha state; profile coefficient packing happens in the next kernel.
        numba_residual.update_residual_compact(
            residual_surface_fields,
            float(alpha_state[0]),
            float(alpha_state[1]),
            root_fields,
            surface_fields,
        )
        out.fill(0.0)
        # Packed residual assembly projects each active residual block onto its
        # corresponding basis/order metadata.  Inactive profile blocks have no
        # entries here and therefore contribute no equations.
        numba_residual.run_residual_blocks_packed_precomputed_auto(
            out,
            residual_pack_scratch,
            residual_pack_scratch_rows,
            plan.residual_binding_layout.active_residual_block_codes,
            plan.residual_binding_layout.active_residual_block_orders,
            plan.residual_binding_layout.active_residual_block_radial_powers,
            profile_workspace.active_coeff_index_rows,
            profile_workspace.active_lengths,
            residual_surface_fields,
            grid_radial_fields,
            grid_poloidal_fields,
            grid_k_max,
            grid_l_max,
            weights,
            a,
            R0,
            B0,
        )

    return runner


def build_collocation_runner_into(
    *,
    geometry_workspace: GeometryWorkspace,
    residual_workspace: ResidualWorkspace,
    profile_stage_runner: Callable[[np.ndarray], None],
    geometry_stage_runner: Callable[[], None],
    source_stage_runner: Callable[[], tuple[float, float]],
    alpha_state: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], None]:
    """Bind the collocation residual path into a caller-provided output vector."""
    geometry_surface_fields = geometry_workspace.surface_fields

    def runner(x_eval: np.ndarray, out: np.ndarray) -> None:
        # Collocation intentionally runs the same profile/geometry/source stages
        # as variational residuals, then writes pointwise weighted G instead of
        # Galerkin-projected blocks.
        profile_stage_runner(x_eval)
        geometry_stage_runner()
        alpha1, alpha2 = source_stage_runner()
        alpha_state[0] = float(alpha1)
        alpha_state[1] = float(alpha2)
        numba_residual.update_residual_compact(
            residual_workspace.surface_fields,
            float(alpha_state[0]),
            float(alpha_state[1]),
            residual_workspace.root_fields,
            geometry_surface_fields,
        )
        numba_residual.write_weighted_scaled_g_collocation_field_into(
            out,
            residual_workspace.surface_fields[0],
            geometry_surface_fields,
            residual_workspace.collocation_sqrt_weights,
            0,
        )

    return runner


def build_fused_residual_runner_into(
    *,
    plan: OperatorBuildPlan,
    case: Problem,
    grid_workspace: GridWorkspace,
    residual_binding_layout: ResidualBindingLayout,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    alpha_state: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
    profile_stage_runner: Callable[[np.ndarray], None],
    geometry_stage_runner: Callable[[], None],
    source_stage_runner: Callable[[], tuple[float, float]],
    residual_full_stage_runner_into: Callable[[np.ndarray], None],
) -> Callable[[np.ndarray, np.ndarray], None]:
    """Bind the fused packed-state residual runner into a caller output vector."""
    if plan.source_execution.requires_optimized_psin_profile and not psin_profile_fields_available:
        # Fused Numba code cannot pull optimized psin fields that do not exist
        # in ProfileWorkspace yet.  The sequential runner keeps behavior correct
        # for layout-valid but not-fusible source/profile combinations.
        def sequential_runner(x_eval: np.ndarray, out: np.ndarray) -> None:
            profile_stage_runner(x_eval)
            geometry_stage_runner()
            alpha1, alpha2 = source_stage_runner()
            alpha_state[0] = float(alpha1)
            alpha_state[1] = float(alpha2)
            residual_full_stage_runner_into(out)

        return sequential_runner
    return numba_operator.bind_fused_residual_runner_into(
        source_plan=plan.source_plan,
        source_execution=plan.source_execution,
        grid_workspace=grid_workspace,
        residual_binding_layout=residual_binding_layout,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        alpha_state=alpha_state,
        c_active_order=int(c_effective_order),
        s_active_order=int(s_effective_order),
        a=float(case.a),
        R0=float(case.R0),
        Z0=float(case.Z0),
        B0=float(case.B0),
        fix_rho=fix_rho,
    )
