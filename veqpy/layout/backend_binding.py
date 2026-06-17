"""
Module: layout.backend_binding

Role:
- Dispatch operator layout binding to the selected backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from veqpy.engine.backend import normalize_backend
from veqpy.layout.jax_binding import build_jax_operator_layout
from veqpy.layout.numba_binding import build_operator_layout as build_numba_operator_layout

if TYPE_CHECKING:
    from veqpy.model.problem import Problem
    from veqpy.operator.build_plan import OperatorBuildPlan, ResidualBindingLayout
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace

from .runtime import OperatorLayout


def build_operator_layout(
    *,
    plan: OperatorBuildPlan,
    problem: Problem,
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
    backend: str = "numba",
    backend_options: object | None = None,
) -> OperatorLayout:
    """Build an executable operator layout for the requested backend."""

    backend_name = normalize_backend(backend)
    if backend_name == "numba":
        return build_numba_operator_layout(
            plan=plan,
            problem=problem,
            profile_workspace=profile_workspace,
            geometry_workspace=geometry_workspace,
            source_workspace=source_workspace,
            residual_workspace=residual_workspace,
            grid_workspace=grid_workspace,
            residual_binding_layout=residual_binding_layout,
            c_effective_order=c_effective_order,
            s_effective_order=s_effective_order,
            fix_rho=fix_rho,
            psin_profile_fields_available=psin_profile_fields_available,
        )
    return build_jax_operator_layout(
        plan=plan,
        problem=problem,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        grid_workspace=grid_workspace,
        residual_binding_layout=residual_binding_layout,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        fix_rho=fix_rho,
        psin_profile_fields_available=psin_profile_fields_available,
        backend_options=backend_options,
    )
