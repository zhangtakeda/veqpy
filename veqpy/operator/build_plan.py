"""
Module: operator.build_plan

Role:
- Build the immutable operator topology/configuration plan from Grid + OperatorCase.
- Keep packed topology, source route binding, and residual binding construction out of Operator.

Public API:
- OperatorBuildPlan
- build_operator_plan
- refresh_operator_plan_for_case
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import veqpy.engine.backend_abi as backend_abi
from veqpy.engine import validate_route
from veqpy.model.grid import Grid
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import (
    PROFILE_OFFSET_SPECS,
    PROFILE_STATIC_KWARGS,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_residual_block_metadata,
    build_residual_block_radial_powers,
    build_shape_profile_names,
    get_prefix_profile_names,
    packed_size,
)
from veqpy.operator.source_plan import SourcePlan, build_source_plan
from veqpy.workspace import GridWorkspace


@dataclass(frozen=True, slots=True)
class ResidualBindingLayout:
    """Read-only metadata bound to residual packing runners."""

    active_profile_names: tuple[str, ...]
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray


@dataclass(frozen=True, slots=True)
class OperatorBuildPlan:
    """Static topology and case-derived runtime binding plan for Operator."""

    grid_workspace: GridWorkspace
    prefix_profile_names: tuple[str, ...]
    shape_profile_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    profile_index: dict[str, int]
    c_profile_names: tuple[str, ...]
    s_profile_names: tuple[str, ...]
    profile_L: np.ndarray
    coeff_index: np.ndarray
    order_offsets: np.ndarray
    active_profile_mask: np.ndarray
    active_profile_ids: np.ndarray
    x_size: int
    source_route_spec: object
    source_plan: SourcePlan
    source_execution: backend_abi.SourceExecutionABI
    residual_binding_layout: ResidualBindingLayout
    profile_static_kwargs_by_name: dict[str, dict[str, int]]
    profile_offset_specs: dict[str, float | str]


def build_operator_plan(
    *,
    grid: Grid,
    case: OperatorCase,
    source_interpolation_kind: str,
) -> OperatorBuildPlan:
    """Build full Operator topology from an initial Grid + case."""

    grid_workspace = build_grid_workspace(grid)
    prefix_profile_names = get_prefix_profile_names()
    shape_profile_names = build_shape_profile_names(grid_workspace.M_max)
    profile_names = build_profile_names(grid_workspace.M_max)
    profile_index = build_profile_index(profile_names)
    fourier_profile_names = build_fourier_profile_names(grid_workspace.M_max)
    c_profile_names = tuple(name for name in fourier_profile_names if name.startswith("c"))
    s_profile_names = tuple(name for name in fourier_profile_names if name.startswith("s"))

    profile_L, coeff_index, order_offsets = build_profile_layout(
        case.profile_coeffs,
        profile_names=profile_names,
        prefix_profile_names=prefix_profile_names,
    )
    active_profile_mask, active_profile_ids = build_active_profile_metadata(
        profile_L,
        profile_names=profile_names,
    )
    x_size = packed_size(coeff_index)
    residual_binding_layout = build_residual_binding_layout(
        profile_names=profile_names,
        active_profile_ids=active_profile_ids,
        K_values=grid_workspace.K_values,
    )
    profile_static_kwargs_by_name, profile_offset_specs = build_profile_config(
        grid_workspace=grid_workspace,
        c_profile_names=c_profile_names,
        s_profile_names=s_profile_names,
    )
    source_route_spec = validate_route(case.route, case.coordinate, case.nodes)
    source_plan = build_source_plan(
        case=case,
        source_route_spec=source_route_spec,
        interpolation_kind=source_interpolation_kind,
    )
    source_execution = backend_abi.build_source_execution_abi(
        source_plan=source_plan,
        profile_index=profile_index,
        profile_L=profile_L,
        coeff_index=coeff_index,
        active_profile_ids=active_profile_ids,
    )

    return OperatorBuildPlan(
        grid_workspace=grid_workspace,
        prefix_profile_names=prefix_profile_names,
        shape_profile_names=shape_profile_names,
        profile_names=profile_names,
        profile_index=profile_index,
        c_profile_names=c_profile_names,
        s_profile_names=s_profile_names,
        profile_L=profile_L,
        coeff_index=coeff_index,
        order_offsets=order_offsets,
        active_profile_mask=active_profile_mask,
        active_profile_ids=active_profile_ids,
        x_size=x_size,
        source_route_spec=source_route_spec,
        source_plan=source_plan,
        source_execution=source_execution,
        residual_binding_layout=residual_binding_layout,
        profile_static_kwargs_by_name=profile_static_kwargs_by_name,
        profile_offset_specs=profile_offset_specs,
    )


def refresh_operator_plan_for_case(
    plan: OperatorBuildPlan,
    *,
    case: OperatorCase,
    source_interpolation_kind: str,
) -> OperatorBuildPlan:
    """Refresh case-dependent route/source ABI while preserving packed topology."""

    source_route_spec = validate_route(case.route, case.coordinate, case.nodes)
    source_plan = build_source_plan(
        case=case,
        source_route_spec=source_route_spec,
        interpolation_kind=source_interpolation_kind,
    )
    source_execution = backend_abi.build_source_execution_abi(
        source_plan=source_plan,
        profile_index=plan.profile_index,
        profile_L=plan.profile_L,
        coeff_index=plan.coeff_index,
        active_profile_ids=plan.active_profile_ids,
    )
    return replace(
        plan,
        source_route_spec=source_route_spec,
        source_plan=source_plan,
        source_execution=source_execution,
    )


def build_grid_workspace(grid: Grid) -> GridWorkspace:
    """Lower Grid into the static arrays consumed by runtime binding."""

    return GridWorkspace.from_grid(grid)


def build_residual_binding_layout(
    *,
    profile_names: tuple[str, ...],
    active_profile_ids: np.ndarray,
    K_values: np.ndarray,
) -> ResidualBindingLayout:
    """Build fixed residual pack metadata from active packed profiles."""

    active_profile_names = tuple(profile_names[int(p)] for p in active_profile_ids)
    active_residual_block_codes, active_residual_block_orders = build_residual_block_metadata(
        active_profile_names
    )
    active_residual_block_radial_powers = build_residual_block_radial_powers(
        active_profile_names,
        K_values=K_values,
    )
    return ResidualBindingLayout(
        active_profile_names=active_profile_names,
        active_residual_block_codes=active_residual_block_codes,
        active_residual_block_orders=active_residual_block_orders,
        active_residual_block_radial_powers=active_residual_block_radial_powers,
    )


def build_profile_config(
    *,
    grid_workspace: GridWorkspace,
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, int]], dict[str, float | str]]:
    """Build profile construction kwargs and offset specs from static topology."""

    profile_static_kwargs_by_name = {
        name: dict(kwargs) for name, kwargs in PROFILE_STATIC_KWARGS.items()
    }
    for name in c_profile_names + s_profile_names:
        order = int(name[1:])
        profile_static_kwargs_by_name[name] = (
            {} if order == 0 else {"power": int(grid_workspace.K_values[order])}
        )
    return profile_static_kwargs_by_name, dict(PROFILE_OFFSET_SPECS)
