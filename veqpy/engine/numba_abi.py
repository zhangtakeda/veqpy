"""
Module: engine.numba_abi

Role:
- Define explicit ABI binding contracts used by the numba fused backend.
- Move bind-time data selection out of numba implementation into the engine ABI module.
- Keep Python-side bundles coarse: sampled data travels as field slabs, while
  operators, metadata, scratch buffers, and state keep distinct names.

Public API:
- NumbaSourceBindingPlan
- FusedHotRuntimeABI
- FusedResidualPackABI
- FusedSourceEvalABI
- build_source_execution_abi
- build_fused_hot_runtime_abi
- build_fused_residual_pack_abi
- build_fused_source_eval_abi
- build_profile_owned_psin_source_abi
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine import numba_source
from veqpy.operator.source_execution import (
    SUPPORTED_SOURCE_ROUTE_KEYS,
    SourceExecutionPlan,
    build_source_execution_plan,
)
from veqpy.operator.source_plan import SOURCE_PARAMETERIZATION_CODES

if TYPE_CHECKING:
    from veqpy.operator.build_plan import ResidualBindingLayout
    from veqpy.operator.source_plan import SourcePlan
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace


RouteKey = tuple[str, str, str]
SUPPORTED_FUSED_SOURCE_ROUTE_KEYS: frozenset[RouteKey] = SUPPORTED_SOURCE_ROUTE_KEYS
SourceExecutionABI = SourceExecutionPlan


@dataclass(frozen=True, slots=True)
class NumbaSourceBindingPlan:
    """Concrete Numba callable and integer-code adapter for a source plan."""

    route_key: RouteKey
    kernel: Callable
    coordinate_code: int
    parameterization_code: int

    @classmethod
    def from_source_plan(cls, source_plan: SourcePlan) -> NumbaSourceBindingPlan:
        route_spec = numba_source.validate_route(
            source_plan.route,
            source_plan.coordinate,
            source_plan.nodes,
        )
        return cls(
            route_key=source_plan.route_key,
            kernel=route_spec.implementation,
            coordinate_code=int(route_spec.coordinate_code),
            parameterization_code=int(SOURCE_PARAMETERIZATION_CODES[source_plan.parameterization]),
        )


def build_source_execution_abi(
    *,
    source_plan: SourcePlan,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_profile_ids: np.ndarray,
) -> SourceExecutionABI:
    """Build source-route ABI metadata from a source plan and packed profile layout."""
    return build_source_execution_plan(
        source_plan=source_plan,
        profile_index=profile_index,
        profile_L=profile_L,
        coeff_index=coeff_index,
        active_profile_ids=active_profile_ids,
    )


@dataclass(frozen=True, slots=True)
class FusedHotRuntimeABI:
    """Array bundle required by fused profile/geometry hot-path kernels.

    All arrays are borrowed views from workspaces.  The ABI does not own memory;
    it freezes which rows/fields a compiled runner will read and overwrite.
    """

    profile_fields: np.ndarray
    profile_rp_fields: np.ndarray
    profile_env_fields: np.ndarray
    active_profile_ids: np.ndarray
    grid_radial_fields: np.ndarray
    grid_k_max: int
    grid_l_max: int
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_amplitude_powers: np.ndarray
    active_coeff_index_rows: np.ndarray
    active_lengths: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    c_family_source_profile_ids: np.ndarray
    s_family_source_profile_ids: np.ndarray
    geometry_surface_fields: np.ndarray
    geometry_radial_fields: np.ndarray
    grid_poloidal_fields: np.ndarray
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray
    c_active_order: int
    s_active_order: int
    a: float
    R0: float
    Z0: float


@dataclass(frozen=True, slots=True)
class FusedResidualPackABI:
    """Array bundle required to pack fused residual blocks.

    The residual packer receives integer metadata from packed_layout and scratch
    from ResidualWorkspace.  It must not inspect profile names at runtime.
    """

    residual_pack_scratch: np.ndarray
    residual_pack_scratch_rows: np.ndarray
    residual_surface_fields: np.ndarray
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray
    active_coeff_index_rows: np.ndarray
    active_lengths: np.ndarray
    grid_radial_fields: np.ndarray
    grid_poloidal_fields: np.ndarray
    grid_k_max: int
    grid_l_max: int
    weights: np.ndarray
    a: float
    R0: float
    B0: float


@dataclass(frozen=True, slots=True)
class FusedSourceEvalABI:
    """Array and kernel bundle required by fused source evaluation.

    Source kernels are flat Numba callables.  This object supplies the selected
    slab kernel plus the geometry/source arrays needed to call it without
    reaching back into Python objects. Route-specific optimized profile fields
    are bound by the caller so this generic ABI does not imply profile ownership.
    """

    source_kernel: Callable
    coordinate_code: int
    weights: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    grid_radial_fields: np.ndarray
    n_axis_fix: int
    radial_fields: np.ndarray
    surface_fields: np.ndarray
    scaled_Ip: float
    beta: float
    array_scratch: np.ndarray
    matrix_scratch: np.ndarray
    B0: float


@dataclass(frozen=True, slots=True)
class _ProfileOwnedPsinSourceABI:
    """Python-side source materialization bundle for optimized-psin routes."""

    source_target_root_fields: np.ndarray
    grid_radial_fields: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    source_psin_query: np.ndarray
    source_parameter_query: np.ndarray
    heat_spline_coeff: np.ndarray
    current_spline_coeff: np.ndarray
    barycentric_weights: np.ndarray
    use_barycentric: bool
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    psin_profile_fields: np.ndarray
    parameterization_code: int
    scaled_heat: np.ndarray
    scaled_current: np.ndarray


def build_fused_hot_runtime_abi(
    *,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
) -> FusedHotRuntimeABI:
    """Collect profile, geometry, and grid arrays for fused hot-path execution."""
    return FusedHotRuntimeABI(
        profile_fields=profile_workspace.profile_fields,
        profile_rp_fields=profile_workspace.profile_rp_fields,
        profile_env_fields=profile_workspace.profile_env_fields,
        active_profile_ids=profile_workspace.active_profile_ids,
        grid_radial_fields=grid_workspace.radial_fields,
        grid_k_max=int(grid_workspace.K_max),
        grid_l_max=int(grid_workspace.L_max),
        active_offsets=profile_workspace.active_offsets,
        active_scales=profile_workspace.active_scales,
        active_amplitude_powers=profile_workspace.active_amplitude_powers,
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        c_family_fields=profile_workspace.c_family_fields,
        s_family_fields=profile_workspace.s_family_fields,
        c_family_base_fields=profile_workspace.c_family_base_fields,
        s_family_base_fields=profile_workspace.s_family_base_fields,
        c_family_source_profile_ids=profile_workspace.c_family_source_profile_ids,
        s_family_source_profile_ids=profile_workspace.s_family_source_profile_ids,
        geometry_surface_fields=geometry_workspace.surface_fields,
        geometry_radial_fields=geometry_workspace.radial_fields,
        grid_poloidal_fields=grid_workspace.poloidal_fields,
        h_fields=profile_workspace.fields_for("h"),
        v_fields=profile_workspace.fields_for("v"),
        k_fields=profile_workspace.fields_for("k"),
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )


def build_fused_residual_pack_abi(
    *,
    grid_workspace: GridWorkspace,
    residual_binding_layout: ResidualBindingLayout,
    profile_workspace: ProfileWorkspace,
    residual_workspace: ResidualWorkspace,
    a: float,
    R0: float,
    B0: float,
) -> FusedResidualPackABI:
    """Collect packed residual assembly arrays for fused residual execution."""
    return FusedResidualPackABI(
        residual_pack_scratch=residual_workspace.pack_scratch,
        residual_pack_scratch_rows=residual_workspace.pack_scratch_rows,
        residual_surface_fields=residual_workspace.surface_fields,
        active_residual_block_codes=residual_binding_layout.active_residual_block_codes,
        active_residual_block_orders=residual_binding_layout.active_residual_block_orders,
        active_residual_block_radial_powers=(
            residual_binding_layout.active_residual_block_radial_powers
        ),
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        grid_radial_fields=grid_workspace.radial_fields,
        grid_poloidal_fields=grid_workspace.poloidal_fields,
        grid_k_max=int(grid_workspace.K_max),
        grid_l_max=int(grid_workspace.L_max),
        weights=grid_workspace.weights,
        a=a,
        R0=R0,
        B0=B0,
    )


def build_fused_source_eval_abi(
    *,
    source_plan: SourcePlan,
    grid_workspace: GridWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    B0: float,
    fix_rho: float,
) -> FusedSourceEvalABI:
    """Collect arrays and constants required by fused source evaluation."""
    # ``fix_rho`` is lowered once at bind time; source kernels only need the
    # integer cutoff for axis regularization.
    n_axis_fix = int(np.searchsorted(grid_workspace.rho, fix_rho))
    source_binding = NumbaSourceBindingPlan.from_source_plan(source_plan)

    return FusedSourceEvalABI(
        source_kernel=source_binding.kernel,
        coordinate_code=int(source_binding.coordinate_code),
        weights=grid_workspace.weights,
        differentiator=grid_workspace.differentiator,
        accumulator=grid_workspace.accumulator,
        grid_radial_fields=grid_workspace.radial_fields,
        n_axis_fix=n_axis_fix,
        radial_fields=geometry_workspace.radial_fields,
        surface_fields=geometry_workspace.surface_fields,
        scaled_Ip=float(source_plan.scaled_Ip),
        beta=float(source_plan.beta),
        array_scratch=source_workspace.array_scratch,
        matrix_scratch=source_workspace.matrix_scratch,
        B0=B0,
    )


def build_profile_owned_psin_source_abi(
    *,
    source_plan: SourcePlan,
    source_execution: SourceExecutionABI,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    source_workspace: SourceWorkspace,
) -> _ProfileOwnedPsinSourceABI:
    """Collect scratch arrays for routes where psin is an optimized profile."""
    del source_execution
    # This bundle is consumed only by Python-side runner binding; Numba kernels
    # still receive the individual arrays, not the dataclass object.
    return _ProfileOwnedPsinSourceABI(
        source_target_root_fields=source_workspace.target_root_fields,
        grid_radial_fields=grid_workspace.radial_fields,
        differentiator=grid_workspace.differentiator,
        accumulator=grid_workspace.accumulator,
        source_psin_query=source_workspace.psin_query,
        source_parameter_query=source_workspace.parameter_query,
        heat_spline_coeff=source_workspace.heat_spline_coeff,
        current_spline_coeff=source_workspace.current_spline_coeff,
        barycentric_weights=source_workspace.barycentric_weights,
        use_barycentric=bool(source_plan.uses_barycentric_interpolation),
        materialized_heat_input=source_workspace.materialized_heat_input,
        materialized_current_input=source_workspace.materialized_current_input,
        psin_profile_fields=profile_workspace.fields_for("psin"),
        parameterization_code=int(
            NumbaSourceBindingPlan.from_source_plan(source_plan).parameterization_code
        ),
        scaled_heat=source_plan.scaled_heat,
        scaled_current=source_plan.scaled_current,
    )
