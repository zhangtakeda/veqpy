"""
Module: engine.backend_abi

Role:
- Define explicit ABI binding contracts used by the numba fused backend.
- Move bind-time data selection out of numba implementation into the engine ABI module.

Public API:
- SourceExecutionABI
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
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine.numba_source import SOURCE_ROUTE_KEYS, resolve_source_scratch_kernel

if TYPE_CHECKING:
    from veqpy.operator.build_plan import ResidualBindingLayout
    from veqpy.operator.source_plan import SourcePlan
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace


RouteKey = tuple[str, str, str]

PROFILE_OWNED_PSIN_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {
        ("PF", "psin", "uniform"),
        ("PP", "psin", "uniform"),
        ("PI", "psin", "uniform"),
        ("PJ1", "psin", "uniform"),
        ("PQ", "psin", "uniform"),
    }
)


SUPPORTED_FUSED_SOURCE_ROUTE_KEYS: frozenset[RouteKey] = frozenset(SOURCE_ROUTE_KEYS)


@dataclass(frozen=True, slots=True)
class SourceExecutionABI:
    """Source-route execution requirements derived at bind time."""

    route_key: RouteKey
    psin_active_length: int
    has_active_f_profile: bool
    requires_optimized_psin_profile: bool
    requires_psin_query_workspace: bool
    requires_source_parameter_query: bool
    requires_target_root_fields: bool


def _active_profile_slot_and_length(
    name: str,
    *,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    active_profile_ids: np.ndarray,
) -> tuple[int, int]:
    profile_id = int(profile_index.get(name, -1))
    if profile_id < 0:
        return -1, 0
    length = int(profile_L[profile_id]) + 1
    if length <= 0:
        return -1, 0

    active_slot = -1
    for slot, active_profile_id in enumerate(active_profile_ids):
        if int(active_profile_id) == profile_id:
            active_slot = int(slot)
            break

    return active_slot, length


def build_source_execution_abi(
    *,
    source_plan: SourcePlan,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_profile_ids: np.ndarray,
) -> SourceExecutionABI:
    """Build source-route ABI metadata from a source plan and packed profile layout."""
    route_key = (source_plan.route, source_plan.coordinate, source_plan.nodes)
    del coeff_index  # preserved in the signature for call-site compatibility
    psin_active_slot, psin_active_length = _active_profile_slot_and_length(
        "psin",
        profile_index=profile_index,
        profile_L=profile_L,
        active_profile_ids=active_profile_ids,
    )
    F_active_slot, F_active_length = _active_profile_slot_and_length(
        "F",
        profile_index=profile_index,
        profile_L=profile_L,
        active_profile_ids=active_profile_ids,
    )

    if route_key not in SUPPORTED_FUSED_SOURCE_ROUTE_KEYS:
        raise ValueError(f"Unsupported source route key {route_key!r}")
    if psin_active_length > 0 and psin_active_slot < 0:
        raise ValueError("psin is active but has no active profile slot")
    if F_active_length > 0 and F_active_slot < 0:
        raise ValueError("F is active but has no active profile slot")
    if route_key[0] == "PQ" and F_active_length > 0:
        raise ValueError("PQ strict routes do not accept an active F profile")

    requires_optimized_psin_profile = route_key in PROFILE_OWNED_PSIN_ROUTE_KEYS
    if requires_optimized_psin_profile and psin_active_length <= 0:
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} requires an active psin profile"
        )
    if (
        source_plan.coordinate == "psin"
        and not requires_optimized_psin_profile
        and psin_active_length > 0
    ):
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} does not accept an active psin "
            "profile because psin is source-owned"
        )

    is_pj2_psin_uniform = route_key == ("PJ2", "psin", "uniform")
    return SourceExecutionABI(
        route_key=route_key,
        psin_active_length=psin_active_length,
        has_active_f_profile=F_active_length > 0,
        requires_optimized_psin_profile=requires_optimized_psin_profile,
        requires_psin_query_workspace=(requires_optimized_psin_profile or is_pj2_psin_uniform),
        requires_source_parameter_query=bool(
            source_plan.coordinate == "psin" and source_plan.parameterization != "identity"
        ),
        requires_target_root_fields=(requires_optimized_psin_profile or is_pj2_psin_uniform),
    )


@dataclass(frozen=True, slots=True)
class FusedHotRuntimeABI:
    """Array bundle required by fused profile/geometry hot-path kernels."""

    profile_fields: np.ndarray
    profile_rp_fields: np.ndarray
    profile_env_fields: np.ndarray
    active_profile_ids: np.ndarray
    T: np.ndarray
    T_r: np.ndarray
    T_rr: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
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
    rho: np.ndarray
    theta: np.ndarray
    cos_mtheta: np.ndarray
    sin_mtheta: np.ndarray
    m_cos_mtheta: np.ndarray
    m_sin_mtheta: np.ndarray
    m2_cos_mtheta: np.ndarray
    m2_sin_mtheta: np.ndarray
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray
    f_profile_fields: np.ndarray
    has_active_f_profile: bool
    c_active_order: int
    s_active_order: int
    a: float
    R0: float
    Z0: float


@dataclass(frozen=True, slots=True)
class FusedResidualPackABI:
    """Array bundle required to pack fused residual blocks."""

    residual_pack_scratch: np.ndarray
    residual_surface_fields: np.ndarray
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray
    active_coeff_index_rows: np.ndarray
    active_lengths: np.ndarray
    sin_mtheta: np.ndarray
    cos_mtheta: np.ndarray
    rho_powers: np.ndarray
    y: np.ndarray
    T: np.ndarray
    weights: np.ndarray
    a: float
    R0: float
    B0: float


@dataclass(frozen=True, slots=True)
class FusedSourceEvalABI:
    """Array and kernel bundle required by fused source evaluation."""

    source_kernel: Callable
    scratch_source_kernel: Callable | None
    coordinate_code: int
    weights: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    rho: np.ndarray
    n_axis_fix: int
    radial_fields: np.ndarray
    surface_fields: np.ndarray
    f_profile_u: np.ndarray
    Ip: float
    beta: float
    source_scratch_1d: np.ndarray
    source_scratch_2d: np.ndarray
    B0: float


def build_fused_hot_runtime_abi(
    *,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_execution: SourceExecutionABI,
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
        T=grid_workspace.T,
        T_r=grid_workspace.T_r,
        T_rr=grid_workspace.T_rr,
        active_offsets=profile_workspace.active_offsets,
        active_scales=profile_workspace.active_scales,
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
        rho=grid_workspace.rho,
        theta=grid_workspace.theta,
        cos_mtheta=grid_workspace.cos_mtheta,
        sin_mtheta=grid_workspace.sin_mtheta,
        m_cos_mtheta=grid_workspace.m_cos_mtheta,
        m_sin_mtheta=grid_workspace.m_sin_mtheta,
        m2_cos_mtheta=grid_workspace.m2_cos_mtheta,
        m2_sin_mtheta=grid_workspace.m2_sin_mtheta,
        h_fields=profile_workspace.fields_for("h"),
        v_fields=profile_workspace.fields_for("v"),
        k_fields=profile_workspace.fields_for("k"),
        f_profile_fields=profile_workspace.fields_for("F"),
        has_active_f_profile=bool(source_execution.has_active_f_profile),
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
        residual_surface_fields=residual_workspace.surface_fields,
        active_residual_block_codes=residual_binding_layout.active_residual_block_codes,
        active_residual_block_orders=residual_binding_layout.active_residual_block_orders,
        active_residual_block_radial_powers=(
            residual_binding_layout.active_residual_block_radial_powers
        ),
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        sin_mtheta=grid_workspace.sin_mtheta,
        cos_mtheta=grid_workspace.cos_mtheta,
        rho_powers=grid_workspace.rho_powers,
        y=grid_workspace.y,
        T=grid_workspace.T,
        weights=grid_workspace.weights,
        a=a,
        R0=R0,
        B0=B0,
    )


def build_fused_source_eval_abi(
    *,
    source_plan: SourcePlan,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    B0: float,
    fix_rho: float,
) -> FusedSourceEvalABI:
    """Collect arrays and constants required by fused source evaluation."""
    source_kernel = source_plan.kernel

    n_axis_fix = int(np.searchsorted(grid_workspace.rho, fix_rho))

    return FusedSourceEvalABI(
        source_kernel=source_kernel,
        scratch_source_kernel=resolve_source_scratch_kernel(source_kernel),
        coordinate_code=int(source_plan.coordinate_code),
        weights=grid_workspace.weights,
        differentiator=grid_workspace.differentiator,
        accumulator=grid_workspace.accumulator,
        rho=grid_workspace.rho,
        n_axis_fix=n_axis_fix,
        radial_fields=geometry_workspace.radial_fields,
        surface_fields=geometry_workspace.surface_fields,
        f_profile_u=profile_workspace.values_for("F"),
        Ip=float(source_plan.Ip),
        beta=float(source_plan.beta),
        source_scratch_1d=source_workspace.scratch_1d,
        source_scratch_2d=source_workspace.scratch_2d,
        B0=B0,
    )


def build_profile_owned_psin_source_abi(
    *,
    source_plan: SourcePlan,
    source_execution: SourceExecutionABI,
    grid_workspace: GridWorkspace,
    profile_workspace: ProfileWorkspace,
    source_workspace: SourceWorkspace,
) -> SimpleNamespace:
    """Collect scratch arrays for routes where psin is an optimized profile."""
    del source_execution
    return SimpleNamespace(
        source_target_root_fields=source_workspace.target_root_fields,
        rho=grid_workspace.rho,
        differentiator=grid_workspace.differentiator,
        accumulator=grid_workspace.accumulator,
        source_psin_query=source_workspace.psin_query,
        source_parameter_query=source_workspace.parameter_query,
        heat_spline_coeff=source_workspace.heat_spline_coeff,
        current_spline_coeff=source_workspace.current_spline_coeff,
        barycentric_weights=source_workspace.barycentric_weights,
        use_barycentric=bool(source_plan.uses_barycentric_interpolation),
        endpoint_blend=source_workspace.endpoint_blend,
        materialized_heat_input=source_workspace.materialized_heat_input,
        materialized_current_input=source_workspace.materialized_current_input,
        psin_profile_fields=profile_workspace.fields_for("psin"),
        parameterization_code=int(source_plan.parameterization_code),
        heat_input=source_plan.heat_input,
        current_input=source_plan.current_input,
    )
