"""
Module: operator.source_execution

Role:
- Derive backend-neutral source ownership and workspace requirements.
- Keep active psin/F ownership validation independent from concrete kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from veqpy.operator.source_routes import SOURCE_ROUTE_KEY_SET, RouteKey

if TYPE_CHECKING:
    from veqpy.model.problem import Problem
    from veqpy.operator.source_plan import SourcePlan


PROFILE_OWNED_PSIN_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {
        ("PF", "psin", "uniform"),
        ("PP", "psin", "uniform"),
        ("PI", "psin", "uniform"),
        ("PJ1", "psin", "uniform"),
        ("PQ", "psin", "uniform"),
    }
)
SUPPORTED_SOURCE_ROUTE_KEYS: frozenset[RouteKey] = frozenset(SOURCE_ROUTE_KEY_SET)


@dataclass(frozen=True, slots=True)
class SourceExecutionPlan:
    """Backend-neutral source route execution requirements."""

    route_key: RouteKey
    psin_active_length: int
    f_active_length: int
    requires_optimized_psin_profile: bool
    requires_optimized_f_profile: bool
    requires_psin_query_workspace: bool
    requires_source_parameter_query: bool
    requires_target_root_fields: bool


def active_profile_slot_and_length(
    name: str,
    *,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    active_profile_ids: np.ndarray,
) -> tuple[int, int]:
    """Return active slot and coefficient length for a named profile."""

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


def build_source_execution_plan(
    *,
    source_plan: SourcePlan,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_profile_ids: np.ndarray,
) -> SourceExecutionPlan:
    """Build backend-neutral source ownership metadata from layout state."""

    route_key = source_plan.route_key
    del coeff_index  # preserved for adapter compatibility during the ABI split

    psin_active_slot, psin_active_length = active_profile_slot_and_length(
        "psin",
        profile_index=profile_index,
        profile_L=profile_L,
        active_profile_ids=active_profile_ids,
    )
    f_active_slot, f_active_length = active_profile_slot_and_length(
        "F",
        profile_index=profile_index,
        profile_L=profile_L,
        active_profile_ids=active_profile_ids,
    )

    if route_key not in SUPPORTED_SOURCE_ROUTE_KEYS:
        raise ValueError(f"Unsupported source route key {route_key!r}")
    if psin_active_length > 0 and psin_active_slot < 0:
        raise ValueError("psin is active but has no active profile slot")
    if f_active_length > 0 and f_active_slot < 0:
        raise ValueError("F is active but has no active profile slot")

    requires_optimized_psin_profile = route_key in PROFILE_OWNED_PSIN_ROUTE_KEYS
    requires_optimized_f_profile = route_key[0] == "PJ2"

    if f_active_length > 0 and not requires_optimized_f_profile:
        raise ValueError(
            f"{route_key[0]} does not accept an active F profile; "
            "active F is only supported for PJ2"
        )
    if requires_optimized_f_profile and f_active_length <= 0:
        raise ValueError(f"{route_key[0]} requires an active F profile")
    if f_active_length > 0 and psin_active_length > 0:
        raise ValueError("Active F and active psin profiles are mutually exclusive")

    if requires_optimized_psin_profile and psin_active_length <= 0:
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} requires an active psin profile"
        )
    if not requires_optimized_psin_profile and psin_active_length > 0:
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} does not accept an active psin "
            "profile"
        )

    is_pj2_psin_uniform = route_key == ("PJ2", "psin", "uniform")
    return SourceExecutionPlan(
        route_key=route_key,
        psin_active_length=psin_active_length,
        f_active_length=f_active_length,
        requires_optimized_psin_profile=requires_optimized_psin_profile,
        requires_optimized_f_profile=requires_optimized_f_profile,
        requires_psin_query_workspace=(requires_optimized_psin_profile or is_pj2_psin_uniform),
        requires_source_parameter_query=bool(
            source_plan.coordinate == "psin" and source_plan.parameterization != "identity"
        ),
        requires_target_root_fields=(requires_optimized_psin_profile or is_pj2_psin_uniform),
    )


def validate_source_plan_profile_support(
    *,
    source_plan: SourcePlan,
    source_execution: object,
    problem: Problem,
) -> None:
    """Validate active profile ownership for a source plan."""

    route_key = source_plan.route_key
    if route_key != tuple(getattr(source_execution, "route_key")):
        raise ValueError(
            f"Source execution binding route mismatch: plan={route_key!r}, "
            f"binding={getattr(source_execution, 'route_key')!r}"
        )

    has_active_psin = int(getattr(source_execution, "psin_active_length", 0)) > 0
    has_active_f = int(getattr(source_execution, "f_active_length", 0)) > 0
    requires_active_f = bool(getattr(source_execution, "requires_optimized_f_profile", False))
    if has_active_f and not requires_active_f:
        raise ValueError(
            f"{problem.route} does not accept an active F profile; "
            "active F is only supported for PJ2"
        )
    if requires_active_f and not has_active_f:
        raise ValueError(f"{problem.route} requires an active F profile")
    if has_active_f and has_active_psin:
        raise ValueError("Active F and active psin profiles are mutually exclusive")
    if (
        bool(getattr(source_execution, "requires_optimized_psin_profile", False))
        and not has_active_psin
    ):
        raise ValueError(f"{problem.route} requires an active psin profile")
    if (
        source_plan.is_psin_coordinate
        and has_active_psin
        and not bool(getattr(source_execution, "requires_optimized_psin_profile", False))
    ):
        raise ValueError(f"{problem.route} does not accept an active psin profile")
