"""
Module: numba_core.source_plan

Role:
- Own source route plans and source input validation.
- Keep source binding validation at bind-time, before runtime memory refresh and engine calls.

Notes:
- This module owns immutable source plans consumed by the kernel runtime.
- It does not allocate runtime arrays, run source kernels, or implement source mathematics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from veqlib.facade.source_semantics import materialize_source_inputs
from veqpy.model.numerics import (
    SOURCE_INTERP_DEFAULT,
    normalize_source_interpolation_kind,
    source_interpolation_kind_is_barycentric,
)

from .numba_source import (
    COORDINATE_CODES,
    source_parameterization_for_route_key,
)

RouteKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """Describe the read-only source semantics and runner binding plan.

    This is the semantic layer: route, coordinate, node layout, interpolation
    choice, plan-ready input arrays, and global constraints. ``scaled_heat``,
    ``scaled_current``, and ``scaled_Ip`` are the arrays/scalars consumed by
    layout binding after setup validation. Runtime ownership decisions are
    derived later in ``SourceExecutionABI``.
    """

    route: str
    kernel: Callable
    coordinate: str
    nodes: str
    parameterization: str
    source_sample_count: int
    scaled_heat: np.ndarray
    scaled_current: np.ndarray
    scaled_Ip: float
    beta: float
    interpolation_kind: str

    @property
    def is_grid_nodes(self) -> bool:
        """Whether source samples are already defined on the operator grid."""
        return self.nodes == "grid"

    @property
    def is_psin_coordinate(self) -> bool:
        """Whether source samples are parameterized by normalized flux."""
        return self.coordinate == "psin"

    @property
    def route_key(self) -> tuple[str, str, str]:
        """Normalized ``(route, coordinate, nodes)`` source dispatch key."""
        return (self.route, self.coordinate, self.nodes)

    @property
    def coordinate_code(self) -> int:
        """Integer coordinate code consumed by numba source kernels."""
        return int(COORDINATE_CODES[self.coordinate])

    @property
    def parameterization_code(self) -> int:
        """Integer source-parameterization code consumed by numba kernels."""
        return int(SOURCE_PARAMETERIZATION_CODES[self.parameterization])

    @property
    def uses_barycentric_interpolation(self) -> bool:
        """Whether non-grid source interpolation uses barycentric weights."""
        return not self.is_grid_nodes and source_interpolation_kind_is_barycentric(
            self.interpolation_kind
        )


SOURCE_PARAMETERIZATION_CODES = {
    "identity": 0,
    "sqrt_psin": 1,
}


def build_source_plan(
    *,
    case: object,
    source_route_spec: object,
    interpolation_kind: str = SOURCE_INTERP_DEFAULT,
) -> SourcePlan:
    """Build the immutable source plan for a runtime case."""
    scaled_heat, scaled_current, scaled_Ip, beta = _scaled_source_inputs(case)
    # Parameterization is route-specific.  For example PP/psin/uniform samples
    # on sqrt(psin) to bias resolution near the magnetic axis while all kernels
    # still exchange normalized psin/root fields internally.
    route_key = (
        str(case.route).upper(),
        str(case.coordinate).lower(),
        str(case.nodes).lower(),
    )
    return SourcePlan(
        route=str(case.route).upper(),
        kernel=source_route_spec.implementation,
        coordinate=str(case.coordinate).lower(),
        nodes=str(case.nodes).lower(),
        parameterization=source_parameterization_for_route_key(route_key),
        source_sample_count=int(scaled_heat.shape[0]),
        scaled_heat=scaled_heat,
        scaled_current=scaled_current,
        scaled_Ip=scaled_Ip,
        beta=beta,
        interpolation_kind=(
            # Grid-node sources are already sampled on operator rho; leave the
            # interpolation slot empty so runtime binding cannot remap them.
            ""
            if str(case.nodes).lower() == "grid"
            else normalize_source_interpolation_kind(interpolation_kind)
        ),
    )


def _scaled_source_inputs(case: object) -> tuple[np.ndarray, np.ndarray, float, float]:
    materialized = materialize_source_inputs(
        route=str(case.route).upper(),
        heat=case.heat_input,
        current=case.current_input,
        Ip=float(case.Ip),
        beta=float(case.beta),
        heat_name="heat_input",
        current_name="current_input",
        advice="Pass unnormalized runtime values; SourcePlan applies mu0 scaling once.",
    )
    return (
        materialized.scaled_heat,
        materialized.scaled_current,
        materialized.scaled_Ip,
        materialized.beta,
    )


def validate_source_plan_profile_support(
    *,
    source_plan: SourcePlan,
    source_execution: object,
    case: object,
) -> None:
    """Validate the source plan against active profile ownership."""
    route_key = source_plan.route_key
    if route_key != tuple(getattr(source_execution, "route_key")):
        raise ValueError(
            f"Source execution binding route mismatch: plan={route_key!r}, "
            f"binding={getattr(source_execution, 'route_key')!r}"
        )

    has_active_psin = int(getattr(source_execution, "psin_active_length", 0)) > 0
    has_active_F = int(getattr(source_execution, "f_active_length", 0)) > 0
    requires_active_F = bool(getattr(source_execution, "requires_optimized_f_profile", False))
    if has_active_F and not requires_active_F:
        raise ValueError(
            f"{case.route} expects no active F profile; "
            "active F is only supported for PJ2"
        )
    if requires_active_F and not has_active_F:
        raise ValueError(f"{case.route} requires an active F profile")
    if has_active_F and has_active_psin:
        raise ValueError("Active F and active psin profiles are mutually exclusive")
    if (
        bool(getattr(source_execution, "requires_optimized_psin_profile", False))
        and not has_active_psin
    ):
        # PF/PP/PI/PJ1/PQ psin-uniform routes query external source samples at
        # the current optimized psin each residual evaluation.
        raise ValueError(f"{case.route} requires an active psin profile")
    if (
        source_plan.is_psin_coordinate
        and has_active_psin
        and not bool(getattr(source_execution, "requires_optimized_psin_profile", False))
    ):
        # Source-owned psin routes reconstruct flux in the source kernel.  An
        # active psin profile would create two independent owners of the same
        # root field and stale source queries.
        raise ValueError(
            f"{case.route} expects no active psin profile"
        )


def validate_source_inputs(case: object, nr: int) -> None:
    """Validate source input lengths for grid-owned and sampled routes."""
    if case.heat_input.shape != case.current_input.shape:
        raise ValueError(
            "Expected heat_input/current_input to share a shape, "
            f"got {case.heat_input.shape} and {case.current_input.shape}"
        )
    if case.nodes == "grid" and case.heat_input.shape[0] != nr:
        # Grid-node routes skip interpolation entirely, so source samples must
        # already match the operator radial grid.
        raise ValueError(
            f"Expected grid inputs to have shape ({nr},), got {case.heat_input.shape}"
        )
    if case.heat_input.shape[0] < 1:
        raise ValueError(
            f"Expected {case.coordinate}-coordinate inputs to contain at least one sample"
        )
