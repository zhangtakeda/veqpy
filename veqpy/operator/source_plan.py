"""
Module: operator.source_plan

Role:
- Own source route plans and source input validation.
- Keep user/model compatibility at bind-time, before runtime memory refresh and engine calls.

Notes:
- This module builds immutable plans from ``OperatorCase`` and resolved route specs.
- It does not allocate runtime arrays, run source kernels, or implement source mathematics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine.numba_source import (
    COORDINATE_CODES,
    source_parameterization_for_route_key,
)
from veqpy.math.interpolate import (
    SOURCE_INTERP_DEFAULT,
    normalize_source_interpolation_kind,
    source_interpolation_kind_is_barycentric,
)

if TYPE_CHECKING:
    from veqpy.operator.operator_case import OperatorCase

RouteKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """Describe the read-only source semantics and runner binding plan.

    This is the semantic layer: route, coordinate, node layout, interpolation
    choice, input arrays, and global constraints.  Runtime ownership decisions
    are derived later in ``SourceExecutionABI``.
    """

    route: str
    kernel: Callable
    coordinate: str
    nodes: str
    parameterization: str
    source_sample_count: int
    heat_input: np.ndarray
    current_input: np.ndarray
    Ip: float
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


def _source_route_key(source_plan: SourcePlan) -> tuple[str, str, str]:
    return (source_plan.route, source_plan.coordinate, source_plan.nodes)


SOURCE_PARAMETERIZATION_CODES = {
    "identity": 0,
    "sqrt_psin": 1,
}


def build_source_plan(
    *,
    case: OperatorCase,
    source_route_spec: object,
    interpolation_kind: str = SOURCE_INTERP_DEFAULT,
) -> SourcePlan:
    """Build the immutable source plan for an ``OperatorCase``."""
    # Parameterization is route-specific.  For example PP/psin/uniform samples
    # on sqrt(psin) to bias resolution near the magnetic axis while all kernels
    # still exchange normalized psin/root fields internally.
    return SourcePlan(
        route=str(case.route).upper(),
        kernel=source_route_spec.implementation,
        coordinate=str(case.coordinate).lower(),
        nodes=str(case.nodes).lower(),
        parameterization=source_parameterization_for_route_key(
            (str(case.route).upper(), str(case.coordinate).lower(), str(case.nodes).lower())
        ),
        source_sample_count=int(case.heat_input.shape[0]),
        heat_input=case.heat_input,
        current_input=case.current_input,
        Ip=float(case.Ip),
        beta=float(case.beta),
        interpolation_kind=(
            # Grid-node sources are already sampled on operator rho; leave the
            # interpolation slot empty so runtime binding cannot remap them.
            ""
            if str(case.nodes).lower() == "grid"
            else normalize_source_interpolation_kind(interpolation_kind)
        ),
    )


def validate_source_plan_profile_support(
    *,
    source_plan: SourcePlan,
    source_execution: object,
    case: OperatorCase,
) -> None:
    """Validate source-plan compatibility with active profile ownership."""
    route_key = _source_route_key(source_plan)
    if route_key != tuple(getattr(source_execution, "route_key")):
        raise ValueError(
            f"Source execution binding route mismatch: plan={route_key!r}, "
            f"binding={getattr(source_execution, 'route_key')!r}"
        )

    has_active_psin = int(getattr(source_execution, "psin_active_length", 0)) > 0
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
            f"{case.route} does not accept an active psin profile because psin is source-owned"
        )


def validate_source_inputs(case: OperatorCase, nr: int) -> None:
    """Validate source input lengths for grid-owned and sampled routes."""
    if case.heat_input.shape != case.current_input.shape:
        raise ValueError(
            "Expected heat_input/current_input to share a shape, "
            f"got {case.heat_input.shape} and {case.current_input.shape}"
        )
    if case.nodes == "grid" and case.heat_input.shape[0] != nr:
        # Grid-node routes skip interpolation entirely, so source samples must
        # already match the operator radial grid.
        raise ValueError(f"Expected grid inputs to have shape ({nr},), got {case.heat_input.shape}")
    if case.heat_input.shape[0] < 1:
        raise ValueError(
            f"Expected {case.coordinate}-coordinate inputs to contain at least one sample"
        )
