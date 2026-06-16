"""
Module: operator.source_plan

Role:
- Own source route plans and source input validation.
- Keep user/model compatibility at bind-time, before runtime memory refresh and engine calls.

Notes:
- This module builds immutable plans from ``Problem`` and resolved route specs.
- It does not allocate runtime arrays, run source kernels, or implement source mathematics.
"""

from __future__ import annotations

import warnings
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
    from veqpy.model.problem import Problem

RouteKey = tuple[str, str, str]

MU0 = 4.0e-7 * np.pi
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
CURRENT_PROFILE_ROUTES = frozenset({"PI", "PJ1", "PJ2"})


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
    problem: Problem,
    source_route_spec: object,
    interpolation_kind: str = SOURCE_INTERP_DEFAULT,
) -> SourcePlan:
    """Build the immutable source plan for a ``Problem``."""
    scaled_heat, scaled_current, scaled_Ip, beta = _scaled_source_inputs(problem)
    # Parameterization is route-specific.  For example PP/psin/uniform samples
    # on sqrt(psin) to bias resolution near the magnetic axis while all kernels
    # still exchange normalized psin/root fields internally.
    route_key = (
        str(problem.route).upper(),
        str(problem.coordinate).lower(),
        str(problem.nodes).lower(),
    )
    return SourcePlan(
        route=str(problem.route).upper(),
        kernel=source_route_spec.implementation,
        coordinate=str(problem.coordinate).lower(),
        nodes=str(problem.nodes).lower(),
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
            if str(problem.nodes).lower() == "grid"
            else normalize_source_interpolation_kind(interpolation_kind)
        ),
    )


def _scaled_source_inputs(problem: Problem) -> tuple[np.ndarray, np.ndarray, float, float]:
    route = str(problem.route).upper()
    scaled_heat = _scale_pressure_like_input(problem.heat_input, name="heat_input")
    scaled_current = _scale_current_input(problem.current_input, route=route)
    scaled_Ip = _scale_physical_constraint(float(problem.Ip), name="Ip")
    return scaled_heat, scaled_current, scaled_Ip, float(problem.beta)


def _scale_pressure_like_input(value: np.ndarray, *, name: str) -> np.ndarray:
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs)
    return _readonly_array(value * MU0)


def _scale_current_input(value: np.ndarray, *, route: str) -> np.ndarray:
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    if route in CURRENT_PROFILE_ROUTES:
        if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
            _reject_setup_magnitude(name="current_input", max_abs=max_abs)
        return _readonly_array(value * MU0)
    if not _in_closed_range(max_abs, SETUP_NORMALIZED_ABS_MIN, SETUP_NORMALIZED_ABS_MAX):
        _reject_setup_magnitude(name="current_input", max_abs=max_abs)
    return _readonly_array(value)


def _scale_physical_constraint(value: float, *, name: str) -> float:
    if not np.isfinite(value):
        return value
    max_abs = abs(float(value))
    if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs)
    return float(value) * MU0


def _reject_setup_magnitude(*, name: str, max_abs: float) -> None:
    magnitude_label = f"{name} abs" if name == "Ip" else f"{name} max_abs"
    message = (
        f"Rejected setup input magnitude: {magnitude_label}={max_abs:.6g}. "
        "Pass unnormalized setup values to Problem; SourcePlan applies mu0 scaling once."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise ValueError(message)


def _in_closed_range(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def _readonly_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.setflags(write=False)
    return arr


def validate_source_plan_profile_support(
    *,
    source_plan: SourcePlan,
    source_execution: object,
    problem: Problem,
) -> None:
    """Validate source-plan compatibility with active profile ownership."""
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
            f"{problem.route} does not accept an active F profile; "
            "active F is only supported for PJ2"
        )
    if requires_active_F and not has_active_F:
        raise ValueError(f"{problem.route} requires an active F profile")
    if has_active_F and has_active_psin:
        raise ValueError("Active F and active psin profiles are mutually exclusive")
    if (
        bool(getattr(source_execution, "requires_optimized_psin_profile", False))
        and not has_active_psin
    ):
        # PF/PP/PI/PJ1/PQ psin-uniform routes query external source samples at
        # the current optimized psin each residual evaluation.
        raise ValueError(f"{problem.route} requires an active psin profile")
    if (
        source_plan.is_psin_coordinate
        and has_active_psin
        and not bool(getattr(source_execution, "requires_optimized_psin_profile", False))
    ):
        # Source-owned psin routes reconstruct flux in the source kernel.  An
        # active psin profile would create two independent owners of the same
        # root field and stale source queries.
        raise ValueError(
            f"{problem.route} does not accept an active psin profile"
        )


def validate_source_inputs(problem: Problem, nr: int) -> None:
    """Validate source input lengths for grid-owned and sampled routes."""
    if problem.heat_input.shape != problem.current_input.shape:
        raise ValueError(
            "Expected heat_input/current_input to share a shape, "
            f"got {problem.heat_input.shape} and {problem.current_input.shape}"
        )
    if problem.nodes == "grid" and problem.heat_input.shape[0] != nr:
        # Grid-node routes skip interpolation entirely, so source samples must
        # already match the operator radial grid.
        raise ValueError(
            f"Expected grid inputs to have shape ({nr},), got {problem.heat_input.shape}"
        )
    if problem.heat_input.shape[0] < 1:
        raise ValueError(
            f"Expected {problem.coordinate}-coordinate inputs to contain at least one sample"
        )
