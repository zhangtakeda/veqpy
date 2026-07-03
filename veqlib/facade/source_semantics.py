"""Source-case semantic lowering shared by Kernel facades.

``KernelSource`` is the public raw case input.  Native runtimes still consume the
same scaled arrays as the existing VEQlib core, so this module owns the one
Python-side conversion table during the Kernel API migration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .types import KernelSource, KernelTopology

MU0 = 4.0e-7 * np.pi
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
CURRENT_PROFILE_ROUTES = frozenset({"PI", "PJ1", "PJ2"})


@dataclass(frozen=True, slots=True)
class MaterializedKernelSource:
    """Backend-internal source arrays after route-dependent scaling."""

    scaled_heat: np.ndarray
    scaled_current: np.ndarray
    scaled_Ip: float
    beta: float
    case_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scaled_heat", _readonly_array(self.scaled_heat))
        object.__setattr__(self, "scaled_current", _readonly_array(self.scaled_current))
        object.__setattr__(self, "scaled_Ip", float(self.scaled_Ip))
        object.__setattr__(self, "beta", float(self.beta))
        case_name = None if self.case_name is None else str(self.case_name)
        object.__setattr__(self, "case_name", case_name)


def materialize_kernel_source(
    topology: KernelTopology,
    source: KernelSource,
    *,
    case_name: str | None = None,
) -> MaterializedKernelSource:
    """Validate one raw source case and lower it to backend-internal units."""

    if not isinstance(topology, KernelTopology):
        raise TypeError(f"topology must be KernelTopology, got {type(topology).__name__}")
    if not isinstance(source, KernelSource):
        raise TypeError(f"source must be KernelSource, got {type(source).__name__}")
    _validate_source_length(topology, source)
    route = topology.route
    return MaterializedKernelSource(
        scaled_heat=_scale_pressure_like_input(source.heat_profile, name="heat_profile"),
        scaled_current=_scale_current_input(source.current_profile, route=route),
        scaled_Ip=_scale_physical_constraint(source.Ip, name="Ip"),
        beta=source.beta,
        case_name=source.case_name if case_name is None else case_name,
    )


def _validate_source_length(topology: KernelTopology, source: KernelSource) -> None:
    expected_samples = topology.sample_count
    heat_length = source.heat_profile.size
    current_length = source.current_profile.size
    if heat_length != expected_samples or current_length != expected_samples:
        raise ValueError(
            "case does not match kernel topology: heat_profile and current_profile "
            f"must have length {expected_samples} for "
            f"route={topology.route}/{topology.coordinate}/{topology.nodes}, "
            f"got {heat_length} and {current_length}"
        )


def _scale_pressure_like_input(value: np.ndarray, *, name: str) -> np.ndarray:
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs)
    return _readonly_array(value * MU0)


def _scale_current_input(value: np.ndarray, *, route: str) -> np.ndarray:
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    if route in CURRENT_PROFILE_ROUTES:
        if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
            _reject_setup_magnitude(name="current_profile", max_abs=max_abs)
        return _readonly_array(value * MU0)
    if not _in_closed_range(max_abs, SETUP_NORMALIZED_ABS_MIN, SETUP_NORMALIZED_ABS_MAX):
        _reject_setup_magnitude(name="current_profile", max_abs=max_abs)
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
        "Pass raw case values to KernelSource; facade materialization applies mu0 scaling once."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise ValueError(message)


def _in_closed_range(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def _readonly_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.setflags(write=False)
    return arr
