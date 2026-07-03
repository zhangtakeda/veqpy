"""Shared Python-side source input materialization for VEQlib-facing paths."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * np.pi
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
CURRENT_PROFILE_ROUTES = frozenset({"PI", "PJ1", "PJ2"})


@dataclass(frozen=True, slots=True)
class MaterializedSourceInputs:
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


def materialize_source_inputs(
    *,
    route: str,
    heat: np.ndarray,
    current: np.ndarray,
    Ip: float,
    beta: float,
    heat_name: str,
    current_name: str,
    advice: str,
    case_name: str | None = None,
) -> MaterializedSourceInputs:
    """Lower raw route inputs to the shared backend-internal source units."""

    route_key = str(route).upper()
    return MaterializedSourceInputs(
        scaled_heat=_scale_pressure_like_input(heat, name=heat_name, advice=advice),
        scaled_current=_scale_current_input(
            current,
            route=route_key,
            name=current_name,
            advice=advice,
        ),
        scaled_Ip=_scale_physical_constraint(Ip, name="Ip", advice=advice),
        beta=beta,
        case_name=case_name,
    )


def _scale_pressure_like_input(value: np.ndarray, *, name: str, advice: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs, advice=advice)
    return _readonly_array(array * MU0)


def _scale_current_input(
    value: np.ndarray,
    *,
    route: str,
    name: str,
    advice: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    if route in CURRENT_PROFILE_ROUTES:
        if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
            _reject_setup_magnitude(name=name, max_abs=max_abs, advice=advice)
        return _readonly_array(array * MU0)
    if not _in_closed_range(max_abs, SETUP_NORMALIZED_ABS_MIN, SETUP_NORMALIZED_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs, advice=advice)
    return _readonly_array(array)


def _scale_physical_constraint(value: float, *, name: str, advice: str) -> float:
    if not np.isfinite(value):
        return value
    max_abs = abs(float(value))
    if not _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        _reject_setup_magnitude(name=name, max_abs=max_abs, advice=advice)
    return float(value) * MU0


def _reject_setup_magnitude(*, name: str, max_abs: float, advice: str) -> None:
    magnitude_label = f"{name} abs" if name == "Ip" else f"{name} max_abs"
    message = (
        f"Rejected setup input magnitude: {magnitude_label}={max_abs:.6g}. "
        f"{advice}"
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise ValueError(message)


def _in_closed_range(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def _readonly_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.setflags(write=False)
    return arr
