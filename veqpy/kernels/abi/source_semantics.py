"""
Module: veqpy.kernels.abi.source_semantics

Role:
- Lower public raw ``KernelSource`` inputs into backend-internal source units.

Notes:
- Source route validation and mu0 scaling are centralized here so Cxx and Numba
  backends consume the same materialized source arrays.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from veqpy.kernels.types import KernelSource, KernelTopology
from veqpy.numerics import make_quadrature

MU0 = 4.0e-7 * np.pi
DEFAULT_SOURCE_FIX_RHO = 0.05
SOURCE_REGULARITY_RTOL = 5.0e-2
SOURCE_REGULARITY_ATOL = 1.0e-10
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
CURRENT_PROFILE_ROUTES = frozenset({"PI", "PJ1", "PJ2"})
KERNEL_SOURCE_ADVICE = (
    "Pass raw case values to KernelSource; source lowering applies mu0 scaling once."
)


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
    return materialize_source_inputs(
        route=topology.route,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        sample_count=topology.sample_count,
        grid_size=topology.Nr,
        quadrature=topology.quadrature,
        parameterization=topology.source_parameterization,
        heat=source.heat_profile,
        current=source.current_profile,
        Ip=source.Ip,
        beta=source.beta,
        heat_name="heat_profile",
        current_name="current_profile",
        advice=KERNEL_SOURCE_ADVICE,
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


def materialize_source_inputs(
    *,
    route: str,
    coordinate: str,
    nodes: str,
    sample_count: int,
    heat: np.ndarray,
    current: np.ndarray,
    Ip: float,
    beta: float,
    heat_name: str,
    current_name: str,
    advice: str,
    case_name: str | None = None,
    grid_size: int | None = None,
    quadrature: str = "legendre",
    parameterization: str = "identity",
    fix_rho: float = DEFAULT_SOURCE_FIX_RHO,
) -> MaterializedKernelSource:
    """Lower raw route inputs to the shared backend-internal source units."""

    route_key = str(route).upper()
    coordinate_key = str(coordinate).lower()
    nodes_key = str(nodes).lower()
    rho = _source_rho_axis(
        coordinate=coordinate_key,
        nodes=nodes_key,
        sample_count=int(sample_count),
        grid_size=grid_size,
        quadrature=quadrature,
        parameterization=parameterization,
    )
    regular_heat, regular_current = _regularize_source_profiles(
        route=route_key,
        coordinate=coordinate_key,
        nodes=nodes_key,
        rho=rho,
        heat=heat,
        current=current,
        heat_name=heat_name,
        current_name=current_name,
        fix_rho=float(fix_rho),
    )
    return MaterializedKernelSource(
        scaled_heat=_scale_pressure_like_input(regular_heat, name=heat_name, advice=advice),
        scaled_current=_scale_current_input(
            regular_current,
            route=route_key,
            name=current_name,
            advice=advice,
        ),
        scaled_Ip=_scale_physical_constraint(Ip, name="Ip", advice=advice),
        beta=beta,
        case_name=case_name,
    )


def _source_rho_axis(
    *,
    coordinate: str,
    nodes: str,
    sample_count: int,
    grid_size: int | None,
    quadrature: str,
    parameterization: str,
) -> np.ndarray:
    if nodes == "grid":
        size = sample_count if grid_size is None else int(grid_size)
        rho, _ = make_quadrature(size, scheme=quadrature)
        return np.asarray(rho, dtype=np.float64)

    axis = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    if coordinate == "psin" and parameterization != "sqrt_psin":
        return np.sqrt(axis)
    return axis


def _regularize_source_profiles(
    *,
    route: str,
    coordinate: str,
    nodes: str,
    rho: np.ndarray,
    heat: np.ndarray,
    current: np.ndarray,
    heat_name: str,
    current_name: str,
    fix_rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    heat_array = np.asarray(heat, dtype=np.float64)
    current_array = np.asarray(current, dtype=np.float64)
    heat_kind, current_kind = _source_regularity_kinds(route, coordinate)
    context = f"{route}/{coordinate}/{nodes}"
    return (
        _regularize_source_profile_if_needed(
            heat_array,
            rho=rho,
            kind=heat_kind,
            name=heat_name,
            context=context,
            fix_rho=fix_rho,
        ),
        _regularize_source_profile_if_needed(
            current_array,
            rho=rho,
            kind=current_kind,
            name=current_name,
            context=context,
            fix_rho=fix_rho,
        ),
    )


def _source_regularity_kinds(route: str, coordinate: str) -> tuple[str, str]:
    heat_kind = "linear" if coordinate == "rho" else "even"
    if route in {"PF"}:
        current_kind = "linear" if coordinate == "rho" else "even"
    elif route == "PP":
        current_kind = "linear"
    elif route == "PI":
        current_kind = "quadratic"
    elif route in {"PJ1", "PJ2", "PQ"}:
        current_kind = "even"
    else:
        current_kind = "even"
    return heat_kind, current_kind


def _regularize_source_profile_if_needed(
    value: np.ndarray,
    *,
    rho: np.ndarray,
    kind: str,
    name: str,
    context: str,
    fix_rho: float,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    n_fix = int(np.searchsorted(rho, fix_rho))
    if n_fix <= 0 or n_fix + 1 >= array.size:
        return array
    if not np.all(np.isfinite(array)):
        return array

    repaired = array.copy()
    if kind == "linear":
        _regularize_axis_linear(repaired, rho, n_fix)
    elif kind == "quadratic":
        _regularize_axis_quadratic(repaired, rho, n_fix)
    else:
        _regularize_axis_even(repaired, rho, n_fix)

    head_original = array[:n_fix]
    head_repaired = repaired[:n_fix]
    abs_delta = float(np.max(np.abs(head_original - head_repaired)))
    scale = max(
        float(np.max(np.abs(array))),
        float(np.max(np.abs(repaired))),
        1.0,
    )
    if abs_delta <= SOURCE_REGULARITY_ATOL + SOURCE_REGULARITY_RTOL * scale:
        return array

    message = (
        f"Adjusted source axis regularity: {name} on {context} deviates from "
        f"{kind} magnetic-axis behavior by {abs_delta:.6g}."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return repaired


def _regularize_axis_linear(profile: np.ndarray, rho: np.ndarray, n_fix: int) -> None:
    anchor0 = n_fix
    anchor1 = n_fix + 1
    rho0 = rho[anchor0]
    rho1 = rho[anchor1]
    x0 = rho0 * rho0
    x1 = rho1 * rho1
    ratio0 = profile[anchor0] / rho0
    ratio1 = profile[anchor1] / rho1
    gradient = (ratio1 - ratio0) / (x1 - x0)
    for i in range(n_fix):
        x = rho[i] * rho[i]
        profile[i] = rho[i] * (ratio0 + gradient * (x - x0))


def _regularize_axis_quadratic(profile: np.ndarray, rho: np.ndarray, n_fix: int) -> None:
    anchor0 = n_fix
    anchor1 = n_fix + 1
    rho0 = rho[anchor0]
    rho1 = rho[anchor1]
    x0 = rho0 * rho0
    x1 = rho1 * rho1
    ratio0 = profile[anchor0] / x0
    ratio1 = profile[anchor1] / x1
    gradient = (ratio1 - ratio0) / (x1 - x0)
    for i in range(n_fix):
        x = rho[i] * rho[i]
        profile[i] = x * (ratio0 + gradient * (x - x0))


def _regularize_axis_even(profile: np.ndarray, rho: np.ndarray, n_fix: int) -> None:
    anchor0 = n_fix
    anchor1 = n_fix + 1
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    value0 = profile[anchor0]
    value1 = profile[anchor1]
    gradient = (value1 - value0) / (x1 - x0)
    for i in range(n_fix):
        x = rho[i] * rho[i]
        profile[i] = value0 + gradient * (x - x0)


def _scale_pressure_like_input(value: np.ndarray, *, name: str, advice: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _validate_finite_profile(array, name=name, advice=advice)
    return _readonly_array(array * MU0)


def _scale_current_input(
    value: np.ndarray,
    *,
    route: str,
    name: str,
    advice: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _validate_finite_profile(array, name=name, advice=advice)
    if route in CURRENT_PROFILE_ROUTES:
        return _readonly_array(array * MU0)
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


def _validate_finite_profile(array: np.ndarray, *, name: str, advice: str) -> None:
    if np.all(np.isfinite(array)):
        return
    message = f"Rejected setup input values: {name} must contain only finite values. {advice}"
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise ValueError(message)


def _in_closed_range(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def _readonly_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.setflags(write=False)
    return arr
