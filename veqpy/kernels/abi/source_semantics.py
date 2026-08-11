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

from veqpy.kernels.abi.enums import SOURCE_DRIVER_BY_ROUTE
from veqpy.kernels.types import KernelSource, KernelTopology
from veqpy.numerics import interpolation_matrix, make_calculus, make_quadrature

MU0 = 4.0e-7 * np.pi
DEFAULT_SOURCE_FIX_RHO = 0.05
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
MU0_SCALED_DRIVER_ROUTES = frozenset({"PI", "PJ1", "PJ2", "PJ3"})
KERNEL_SOURCE_ADVICE = (
    "Pass raw case values to KernelSource; source lowering applies mu0 scaling once."
)


@dataclass(frozen=True, slots=True)
class MaterializedKernelSource:
    """Backend-internal source arrays after route-dependent scaling."""

    scaled_pprime: np.ndarray
    scaled_driver: np.ndarray
    scaled_p0: float
    scaled_Ip: float
    beta: float
    case_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scaled_pprime", _readonly_array(self.scaled_pprime))
        object.__setattr__(self, "scaled_driver", _readonly_array(self.scaled_driver))
        object.__setattr__(self, "scaled_p0", float(self.scaled_p0))
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
    expected_driver = SOURCE_DRIVER_BY_ROUTE[topology.route]
    if source.driver_name != expected_driver:
        raise ValueError(
            f"route {topology.route} requires driver {expected_driver!r}, "
            f"got {source.driver_name!r}"
        )
    _validate_source_length(topology, source)
    return materialize_source_inputs(
        route=topology.route,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        sample_count=topology.sample_count,
        grid_size=topology.Nr,
        quadrature=topology.quadrature,
        calculus=topology.calculus,
        parameterization=topology.source_parameterization,
        p=source.p,
        pprime=source.pprime,
        driver=source.driver_profile,
        p0=source.p0,
        Ip=source.Ip,
        beta=source.beta,
        pprime_name="pprime",
        driver_name=source.driver_name,
        advice=KERNEL_SOURCE_ADVICE,
        case_name=source.case_name if case_name is None else case_name,
    )


def _validate_source_length(topology: KernelTopology, source: KernelSource) -> None:
    expected_samples = topology.sample_count
    pressure_length = source.pressure_profile.size
    driver_length = source.driver_profile.size
    if pressure_length != expected_samples or driver_length != expected_samples:
        raise ValueError(
            f"case does not match kernel topology: {source.pressure_name} "
            f"and {source.driver_name} "
            f"must have length {expected_samples} for "
            f"route={topology.route}/{topology.coordinate}/{topology.nodes}, "
            f"got {pressure_length} and {driver_length}"
        )


def materialize_source_inputs(
    *,
    route: str,
    coordinate: str,
    nodes: str,
    sample_count: int,
    driver: np.ndarray,
    p: np.ndarray | None = None,
    pprime: np.ndarray | None = None,
    p0: float | None = None,
    Ip: float,
    beta: float,
    pprime_name: str,
    driver_name: str,
    advice: str,
    case_name: str | None = None,
    grid_size: int | None = None,
    quadrature: str = "legendre",
    calculus: str = "spectral",
    parameterization: str = "identity",
) -> MaterializedKernelSource:
    """Lower raw route inputs to the shared backend-internal source units."""

    route_key = str(route).upper()
    coordinate_key = str(coordinate).lower()
    nodes_key = str(nodes).lower()
    raw_pprime, raw_p0, pressure_mode = _lower_pressure_input(
        p=p,
        pprime=pprime,
        p0=p0,
        coordinate=coordinate_key,
        nodes=nodes_key,
        sample_count=int(sample_count),
        grid_size=grid_size,
        quadrature=quadrature,
        calculus=calculus,
        parameterization=parameterization,
        pprime_name=pprime_name,
        advice=advice,
    )
    scaled_p0 = _scale_pressure_offset_input(raw_p0, name="p0", advice=advice)
    if not np.any(raw_pprime != 0.0) and scaled_p0 == 0.0:
        if pressure_mode == "p":
            detail = "p is all zero"
        else:
            detail = f"{pprime_name} is all zero and p0 is zero"
        raise ValueError(
            f"The complete pressure profile is identically zero: {detail}. "
            "A non-zero pressure profile is required until the current-based "
            "alpha fallback is implemented."
        )
    return MaterializedKernelSource(
        scaled_pprime=_scale_pressure_like_input(
            raw_pprime, name=pprime_name, advice=advice
        ),
        scaled_driver=_scale_driver_input(
            driver,
            route=route_key,
            name=driver_name,
            advice=advice,
        ),
        scaled_p0=scaled_p0,
        scaled_Ip=_scale_physical_constraint(Ip, name="Ip", advice=advice),
        beta=beta,
        case_name=case_name,
    )


def _lower_pressure_input(
    *,
    p: np.ndarray | None,
    pprime: np.ndarray | None,
    p0: float | None,
    coordinate: str,
    nodes: str,
    sample_count: int,
    grid_size: int | None,
    quadrature: str,
    calculus: str,
    parameterization: str,
    pprime_name: str,
    advice: str,
) -> tuple[np.ndarray, float, str]:
    has_p = p is not None
    has_pprime = pprime is not None
    if has_p == has_pprime:
        supplied = "both" if has_p else "neither"
        raise ValueError(
            "Exactly one pressure input is required: p or pprime; "
            f"received {supplied}"
        )

    if has_p:
        if p0 is not None:
            raise ValueError("p0 is derived from p and cannot be supplied with p")
        pressure = _source_profile_array(p, name="p", sample_count=sample_count)
        _validate_finite_profile(pressure, name="p", advice=advice)
        derived_pprime, derived_p0 = _differentiate_pressure_profile(
            pressure,
            coordinate=coordinate,
            nodes=nodes,
            grid_size=grid_size,
            quadrature=quadrature,
            calculus=calculus,
            parameterization=parameterization,
        )
        return derived_pprime, derived_p0, "p"

    pressure_derivative = _source_profile_array(
        pprime,
        name=pprime_name,
        sample_count=sample_count,
    )
    return pressure_derivative, 0.0 if p0 is None else float(p0), "pprime"


def _differentiate_pressure_profile(
    pressure: np.ndarray,
    *,
    coordinate: str,
    nodes: str,
    grid_size: int | None,
    quadrature: str,
    calculus: str,
    parameterization: str,
) -> tuple[np.ndarray, float]:
    coordinate_nodes = _pressure_coordinate_nodes(
        coordinate=coordinate,
        nodes=nodes,
        sample_count=pressure.size,
        grid_size=grid_size,
        quadrature=quadrature,
        parameterization=parameterization,
    )
    if nodes == "uniform":
        edge_pressure = float(pressure[-1])
    else:
        edge_weights = interpolation_matrix(
            coordinate_nodes,
            np.array([1.0], dtype=np.float64),
        )[0]
        edge_pressure = float(np.dot(edge_weights, pressure))

    if np.all(pressure == pressure[0]):
        return np.zeros_like(pressure), float(pressure[0])

    if coordinate == "psin" and parameterization == "sqrt_psin":
        parameter_nodes = np.linspace(
            0.0,
            1.0,
            pressure.size,
            dtype=np.float64,
        )
        parameter_differentiator = _pressure_differentiator(
            parameter_nodes,
            calculus=calculus,
        )
        derivative_by_parameter = parameter_differentiator @ pressure
        pressure_derivative = np.empty_like(pressure)
        pressure_derivative[1:] = (
            derivative_by_parameter[1:] / (2.0 * parameter_nodes[1:])
        )
        second_derivative = parameter_differentiator @ derivative_by_parameter
        pressure_derivative[0] = 0.5 * second_derivative[0]
    else:
        differentiator = _pressure_differentiator(
            coordinate_nodes,
            calculus=calculus,
        )
        pressure_derivative = differentiator @ pressure
    return np.asarray(pressure_derivative, dtype=np.float64), edge_pressure


def _pressure_coordinate_nodes(
    *,
    coordinate: str,
    nodes: str,
    sample_count: int,
    grid_size: int | None,
    quadrature: str,
    parameterization: str,
) -> np.ndarray:
    if nodes == "grid":
        if coordinate == "psin":
            raise ValueError(
                "p input is not defined for coordinate='psin', nodes='grid': "
                "the grid is fixed in rho while psin(rho) is solved at runtime. "
                "Use pprime on grid nodes or use uniform psin samples."
            )
        size = sample_count if grid_size is None else int(grid_size)
        if size != sample_count:
            raise ValueError(
                "grid pressure input length must match the operator grid: "
                f"got {sample_count} and {size}"
            )
        source_nodes, _ = make_quadrature(size, scheme=quadrature)
        return np.asarray(source_nodes, dtype=np.float64)

    parameter_nodes = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    if parameterization == "identity":
        return parameter_nodes
    if coordinate == "psin" and parameterization == "sqrt_psin":
        return parameter_nodes * parameter_nodes
    raise ValueError(
        f"unsupported pressure parameterization {parameterization!r} "
        f"for coordinate={coordinate!r}, nodes={nodes!r}"
    )


def _small_polynomial_differentiator(nodes: np.ndarray) -> np.ndarray:
    count = int(nodes.size)
    if count == 1:
        return np.zeros((1, 1), dtype=np.float64)
    vander = np.vander(nodes, N=count, increasing=True)
    derivative_vander = np.zeros_like(vander)
    for degree in range(1, count):
        derivative_vander[:, degree] = degree * nodes ** (degree - 1)
    return np.linalg.solve(vander.T, derivative_vander.T).T


def _pressure_differentiator(
    nodes: np.ndarray,
    *,
    calculus: str,
) -> np.ndarray:
    if nodes.size < 4:
        return _small_polynomial_differentiator(nodes)
    return make_calculus(nodes, scheme=calculus)[1]


def _source_profile_array(
    value: np.ndarray | None,
    *,
    name: str,
    sample_count: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    if array.size != sample_count:
        raise ValueError(
            f"{name} must contain {sample_count} samples, got {array.size}"
        )
    return array


def _scale_pressure_like_input(value: np.ndarray, *, name: str, advice: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _validate_finite_profile(array, name=name, advice=advice)
    return _readonly_array(array * MU0)


def _scale_pressure_offset_input(value: float, *, name: str, advice: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        message = f"Rejected setup input value: {name} must be finite. {advice}"
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        raise ValueError(message)
    return scalar * MU0


def _scale_driver_input(
    value: np.ndarray,
    *,
    route: str,
    name: str,
    advice: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _validate_finite_profile(array, name=name, advice=advice)
    if route in MU0_SCALED_DRIVER_ROUTES:
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
