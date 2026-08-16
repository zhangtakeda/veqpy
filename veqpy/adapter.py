"""VEQ input adapter: three explicit dictionaries to one stable KernelInput."""

from __future__ import annotations

import numpy as np

from .kernels.abi.enums import (
    PRESSURE_DERIVATIVE_BY_COORDINATE,
    SOURCE_COORDINATE_CODES,
    SOURCE_DRIVER_NAMES,
    source_driver_for,
)
from .kernels.contracts import KernelInput, KernelTopology

_BOUNDARY_KEYS = frozenset(
    {"a", "R0", "Z0", "B0", "kappa_lcfs", "c_lcfs", "s_lcfs"}
)
_PRESSURE_KEYS = frozenset({"P", *PRESSURE_DERIVATIVE_BY_COORDINATE.values()})


class VEQAdapter:
    """Validate and copy one standalone VEQ problem without source remapping."""

    def __init__(self, topology: KernelTopology, buffer: KernelInput) -> None:
        self.topology = topology
        self.buffer = buffer

    def fill(self, boundary: dict, source: dict, targets: dict) -> int:
        """Overwrite the resident Kernel input from three exact dictionaries."""

        _require_exact_dict(boundary, "boundary")
        _require_exact_dict(source, "source")
        _require_exact_dict(targets, "targets")
        _require_exact_keys(boundary, _BOUNDARY_KEYS, "boundary")

        coordinate = self.topology.coordinate
        coordinate_keys = set(source).intersection(SOURCE_COORDINATE_CODES)
        if coordinate_keys != {coordinate}:
            raise ValueError(
                f"source must contain exactly the topology coordinate {coordinate!r}; "
                f"got {sorted(coordinate_keys)}"
            )

        pressure_keys = set(source).intersection(_PRESSURE_KEYS)
        expected_derivative = PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]
        if len(pressure_keys) != 1:
            raise ValueError(
                "source requires exactly one pressure input: "
                f"'P' or {expected_derivative!r}"
            )
        pressure_name = next(iter(pressure_keys))
        if pressure_name not in {"P", expected_derivative}:
            raise ValueError(
                f"coordinate={coordinate!r} requires pressure input 'P' or "
                f"{expected_derivative!r}, got {pressure_name!r}"
            )

        expected_driver = source_driver_for(self.topology.route, coordinate)
        driver_keys = set(source).intersection(SOURCE_DRIVER_NAMES)
        if driver_keys != {expected_driver}:
            raise ValueError(
                f"route={self.topology.route!r}, coordinate={coordinate!r} requires "
                f"source driver {expected_driver!r}; got {sorted(driver_keys)}"
            )

        expected_source_keys = {coordinate, pressure_name, expected_driver}
        if pressure_name == "P":
            if "P0" in source:
                raise ValueError("source P0 cannot be supplied with the full pressure profile P")
        else:
            expected_source_keys.add("P0")
        _require_exact_keys(source, frozenset(expected_source_keys), "source")

        expected_targets = {
            "none": frozenset(),
            "ip": frozenset({"Ip"}),
            "beta": frozenset({"beta"}),
            "both": frozenset({"Ip", "beta"}),
        }[self.topology.constraint]
        _require_exact_keys(targets, expected_targets, "targets")

        nodes = _canonical_normalized_nodes(source[coordinate], name=coordinate)
        pressure = _source_profile(source[pressure_name], pressure_name, nodes.shape)
        driver = _source_profile(source[expected_driver], expected_driver, nodes.shape)
        count = int(nodes.size)
        if count > 1024:
            raise ValueError("VEQ source input has more than the supported 1024 nodes")

        capacity = 256
        while capacity < count:
            capacity *= 2
        if capacity > self.buffer.source_capacity:
            self.buffer.grow_source_capacity(capacity)

        self.buffer.a = _positive_float(boundary["a"], "boundary.a")
        self.buffer.R0 = _positive_float(boundary["R0"], "boundary.R0")
        self.buffer.Z0 = _finite_float(boundary["Z0"], "boundary.Z0")
        self.buffer.B0 = _finite_float(boundary["B0"], "boundary.B0")
        if self.buffer.B0 == 0.0:
            raise ValueError("boundary.B0 must be nonzero")
        self.buffer.kappa_lcfs = _positive_float(
            boundary["kappa_lcfs"], "boundary.kappa_lcfs"
        )
        _copy_resized(self.buffer.c_lcfs, boundary["c_lcfs"], "boundary.c_lcfs")
        _copy_resized(self.buffer.s_lcfs, boundary["s_lcfs"], "boundary.s_lcfs")
        self.buffer.source_count = count
        self.buffer.source_nodes[:count] = nodes
        self.buffer.pressure[:count] = pressure
        self.buffer.driver[:count] = driver
        self.buffer.pressure_code = 0 if pressure_name == "P" else 1
        self.buffer.p0 = (
            0.0 if pressure_name == "P" else _finite_float(source["P0"], "source.P0")
        )
        self.buffer.Ip = (
            _finite_float(targets["Ip"], "targets.Ip")
            if "Ip" in expected_targets
            else np.nan
        )
        self.buffer.beta = (
            _finite_float(targets["beta"], "targets.beta")
            if "beta" in expected_targets
            else np.nan
        )
        self.buffer.has_x0 = False
        self.buffer.clear_unused_source_tail()
        return count


def _require_exact_dict(value: object, name: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be exactly dict, got {type(value).__name__}")


def _require_exact_keys(value: dict, expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"unexpected {extra}")
    raise ValueError(f"{name} keys do not match the contract: {', '.join(details)}")


def _source_profile(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    profile = np.asarray(value, dtype=np.float64)
    if profile.ndim != 1 or profile.shape != shape:
        raise ValueError(f"source.{name} must share coordinate shape {shape}, got {profile.shape}")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"source.{name} must contain only finite values")
    return profile


def _copy_resized(destination: np.ndarray, source: object, name: str) -> None:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if values.size > destination.size:
        raise ValueError(f"{name} has {values.size} entries but topology stores {destination.size}")
    destination.fill(0.0)
    destination[: values.size] = values


def _canonical_normalized_nodes(value: object, *, name: str) -> np.ndarray:
    """Validate one shared source grid and snap accepted endpoints to 0/1."""

    nodes = np.array(value, dtype=np.float64, copy=True)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)):
        raise ValueError(f"source.{name} must contain at least two finite values")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError(f"source.{name} must be strictly increasing")
    if abs(float(nodes[0])) > 1.0e-12 or abs(float(nodes[-1]) - 1.0) > 1.0e-12:
        raise ValueError(f"source.{name} must cover the normalized [0, 1] domain")
    nodes[0] = 0.0
    nodes[-1] = 1.0
    return nodes


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = ["VEQAdapter"]
