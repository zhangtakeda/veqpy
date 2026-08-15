"""VEQ Adapter: frozen Plasma values to a stable KernelInput."""

from __future__ import annotations

import numpy as np
from fusionprime_base import Plasma

from .kernels.contracts import KernelInput, KernelTopology


class VEQAdapter:
    """Copy and remap one Plasma equilibrium into a prepared input buffer."""

    def __init__(self, topology: KernelTopology, buffer: KernelInput) -> None:
        self.topology = topology
        self.buffer = buffer

    def fill(self, plasma: Plasma) -> int:
        """Validate the frozen context and overwrite the bound input buffers."""

        equilibrium = plasma.equilibrium
        geometry = equilibrium.geometry
        coordinate_values = _coordinate_values(equilibrium, self.topology.coordinate)
        pressure_values = _pressure_values(equilibrium, self.topology.coordinate)
        driver_values = _driver_values(plasma, self.topology.route, self.topology.coordinate)

        # Source coordinates are always a runtime explicit input.  The
        # equilibrium grid is the default source representation, but the
        # KernelInput owner may receive any strictly increasing source grid in
        # the same normalized coordinate domain.
        source_nodes = np.asarray(coordinate_values[0], dtype=np.float64)
        pressure = np.asarray(pressure_values[1], dtype=np.float64)
        driver = _remap(driver_values[0], driver_values[1], source_nodes)
        native_source_nodes, source_coordinate_jacobian = _native_source_mapping(
            equilibrium,
            self.topology.coordinate,
            source_nodes,
        )
        count = int(source_nodes.size)
        if count > 1024:
            raise ValueError("VEQ source input has more than the supported 1024 nodes")
        if count < 2 or not np.all(np.isfinite(source_nodes)):
            raise ValueError("source nodes must contain at least two finite values")
        if np.any(np.diff(source_nodes) <= 0.0):
            raise ValueError("source nodes must be strictly increasing")
        if source_nodes[0] < -1.0e-12 or source_nodes[-1] > 1.0 + 1.0e-12:
            raise ValueError("source nodes must cover the normalized [0, 1] domain")

        capacity = 256
        while capacity < count:
            capacity *= 2
        if capacity > self.buffer.source_capacity:
            self.buffer.grow_source_capacity(capacity)

        self.buffer.a = float(geometry.a)
        self.buffer.R0 = float(geometry.R0)
        self.buffer.Z0 = float(geometry.Z0)
        self.buffer.B0 = float(equilibrium.B0)
        self.buffer.kappa_lcfs = float(geometry.kappa_lcfs)
        _copy_resized(self.buffer.c_lcfs, geometry.c_lcfs, "c_lcfs")
        _copy_resized(self.buffer.s_lcfs, geometry.s_lcfs, "s_lcfs")
        self.buffer.source_count = count
        self.buffer.source_nodes[:count] = source_nodes
        self.buffer.native_source_nodes[:count] = native_source_nodes
        self.buffer.source_coordinate_jacobian[:count] = source_coordinate_jacobian
        self.buffer.pressure[:count] = pressure
        self.buffer.driver[:count] = driver
        self.buffer.p0 = float(equilibrium.P0)
        if self.topology.constraint in {"ip", "both"}:
            self.buffer.Ip = _finite_or_default(equilibrium.Ip, 1.0e6)
        else:
            self.buffer.Ip = np.nan
        if self.topology.constraint in {"beta", "both"}:
            self.buffer.beta = _finite_or_default(equilibrium.betat, 0.01)
        else:
            self.buffer.beta = np.nan
        self.buffer.has_x0 = False
        self.buffer.clear_unused_source_tail()
        return count


def _coordinate_values(equilibrium: object, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    key = str(coordinate).lower()
    if key == "r":
        return np.asarray(equilibrium.r, dtype=np.float64), np.asarray(equilibrium.r, dtype=np.float64)
    if key == "psin":
        return np.asarray(equilibrium.psin, dtype=np.float64), np.asarray(equilibrium.psin, dtype=np.float64)
    if key == "rho":
        return np.asarray(equilibrium.rho, dtype=np.float64), np.asarray(equilibrium.rho, dtype=np.float64)
    raise ValueError(f"unsupported source coordinate {coordinate!r}")


def _pressure_values(equilibrium: object, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    key = str(coordinate).lower()
    if key == "r":
        values = equilibrium.P_r
    elif key == "psin":
        values = equilibrium.P_psin
    elif key == "rho":
        values = np.divide(equilibrium.P_r, equilibrium.rho_r)
    else:
        raise ValueError(f"unsupported source coordinate {coordinate!r}")
    return _coordinate_values(equilibrium, coordinate)[0], np.asarray(values, dtype=np.float64)


def _driver_values(plasma: Plasma, route: str, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    equilibrium = plasma.equilibrium
    current = plasma.current
    route_key = str(route).upper()
    coordinate_key = str(coordinate).lower()
    if route_key == "PP":
        values = equilibrium.psi_r
        nodes = equilibrium.rho if coordinate_key == "rho" else equilibrium.r
    elif route_key == "PI":
        values = equilibrium.Itor
        nodes = equilibrium.rho if coordinate_key == "rho" else equilibrium.r
    elif route_key == "PJ1":
        values = equilibrium.jtor
        nodes = equilibrium.rho if coordinate_key == "rho" else equilibrium.r
    elif route_key == "PJ2":
        values = equilibrium.jpara
        nodes = equilibrium.rho if coordinate_key == "rho" else equilibrium.r
    elif route_key == "PJ3":
        values = equilibrium.jtotal
        nodes = equilibrium.rho if coordinate_key == "rho" else equilibrium.r
    elif route_key == "PQ":
        current_q = np.asarray(current.q, dtype=np.float64)
        if np.all(np.isfinite(current_q)):
            values = current_q
            nodes = np.asarray(current.rho, dtype=np.float64)
        else:
            values = np.asarray(equilibrium.q, dtype=np.float64)
            nodes = np.asarray(
                equilibrium.rho if coordinate_key == "rho" else equilibrium.r,
                dtype=np.float64,
            )
    else:
        if str(route_key) != "PF":
            raise ValueError(f"unsupported source route {route!r}")
        if coordinate_key == "r":
            values = equilibrium.FF_r
            nodes = equilibrium.r
        elif coordinate_key == "psin":
            values = equilibrium.FF_psin
            nodes = equilibrium.psin
        elif coordinate_key == "rho":
            values = np.divide(equilibrium.FF_r, equilibrium.rho_r)
            nodes = equilibrium.rho
        else:
            raise ValueError(f"unsupported source coordinate {coordinate!r}")
    return np.asarray(nodes, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _remap(nodes: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_nodes = np.asarray(nodes, dtype=np.float64)
    source_values = np.asarray(values, dtype=np.float64)
    if source_nodes.ndim != 1 or source_values.shape != source_nodes.shape:
        raise ValueError("source coordinate and profile must share one-dimensional shape")
    if not np.all(np.isfinite(source_nodes)) or not np.all(np.isfinite(source_values)):
        raise ValueError("source coordinate and profile must be finite")
    if np.any(np.diff(source_nodes) <= 0.0):
        raise ValueError("source coordinate must be strictly increasing")
    if source_nodes[0] > target[0] + 1.0e-12 or source_nodes[-1] < target[-1] - 1.0e-12:
        raise ValueError("source profile does not cover the target normalized domain")
    return np.interp(target, source_nodes, source_values).astype(np.float64, copy=False)


def _copy_resized(destination: np.ndarray, source: np.ndarray, name: str) -> None:
    source = np.asarray(source, dtype=np.float64)
    if source.size > destination.size:
        raise ValueError(f"{name} has {source.size} entries but topology stores {destination.size}")
    destination.fill(0.0)
    destination[: source.size] = source


def _finite_or_default(value: float, default: float) -> float:
    result = float(value)
    return result if np.isfinite(result) else float(default)


def _native_source_mapping(
    equilibrium: object,
    coordinate: str,
    source_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-r nodes and d(coordinate)/dr for backend lowering."""

    if str(coordinate).lower() != "rho":
        return source_nodes.copy(), np.ones_like(source_nodes)
    native_nodes = np.asarray(equilibrium.r, dtype=np.float64)
    jacobian = np.asarray(equilibrium.rho_r, dtype=np.float64)
    if native_nodes.shape != source_nodes.shape or jacobian.shape != source_nodes.shape:
        raise ValueError("rho source mapping must match the equilibrium source grid")
    return native_nodes, jacobian


__all__ = ["VEQAdapter"]
