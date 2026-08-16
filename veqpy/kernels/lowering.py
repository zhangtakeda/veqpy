"""Private lowering from the four public ABI buffers to backend case views.

The objects returned here are implementation details.  They are rebuilt only
when a bound ``KernelInput`` changes and are never returned from the public
Kernel methods.
"""

from __future__ import annotations

import numpy as np

from .contracts import KernelConfig, KernelInput, KernelTopology, lower_config
from .types import _BoundaryCase, _BuildPolicy, _SourceCase


def boundary_case(topology: KernelTopology, value: KernelInput) -> _BoundaryCase:
    """Create the private coefficient view consumed by numerical kernels."""

    c_count = max(1, int(topology.M_max) + 1)
    s_count = max(0, int(topology.M_max))
    c_offsets = np.zeros(c_count, dtype=np.float64)
    s_offsets = np.zeros(s_count, dtype=np.float64)
    c_take = min(c_count, value.c_lcfs.size)
    s_take = min(s_count, value.s_lcfs.size)
    c_offsets[:c_take] = value.c_lcfs[:c_take]
    s_offsets[:s_take] = value.s_lcfs[:s_take]
    return _BoundaryCase(
        a=value.a,
        R0=value.R0,
        Z0=value.Z0,
        B0=value.B0,
        ka=value.kappa_lcfs,
        c_offsets=c_offsets,
        s_offsets=s_offsets,
    )


def source_case(topology: KernelTopology, value: KernelInput) -> _SourceCase:
    """Create a private source view using only the active capacity prefix."""

    stop = int(value.source_count)
    pressure = np.asarray(value.pressure[:stop], dtype=np.float64)
    derivative = (
        None
        if value.pressure_derivative is None
        else np.asarray(value.pressure_derivative[:stop], dtype=np.float64)
    )
    coordinate = topology.coordinate
    pressure_name = {
        "r": "P_r",
        "psin": "P_psin",
        "rho": "P_rho",
    }[coordinate]
    driver_name = _driver_name(topology.route, coordinate)
    kwargs: dict[str, object] = {"Ip": value.Ip, "beta": value.beta}
    kwargs["source_nodes"] = np.asarray(value.source_nodes[:stop], dtype=np.float64)
    if value.pressure_code == 0:
        kwargs["p"] = pressure
    else:
        kwargs[pressure_name] = pressure if derivative is None else derivative
        kwargs["p0"] = value.p0
    kwargs[driver_name] = np.asarray(value.driver[:stop], dtype=np.float64)
    return _SourceCase(**kwargs)


def build_policy(*, backend: str, layout: str = "degree") -> _BuildPolicy:
    """Create the private backend recipe from a normalized public token."""

    normalized = str(backend).strip().lower()
    if normalized == "cxx":
        normalized = "cxx-relaxed"
    if normalized == "numba":
        return _BuildPolicy(backend="numba", layout=layout, build="numba")
    if normalized == "cxx-strict":
        return _BuildPolicy(backend="cxx-strict", layout=layout, build="release-strict")
    if normalized == "cxx-relaxed":
        return _BuildPolicy(backend="cxx-relaxed", layout=layout, build="release-relaxed")
    raise ValueError("backend must be numba, cxx, cxx-strict, or cxx-relaxed")


def private_config(value: KernelConfig) -> object:
    """Return the backend-only string policy for a numeric public config."""

    return lower_config(value)


def _driver_name(route: str, coordinate: str) -> str:
    if str(route).upper() == "PF":
        return {"r": "FF_r", "psin": "FF_psin", "rho": "FF_rho"}[str(coordinate).lower()]
    return {
        "PP": "psi_r",
        "PI": "itor",
        "PJ1": "jtor",
        "PJ2": "jpara",
        "PJ3": "jtotal",
        "PQ": "q",
    }[str(route).upper()]


__all__ = ["boundary_case", "build_policy", "private_config", "source_case"]
