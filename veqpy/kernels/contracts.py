"""Stable four-buffer Kernel ABI used by the VEQPy 2.x runtime.

The public numerical boundary deliberately contains only topology, input,
configuration, and output data.  Backend recipes, source lowering objects,
and solver snapshots stay private to the implementation modules.  ``Input``
and ``Output`` are mutable owners of long-lived NumPy buffers; the topology
and configuration are immutable values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .abi.options import (
    CONTINUE_POLICY_CODES,
    INITIAL_POLICY_CODES,
    RESIDUAL_NORMALIZATION_CODES,
    SOLVER_METHOD_CODES,
)
from .types import KernelTopology as _KernelTopology
from .types import _BackendConfig

# The canonical topology dataclass lives in ``types`` with an internal-only
# constructor contract.  This alias is the single named ABI binding; keeping
# one class avoids a second topology record with subtly different fields.
KernelTopology = _KernelTopology


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Numeric execution policy independent of topology or backend choice."""

    method_code: int = SOLVER_METHOD_CODES["powell"]
    max_residual: float = 1.0e-6
    max_evaluations: int | None = None
    accepted_residual_factor: float = 10.0
    accepted_residual_floor: float = 1.0e-5
    initial_code: int = INITIAL_POLICY_CODES["cold"]
    continuation_code: int = CONTINUE_POLICY_CODES["warm"]
    norm_code: int = RESIDUAL_NORMALIZATION_CODES["fast"]
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e6
    residual_normalization_huber_tau: float = 3.0
    residual_normalization_probe_count: int = 4
    residual_normalization_probe_step: float = 1.0e-6
    residual_normalization_sensitivity_lambda: float = 0.5

    def __post_init__(self) -> None:
        _validate_code(self.method_code, SOLVER_METHOD_CODES, "method_code")
        _validate_code(self.initial_code, INITIAL_POLICY_CODES, "initial_code")
        _validate_code(self.continuation_code, CONTINUE_POLICY_CODES, "continuation_code")
        _validate_code(self.norm_code, RESIDUAL_NORMALIZATION_CODES, "norm_code")
        if self.max_evaluations is not None:
            if type(self.max_evaluations) is not int or self.max_evaluations < 0:
                raise ValueError("max_evaluations must be a non-negative int or None")
        if type(self.residual_normalization_probe_count) is not int:
            raise ValueError("residual_normalization_probe_count must be an int")
        if self.residual_normalization_probe_count < 0:
            raise ValueError("residual_normalization_probe_count must be non-negative")
        for name in (
            "max_residual",
            "accepted_residual_factor",
            "accepted_residual_floor",
            "residual_normalization_floor",
            "residual_normalization_max_ratio",
            "residual_normalization_huber_tau",
            "residual_normalization_probe_step",
            "residual_normalization_sensitivity_lambda",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


def lower_config(value: KernelConfig) -> _BackendConfig:
    """Lower numeric public configuration to the private backend policy."""

    return _BackendConfig(
        method=_inverse_code(SOLVER_METHOD_CODES, value.method_code, "method_code"),
        max_residual=value.max_residual,
        max_evaluations=value.max_evaluations,
        accepted_residual_factor=value.accepted_residual_factor,
        accepted_residual_floor=value.accepted_residual_floor,
        initial=_inverse_code(INITIAL_POLICY_CODES, value.initial_code, "initial_code"),
        continuation=_inverse_code(CONTINUE_POLICY_CODES, value.continuation_code, "continuation_code"),
        norm=_inverse_code(RESIDUAL_NORMALIZATION_CODES, value.norm_code, "norm_code"),
        residual_normalization_floor=value.residual_normalization_floor,
        residual_normalization_max_ratio=value.residual_normalization_max_ratio,
        residual_normalization_huber_tau=value.residual_normalization_huber_tau,
        residual_normalization_probe_count=value.residual_normalization_probe_count,
        residual_normalization_probe_step=value.residual_normalization_probe_step,
        residual_normalization_sensitivity_lambda=value.residual_normalization_sensitivity_lambda,
    )


@dataclass(slots=True)
class KernelInput:
    """Preallocated, numeric-only per-case input buffers.

    The source arrays use the first ``source_count`` entries.  They start at
    capacity 256 and are replaced only by the Adapter's preprocess growth
    epochs.  ``pressure_code`` selects the primitive-versus-derivative
    representation; no strings or Python case objects enter the hot ABI.
    """

    a: float
    R0: float
    Z0: float
    B0: float
    kappa_lcfs: float
    c_lcfs: np.ndarray
    s_lcfs: np.ndarray
    pressure: np.ndarray
    driver: np.ndarray
    source_nodes: np.ndarray
    source_count: int
    p0: float = 0.0
    Ip: float = np.nan
    beta: float = np.nan
    x0: np.ndarray | None = None
    has_x0: bool = False
    pressure_code: int = 1
    capacity_epoch: int = 0

    def __post_init__(self) -> None:
        _pressure_kind_code(self.pressure_code)
        self.a = _finite_float(self.a, "a")
        self.R0 = _finite_float(self.R0, "R0")
        self.Z0 = _finite_float(self.Z0, "Z0")
        self.B0 = _finite_float(self.B0, "B0")
        self.kappa_lcfs = _finite_float(self.kappa_lcfs, "kappa_lcfs")
        if self.a <= 0.0 or self.R0 <= 0.0 or self.kappa_lcfs <= 0.0:
            raise ValueError("a, R0, and kappa_lcfs must be positive")
        self.c_lcfs = _owned_float_array(self.c_lcfs, "c_lcfs")
        self.s_lcfs = _owned_float_array(self.s_lcfs, "s_lcfs")
        self.pressure = _owned_float_array(self.pressure, "pressure")
        self.driver = _owned_float_array(self.driver, "driver")
        self.source_nodes = _owned_float_array(self.source_nodes, "source_nodes")
        if self.pressure.shape != self.driver.shape:
            raise ValueError("pressure and driver must have identical shapes")
        if self.source_nodes.shape != self.pressure.shape:
            raise ValueError("source_nodes and pressure must have identical shapes")
        if type(self.source_count) is not int or self.source_count < 0:
            raise ValueError("source_count must be a non-negative int")
        if self.source_count > self.pressure.size:
            raise ValueError("source_count exceeds the allocated source capacity")
        if type(self.capacity_epoch) is not int or self.capacity_epoch < 0:
            raise ValueError("capacity_epoch must be a non-negative int")
        self.p0 = _finite_float(self.p0, "p0")
        self.Ip = float(self.Ip)
        self.beta = float(self.beta)
        if self.x0 is not None:
            self.x0 = _owned_float_array(self.x0, "x0")
        self.has_x0 = bool(self.has_x0)
        if self.has_x0 and self.x0 is None:
            raise ValueError("has_x0=True requires x0")

    @classmethod
    def allocate(cls, topology: KernelTopology) -> "KernelInput":
        """Allocate the initial 256-entry source capacity epoch."""

        capacity = 256
        source_nodes = np.zeros(capacity, dtype=np.float64)
        return cls(
            a=1.0,
            R0=1.0,
            Z0=0.0,
            B0=1.0,
            kappa_lcfs=1.0,
            c_lcfs=np.zeros(max(1, int(topology.M_max) + 1), dtype=np.float64),
            s_lcfs=np.zeros(max(0, int(topology.M_max)), dtype=np.float64),
            pressure=np.zeros(capacity, dtype=np.float64),
            driver=np.zeros(capacity, dtype=np.float64),
            source_nodes=source_nodes,
            source_count=0,
            x0=np.zeros(int(topology.x_size), dtype=np.float64),
            has_x0=False,
        )

    @property
    def source_capacity(self) -> int:
        """Current resident source capacity, independent of KernelTopology."""

        return int(self.source_nodes.size)

    def grow_source_capacity(self, capacity: int) -> None:
        """Replace source arrays with one larger power-of-two capacity epoch."""

        capacity = int(capacity)
        if capacity <= self.source_capacity:
            return
        if capacity > 1024:
            raise ValueError("VEQ source capacity is limited to 1024 nodes")
        old_count = int(self.source_count)
        for name in (
            "pressure",
            "driver",
            "source_nodes",
        ):
            old = getattr(self, name)
            if old is None:
                continue
            expanded = np.zeros(capacity, dtype=np.float64)
            expanded[:old_count] = old[:old_count]
            setattr(self, name, expanded)
        self.capacity_epoch += 1

    def clear_unused_source_tail(self) -> None:
        """Deterministically zero the unused source capacity suffix."""

        self.pressure[self.source_count :] = 0.0
        self.driver[self.source_count :] = 0.0
        self.source_nodes[self.source_count :] = 0.0


@dataclass(slots=True)
class KernelOutput:
    """Identity-stable output buffers overwritten by every Kernel solve."""

    success: bool = False
    info: int = 0
    nfev: int = 0
    njev: int = 0
    callbacks: int = 0
    jacobian_component_evaluations: int = 0
    jvp_evaluations: int = 0
    linear_iterations: int = 0
    raw_norm: float = np.nan
    scaled_norm: float = np.nan
    elapsed_ms: float = 0.0
    preprocess_ms: float = 0.0
    solve_ms: float = 0.0
    postprocess_ms: float = 0.0
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    raw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    scaled: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    alpha: np.ndarray = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    psin: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    psin_r: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    psin_rr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    FF_psi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    P_psi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @classmethod
    def allocate(cls, topology: KernelTopology) -> "KernelOutput":
        """Allocate all fixed-shape solver and materializer arrays once."""

        nr = int(topology.Nr)
        size = int(topology.x_size)
        return cls(
            x=np.zeros(size, dtype=np.float64),
            raw=np.zeros(size, dtype=np.float64),
            scaled=np.zeros(size, dtype=np.float64),
            alpha=np.zeros(2, dtype=np.float64),
            psin=np.zeros(nr, dtype=np.float64),
            psin_r=np.zeros(nr, dtype=np.float64),
            psin_rr=np.zeros(nr, dtype=np.float64),
            FF_psi=np.zeros(nr, dtype=np.float64),
            P_psi=np.zeros(nr, dtype=np.float64),
        )

    def reset(self) -> None:
        """Reset scalar diagnostics without changing any owned array identity."""

        self.success = False
        self.info = 0
        self.nfev = 0
        self.njev = 0
        self.callbacks = 0
        self.jacobian_component_evaluations = 0
        self.jvp_evaluations = 0
        self.linear_iterations = 0
        self.raw_norm = np.nan
        self.scaled_norm = np.nan
        self.elapsed_ms = 0.0
        self.preprocess_ms = 0.0
        self.solve_ms = 0.0
        self.postprocess_ms = 0.0


def _pressure_kind_code(value: int) -> int:
    if type(value) is int:
        if value in (0, 1):
            return int(value)
        raise ValueError("pressure_code must be 0 (primitive) or 1 (derivative)")
    raise ValueError("pressure_code must be an int")


def _validate_code(value: int, table: dict[str, int], name: str) -> None:
    if type(value) is not int or value not in table.values():
        choices = ", ".join(str(code) for code in sorted(table.values()))
        raise ValueError(f"{name} must be one of {choices}")


def _inverse_code(table: dict[str, int], value: int, name: str) -> str:
    _validate_code(value, table, name)
    return next(token for token, code in table.items() if code == value)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _owned_float_array(value: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


__all__ = ["KernelConfig", "KernelInput", "KernelOutput", "KernelTopology"]
