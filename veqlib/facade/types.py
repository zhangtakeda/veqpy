"""Typed VEQlib facade values lowered to the native kernel ABI.

The dataclasses in this module canonicalize topology/build/runtime inputs into
stable Python objects before ``kernel.py`` lowers them to C++ scalars, enum
codes, and 1D C-contiguous ``float64`` arrays. They are VEQlib contracts, not
external operator/source-plan adapters.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, fields, replace
from typing import Any, Self

import numpy as np

from .options import (
    continue_policy_code,
    initial_policy_code,
    residual_normalization_code,
    solver_method_code,
)

_KERNEL_TOPOLOGY_KEY_LENGTH = 32
# These integer wire values mirror the generated C++ ABI enum contract.
# Changing them is a cross-language compatibility change, not a Python refactor.
_SOURCE_ROUTE_CODES = {
    "PF": 1,
    "PP": 2,
    "PI": 3,
    "PJ1": 4,
    "PJ2": 5,
    "PQ": 6,
}
_SOURCE_COORDINATE_CODES = {"rho": 1, "psin": 2}
_SOURCE_CONSTRAINT_CODES = {"null": 0, "Ip": 1, "beta": 2, "Ip_beta": 3}
_SOURCE_NODES_CODES = {"uniform": 1, "grid": 2}
_SOURCE_ACTIVE_FAMILY_CODES = {"none": 0, "psin": 1, "F": 2}
_SOURCE_PARAMETERIZATION_CODES = {"identity": 0, "sqrt_psin": 1}
_SOURCE_CONSTRAINTS_BY_ROUTE = {
    "PF": frozenset({"null", "Ip", "beta"}),
    "PP": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PI": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PJ1": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PJ2": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PQ": frozenset({"null", "Ip", "beta", "Ip_beta"}),
}
_VEQLIB_NATIVE_ROUTE_CONSTRAINTS = {
    (route, coordinate, nodes): constraints
    for route, constraints in _SOURCE_CONSTRAINTS_BY_ROUTE.items()
    for coordinate in ("rho", "psin")
    for nodes in ("uniform", "grid")
}
_LAYOUT_CODES = {"degree": 0, "family": 1}
_BUILD_PRESET_KWARGS: dict[str, dict[str, object]] = {
    "fastmath": {
        "cmake_build_type": "Release",
        "fp_mode": "RELAXED",
        "enable_enzyme": False,
        "enable_native_optimizations": True,
        "enable_thin_lto": True,
        "analysis": False,
    },
    "fastmath-enzyme": {
        "cmake_build_type": "Release",
        "fp_mode": "RELAXED",
        "enable_enzyme": True,
        "enable_native_optimizations": True,
        "enable_thin_lto": True,
        "analysis": False,
    },
    "release": {
        "cmake_build_type": "Release",
        "fp_mode": "STRICT",
        "enable_enzyme": False,
        "enable_native_optimizations": True,
        "enable_thin_lto": True,
        "analysis": False,
    },
    "debug": {
        "cmake_build_type": "Debug",
        "fp_mode": "STRICT",
        "enable_enzyme": False,
        "enable_native_optimizations": False,
        "enable_thin_lto": False,
        "analysis": False,
    },
}
_DEFAULT_ENZYME_JACOBIAN_BATCH_WIDTH = 0

class TopologyError(ValueError):
    """Raised when a VEQlib kernel topology cannot be canonicalized."""


@dataclass(frozen=True, slots=True)
class KernelRecipe:
    """Compile recipe and packed-layout configuration for one VEQlib kernel."""

    layout: str = "degree"
    build: str = "fastmath"
    cmake_build_type: str | None = None
    fp_mode: str | None = None
    enable_enzyme: bool | None = None
    enable_native_optimizations: bool | None = None
    enable_thin_lto: bool | None = None
    analysis: bool | None = None
    enzyme_jacobian_batch_width: int | None = None

    def __post_init__(self) -> None:
        for name, value in self.canonical_kwargs().items():
            object.__setattr__(self, name, value)

    def canonical_kwargs(self) -> dict[str, object]:
        build = _normalize_token(self.build, "build")
        if build not in _BUILD_PRESET_KWARGS:
            raise TopologyError("build must be one of fastmath, fastmath-enzyme, release, or debug")
        preset = _BUILD_PRESET_KWARGS[build]
        return {
            "layout": _normalize_layout(self.layout),
            "build": build,
            "cmake_build_type": _normalize_cmake_build_type(
                self.cmake_build_type,
                default=str(preset["cmake_build_type"]),
            ),
            "fp_mode": _normalize_fp_mode(self.fp_mode, default=str(preset["fp_mode"])),
            "enable_enzyme": _canonical_bool(
                self.enable_enzyme,
                default=bool(preset["enable_enzyme"]),
                name="enable_enzyme",
            ),
            "enable_native_optimizations": _canonical_bool(
                self.enable_native_optimizations,
                default=bool(preset["enable_native_optimizations"]),
                name="enable_native_optimizations",
            ),
            "enable_thin_lto": _canonical_bool(
                self.enable_thin_lto,
                default=bool(preset["enable_thin_lto"]),
                name="enable_thin_lto",
            ),
            "analysis": _canonical_bool(
                self.analysis,
                default=bool(preset["analysis"]),
                name="analysis",
            ),
            "enzyme_jacobian_batch_width": _canonical_enzyme_jacobian_batch_width(
                self.enzyme_jacobian_batch_width
            ),
        }

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "preset": self.build,
            "layout": {
                "packed": self.layout,
                "profile_first": self.layout_profile_first,
                "code": self.layout_code,
            },
            "cmake_build_type": self.cmake_build_type,
            "fp_mode": self.fp_mode,
            "enable_enzyme": self.enable_enzyme,
            "enable_native_optimizations": self.enable_native_optimizations,
            "enable_thin_lto": self.enable_thin_lto,
            "analysis": self.analysis,
            "enzyme_jacobian_batch_width": self.enzyme_jacobian_batch_width,
        }

    @property
    def layout_code(self) -> int:
        return _LAYOUT_CODES[self.layout]

    @property
    def layout_profile_first(self) -> bool:
        return self.layout == "family"


@dataclass(frozen=True, slots=True)
class KernelBoundary:
    """Native VEQlib boundary scalars and Fourier offsets."""

    a: float
    R0: float
    Z0: float
    B0: float
    ka: float = 1.0
    c_offsets: np.ndarray | tuple[float, ...] | list[float] | None = None
    s_offsets: np.ndarray | tuple[float, ...] | list[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "R0", float(self.R0))
        object.__setattr__(self, "Z0", float(self.Z0))
        object.__setattr__(self, "B0", float(self.B0))
        object.__setattr__(self, "ka", float(self.ka))
        object.__setattr__(self, "c_offsets", _readonly_1d_or_default(self.c_offsets, "c_offsets"))
        s_offsets = _readonly_1d_or_default(self.s_offsets, "s_offsets")
        s_copy = np.array(s_offsets, dtype=np.float64, copy=True)
        s_copy[0] = 0.0
        s_copy.setflags(write=False)
        object.__setattr__(self, "s_offsets", s_copy)

    def runtime_args(self) -> tuple[Any, ...]:
        return (
            self.a,
            self.R0,
            self.Z0,
            self.B0,
            self.ka,
            np.ascontiguousarray(self.c_offsets, dtype=np.float64),
            np.ascontiguousarray(self.s_offsets, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class KernelTopology:
    """Native VEQlib compile-time topology independent from artifact recipe."""

    h_count: int
    v_count: int
    kappa_count: int
    psin_count: int
    F_count: int
    c_counts: tuple[int, ...]
    s_counts: tuple[int, ...]
    Nr: int
    Nt: int
    route: str
    coordinate: str
    constraint: str
    nodes: str
    sample_count: int | None = None
    quadrature: str = "legendre"
    calculus: str = "spectral"
    L_max: int | None = None
    M_max: int | None = None
    K_max: int | None = None
    key: str | None = None

    def __post_init__(self) -> None:
        profile_counts = {
            "h_count": _nonnegative_int(self.h_count, "h_count"),
            "v_count": _nonnegative_int(self.v_count, "v_count"),
            "kappa_count": _nonnegative_int(self.kappa_count, "kappa_count"),
            "psin_count": _nonnegative_int(self.psin_count, "psin_count"),
            "F_count": _nonnegative_int(self.F_count, "F_count"),
        }
        c_counts = _trim_trailing_zeros(_nonnegative_int_tuple(self.c_counts, "c_counts"))
        s_counts = _trim_trailing_zeros(_nonnegative_int_tuple(self.s_counts, "s_counts"))
        nr = _positive_int(self.Nr, "Nr")
        nt = _positive_int(self.Nt, "Nt")
        if nr < 4:
            raise TopologyError("Nr must be at least 4")
        if nt < 4:
            raise TopologyError("Nt must be at least 4")

        route = _normalize_token(self.route, "route").upper()
        if route not in _SOURCE_ROUTE_CODES:
            raise TopologyError(f"unsupported route {route!r}")
        coordinate = _normalize_token(self.coordinate, "coordinate").lower()
        if coordinate not in _SOURCE_COORDINATE_CODES:
            raise TopologyError(f"unsupported coordinate {coordinate!r}")
        nodes = _normalize_token(self.nodes, "nodes").lower()
        if nodes not in _SOURCE_NODES_CODES:
            raise TopologyError(f"unsupported source nodes {nodes!r}")
        constraint = _normalize_constraint(self.constraint)
        _validate_source_constraint(route, constraint)
        quadrature = _normalize_token(self.quadrature, "quadrature").lower()
        if quadrature != "legendre":
            raise TopologyError("only legendre quadrature is supported")
        calculus = _normalize_token(self.calculus, "calculus").lower()
        if calculus != "spectral":
            raise TopologyError("only spectral calculus is supported")
        sample_count = _canonical_sample_count(nodes, nr, self.sample_count)
        l_max = _canonical_exact_or_inferred(
            self.L_max,
            _infer_l_max((*profile_counts.values(), *c_counts, *s_counts)),
            "L_max",
        )
        m_max = _canonical_at_least(self.M_max, _infer_m_max(c_counts, s_counts), "M_max")
        k_max = _canonical_at_least(self.K_max, max(2, m_max), "K_max")
        source_active_family = _source_active_family(route, coordinate, nodes)
        _validate_source_active_family(
            source_active_family,
            psin_count=profile_counts["psin_count"],
            f_count=profile_counts["F_count"],
        )
        normalized_values: dict[str, Any] = {
            **profile_counts,
            "c_counts": c_counts,
            "s_counts": s_counts,
            "Nr": nr,
            "Nt": nt,
            "route": route,
            "coordinate": coordinate,
            "constraint": constraint,
            "nodes": nodes,
            "sample_count": sample_count,
            "quadrature": quadrature,
            "calculus": calculus,
            "L_max": l_max,
            "M_max": m_max,
            "K_max": k_max,
        }
        for name, value in normalized_values.items():
            object.__setattr__(self, name, value)
        expected_key = self.compute_key()
        if self.key is not None and self.key != expected_key:
            raise TopologyError(
                "key does not match canonical kernel topology: "
                f"got {self.key!r}, expected {expected_key!r}"
            )
        object.__setattr__(self, "key", expected_key)

    def active_profiles(self) -> dict[str, int]:
        active: dict[str, int] = {}
        if self.h_count > 0:
            active["h"] = self.h_count
        if self.v_count > 0:
            active["v"] = self.v_count
        if self.kappa_count > 0:
            active["k"] = self.kappa_count
        for order, count in enumerate(self.c_counts):
            if count > 0:
                active[f"c{order}"] = count
        for order, count in enumerate(self.s_counts, start=1):
            if count > 0:
                active[f"s{order}"] = count
        if self.psin_count > 0:
            active["psin"] = self.psin_count
        if self.F_count > 0:
            active["F"] = self.F_count
        return active

    def packed_size(self) -> int:
        return sum(self.active_profiles().values())

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "profiles": {
                "h_count": self.h_count,
                "v_count": self.v_count,
                "kappa_count": self.kappa_count,
                "psin_count": self.psin_count,
                "F_count": self.F_count,
                "c_counts": list(self.c_counts),
                "s_counts": list(self.s_counts),
                "L_max": self.L_max,
                "M_max": self.M_max,
                "K_max": self.K_max,
            },
            "grid": {
                "Nr": self.Nr,
                "Nt": self.Nt,
                "quadrature": self.quadrature,
                "calculus": self.calculus,
            },
            "source": self.source_policy_dict(),
        }

    def source_policy_dict(self) -> dict[str, Any]:
        return {
            "route_key": list(self.source_route_key),
            "route": self.route,
            "route_code": self.source_route_code,
            "coordinate": self.coordinate,
            "coordinate_code": self.source_coordinate_code,
            "constraint": self.constraint,
            "constraint_code": self.source_constraint_code,
            "supported_constraints": list(self.source_supported_constraints),
            "uses_Ip": self.source_uses_ip_constraint,
            "uses_beta": self.source_uses_beta_constraint,
            "nodes": self.nodes,
            "nodes_code": self.source_nodes_code,
            "sample_count": self.sample_count,
            "active_family": self.source_active_family,
            "active_family_code": self.source_active_family_code,
            "parameterization": self.source_parameterization,
            "parameterization_code": self.source_parameterization_code,
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def compute_key(self) -> str:
        digest = hashlib.sha256(self.to_json_bytes()).digest()
        encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
        return encoded[:_KERNEL_TOPOLOGY_KEY_LENGTH]

    @property
    def source_route_code(self) -> int:
        return _SOURCE_ROUTE_CODES[self.route]

    @property
    def source_route_key(self) -> tuple[str, str, str]:
        return (self.route, self.coordinate, self.nodes)

    @property
    def source_coordinate_code(self) -> int:
        return _SOURCE_COORDINATE_CODES[self.coordinate]

    @property
    def source_constraint_code(self) -> int:
        return _SOURCE_CONSTRAINT_CODES[self.constraint]

    @property
    def source_nodes_code(self) -> int:
        return _SOURCE_NODES_CODES[self.nodes]

    @property
    def source_active_family(self) -> str:
        return _source_active_family(self.route, self.coordinate, self.nodes)

    @property
    def source_active_family_code(self) -> int:
        return _SOURCE_ACTIVE_FAMILY_CODES[self.source_active_family]

    @property
    def source_parameterization(self) -> str:
        return _source_parameterization(self.route, self.coordinate, self.nodes)

    @property
    def source_parameterization_code(self) -> int:
        return _SOURCE_PARAMETERIZATION_CODES[self.source_parameterization]

    @property
    def source_supported_constraints(self) -> tuple[str, ...]:
        ordered = ("Ip_beta", "Ip", "beta", "null")
        supported = _SOURCE_CONSTRAINTS_BY_ROUTE[self.route]
        return tuple(value for value in ordered if value in supported)

    @property
    def source_uses_ip_constraint(self) -> bool:
        return self.constraint in {"Ip", "Ip_beta"}

    @property
    def source_uses_beta_constraint(self) -> bool:
        return self.constraint in {"beta", "Ip_beta"}

    def validate_supported_for_veqlib_native(self) -> None:
        mismatches: list[str] = []
        if self.quadrature != "legendre":
            mismatches.append(f"quadrature={self.quadrature!r}")
        if self.calculus != "spectral":
            mismatches.append(f"calculus={self.calculus!r}")
        supported = _VEQLIB_NATIVE_ROUTE_CONSTRAINTS.get(self.source_route_key)
        if supported is None or self.constraint not in supported:
            mismatches.append(
                f"route_key={self.source_route_key!r}, constraint={self.constraint!r}"
            )
        if self.source_active_family != "F" and self.F_count > 0:
            mismatches.append("F_count > 0 outside PJ2")
        if self.source_active_family == "F" and self.F_count <= 0:
            mismatches.append("PJ2 requires F_count > 0")
        if self.source_active_family != "psin" and self.psin_count > 0:
            mismatches.append("source-owned topology does not accept psin_count > 0")
        if mismatches:
            raise TopologyError("unsupported VEQlib native topology: " + "; ".join(mismatches))


@dataclass(frozen=True, slots=True)
class KernelSource:
    """Runtime source and physical constraints for one VEQlib kernel solve."""

    scaled_heat: np.ndarray | list[float] | tuple[float, ...]
    scaled_current: np.ndarray | list[float] | tuple[float, ...]
    scaled_Ip: float = np.nan
    beta: float = np.nan
    case_name: str | None = None

    def __post_init__(self) -> None:
        heat = _readonly_1d(self.scaled_heat, "scaled_heat")
        current = _readonly_1d(self.scaled_current, "scaled_current")
        if heat.shape != current.shape:
            raise ValueError(
                "scaled_heat and scaled_current must share the same shape, "
                f"got {heat.shape} and {current.shape}"
            )
        object.__setattr__(self, "scaled_heat", heat)
        object.__setattr__(self, "scaled_current", current)
        object.__setattr__(self, "scaled_Ip", float(self.scaled_Ip))
        object.__setattr__(self, "beta", float(self.beta))
        case_name = None if self.case_name is None else str(self.case_name)
        object.__setattr__(self, "case_name", case_name)

    def runtime_args(self) -> tuple[Any, ...]:
        return (
            self.scaled_heat,
            self.scaled_current,
            self.scaled_Ip,
            self.beta,
        )


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Runtime configuration for one VEQlib kernel invocation."""

    method: str | int = "powell"
    max_residual: float = 1.0e-6
    max_evaluations: int | None = None
    accepted_residual_factor: float = 10.0
    accepted_residual_floor: float = 1.0e-5
    initial: str | int = "cold"
    continuation: str | int = "warm"
    norm: str | int = "fast"
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e6
    residual_normalization_huber_tau: float = 3.0
    residual_normalization_probe_count: int = 4
    residual_normalization_probe_step: float = 1.0e-6
    residual_normalization_sensitivity_lambda: float = 0.5

    def with_overrides(self, **overrides: Any) -> Self:
        field_names = {field.name for field in fields(self)}
        unknown = sorted(name for name in overrides if name not in field_names)
        if unknown:
            names = ", ".join(unknown)
            raise TypeError(f"Unsupported KernelConfig override(s): {names}")
        return replace(self, **overrides)

    def runtime_args(self, *, x_size: int) -> tuple[Any, ...]:
        max_evaluations = (
            x_size * x_size if self.max_evaluations is None else int(self.max_evaluations)
        )
        if max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        return (
            solver_method_code(self.method),
            float(self.max_residual),
            max_evaluations,
            float(self.accepted_residual_factor),
            float(self.accepted_residual_floor),
            initial_policy_code(self.initial),
            continue_policy_code(self.continuation),
            residual_normalization_code(self.norm),
            float(self.residual_normalization_floor),
            float(self.residual_normalization_max_ratio),
            float(self.residual_normalization_huber_tau),
            int(self.residual_normalization_probe_count),
            float(self.residual_normalization_probe_step),
            float(self.residual_normalization_sensitivity_lambda),
        )


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Python-owned snapshot of one VEQlib solve result."""

    elapsed_ms: float
    success: bool
    info: int
    nfev: int
    njev: int
    callbacks: int
    jacobian_component_evaluations: int
    jvp_evaluations: int
    linear_iterations: int
    raw_norm: float
    scaled_norm: float
    x: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    alpha: np.ndarray

    @classmethod
    def from_solve_direct(cls, value: Any) -> Self:
        (
            elapsed_ms,
            success,
            info,
            nfev,
            njev,
            callbacks,
            jacobian_component_evaluations,
            jvp_evaluations,
            linear_iterations,
            raw_norm,
            scaled_norm,
            x,
            raw,
            scaled,
            alpha,
        ) = value
        return cls(
            elapsed_ms=float(elapsed_ms),
            success=bool(success),
            info=int(info),
            nfev=int(nfev),
            njev=int(njev),
            callbacks=int(callbacks),
            jacobian_component_evaluations=int(jacobian_component_evaluations),
            jvp_evaluations=int(jvp_evaluations),
            linear_iterations=int(linear_iterations),
            raw_norm=float(raw_norm),
            scaled_norm=float(scaled_norm),
            x=np.array(x, dtype=np.float64, copy=True),
            raw=np.array(raw, dtype=np.float64, copy=True),
            scaled=np.array(scaled, dtype=np.float64, copy=True),
            alpha=np.array(alpha, dtype=np.float64, copy=True),
        )


def _normalize_token(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TopologyError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TopologyError(f"{name} must not be empty")
    return normalized


def _normalize_constraint(value: str | None) -> str:
    if value is None:
        return "null"
    normalized = _normalize_token(value, "constraint").lower().replace("-", "_")
    mapping = {
        "ip": "Ip",
        "beta": "beta",
        "ip_beta": "Ip_beta",
        "ipbeta": "Ip_beta",
        "null": "null",
        "none": "null",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise TopologyError(f"unsupported constraint {value!r}") from exc


def _validate_source_constraint(route: str, constraint: str) -> None:
    supported = _SOURCE_CONSTRAINTS_BY_ROUTE[route]
    if constraint not in supported:
        raise TopologyError(f"{route} source topology does not support constraint {constraint!r}")


def _normalize_layout(value: str) -> str:
    normalized = _normalize_token(value, "layout").lower().replace("-", "_")
    mapping = {
        "degree": "degree",
        "degree_first": "degree",
        "family": "family",
        "family_first": "family",
        "profile": "family",
        "profile_first": "family",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise TopologyError("layout must be degree or family") from exc


def _source_active_family(route: str, coordinate: str, nodes: str) -> str:
    if route == "PJ2":
        return "F"
    if coordinate == "psin" and nodes == "uniform":
        return "psin"
    return "none"


def _source_parameterization(route: str, coordinate: str, nodes: str) -> str:
    if route == "PP" and coordinate == "psin" and nodes == "uniform":
        return "sqrt_psin"
    return "identity"


def _validate_source_active_family(
    source_active_family: str,
    *,
    psin_count: int,
    f_count: int,
) -> None:
    if source_active_family == "psin" and psin_count <= 0:
        raise TopologyError("psin/uniform source topology requires psin_count > 0")
    if source_active_family == "F" and f_count <= 0:
        raise TopologyError("PJ2 source topology requires F_count > 0")


def _normalize_cmake_build_type(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    normalized = _normalize_token(value, "cmake_build_type").lower()
    if normalized == "debug":
        return "Debug"
    if normalized == "release":
        return "Release"
    raise TopologyError("cmake_build_type must be Debug or Release")


def _normalize_fp_mode(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    normalized = _normalize_token(value, "fp_mode").upper()
    if normalized not in {"STRICT", "FMA", "RELAXED"}:
        raise TopologyError("fp_mode must be STRICT, FMA, or RELAXED")
    return normalized


def _canonical_bool(value: bool | None, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TopologyError(f"{name} must be a bool")
    return value


def _canonical_enzyme_jacobian_batch_width(value: int | None) -> int:
    if value is None:
        return _DEFAULT_ENZYME_JACOBIAN_BATCH_WIDTH
    return _nonnegative_int(value, "enzyme_jacobian_batch_width")


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TopologyError(f"{name} must be an integer")
    if value < 0:
        raise TopologyError(f"{name} must be non-negative")
    return value


def _positive_int(value: int, name: str) -> int:
    integer = _nonnegative_int(value, name)
    if integer <= 0:
        raise TopologyError(f"{name} must be positive")
    return integer


def _nonnegative_int_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TopologyError(f"{name} must be a tuple[int, ...]")
    return tuple(_nonnegative_int(value, f"{name}[{index}]") for index, value in enumerate(values))


def _trim_trailing_zeros(values: tuple[int, ...]) -> tuple[int, ...]:
    end = len(values)
    while end > 0 and values[end - 1] == 0:
        end -= 1
    return values[:end]


def _infer_l_max(profile_counts: tuple[int, ...]) -> int:
    highest = max(profile_counts, default=0)
    if highest < 1:
        raise TopologyError("derived L_max requires at least one active profile count")
    return max(1, highest - 1)


def _infer_m_max(c_counts: tuple[int, ...], s_counts: tuple[int, ...]) -> int:
    c_highest = max((order for order, count in enumerate(c_counts) if count > 0), default=0)
    s_highest = max((order + 1 for order, count in enumerate(s_counts) if count > 0), default=0)
    return max(1, c_highest, s_highest)


def _canonical_exact_or_inferred(value: int | None, inferred: int, name: str) -> int:
    if value is None:
        return inferred
    explicit = _positive_int(value, name)
    if explicit != inferred:
        raise TopologyError(
            f"{name} is inferred as {inferred}; explicit value {explicit} is invalid"
        )
    return explicit


def _canonical_at_least(value: int | None, minimum: int, name: str) -> int:
    if value is None:
        return minimum
    explicit = _positive_int(value, name)
    if explicit < minimum:
        raise TopologyError(f"{name} must be >= {minimum}, got {explicit}")
    return explicit


def _canonical_sample_count(nodes: str, nr: int, sample_count: int | None) -> int:
    if nodes == "grid":
        if sample_count is None:
            return nr
        value = _positive_int(sample_count, "sample_count")
        if value != nr:
            raise TopologyError("grid source nodes require sample_count == Nr")
        return value
    if sample_count is None:
        raise TopologyError("uniform source nodes require an explicit sample_count")
    return _positive_int(sample_count, "sample_count")


def _readonly_1d_or_default(
    value: np.ndarray | list[float] | tuple[float, ...] | None,
    name: str,
) -> np.ndarray:
    if value is None:
        arr = np.zeros(1, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"{name} must have at least one entry")
        arr = arr.copy()
    arr.setflags(write=False)
    return arr


def _readonly_1d(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    out = arr.copy()
    out.setflags(write=False)
    return out
