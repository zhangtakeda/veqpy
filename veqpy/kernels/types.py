"""
Module: veqpy.kernels.types

Role:
- Define typed Kernel dataclasses canonicalized before backend ABI lowering.

Notes:
- These dataclasses own stable Python data only. Construction normalizes
  Python-facing inputs and freezes derived scalar fields; backend helpers lower
  them to artifact identity payloads and runtime tuples.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, replace
from typing import Any

import numpy as np

from veqpy.kernels.abi.enums import (
    LAYOUT_CODES,
    SOURCE_ACTIVE_FAMILY_CODES,
    SOURCE_CONSTRAINT_CODES_BY_FLAGS,
    SOURCE_CONSTRAINT_FLAG_ORDER,
    SOURCE_CONSTRAINT_FLAGS_BY_ROUTE,
    SOURCE_CONSTRAINT_LABELS_BY_FLAGS,
    SOURCE_COORDINATE_CODES,
    SOURCE_NODES_CODES,
    SOURCE_PARAMETERIZATION_CODES,
    SOURCE_ROUTE_CODES,
    SUPPORTED_BACKENDS,
)
from veqpy.kernels.abi.identity import compute_topology_key
from veqpy.kernels.abi.options import (
    continue_policy_code,
    initial_policy_code,
    normalize_continue_policy,
    normalize_initial_policy,
    normalize_residual_normalization,
    normalize_solver_method,
    residual_normalization_code,
    solver_method_code,
)
from veqpy.kernels.errors import TopologyError

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


@dataclass(frozen=True, slots=True)
class KernelRecipe:
    """Artifact recipe and packed-layout configuration for one Kernel."""

    backend: str = "cxx"
    layout: str = "degree"
    build: str = "fastmath"
    cmake_build_type: str | None = None
    fp_mode: str | None = None
    enable_enzyme: bool | None = None
    enable_native_optimizations: bool | None = None
    enable_thin_lto: bool | None = None
    analysis: bool | None = None
    enzyme_jacobian_batch_width: int | None = None
    layout_code: int = field(init=False)
    layout_profile_first: bool = field(init=False)

    def __post_init__(self) -> None:
        build = _normalize_build(self.build)
        preset = _BUILD_PRESET_KWARGS[build]
        layout = _normalize_layout(self.layout)
        normalized_values: dict[str, object] = {
            "backend": _normalize_backend(self.backend),
            "layout": layout,
            "build": build,
            "cmake_build_type": _normalize_cmake_build_type(
                self.cmake_build_type,
                default=str(preset["cmake_build_type"]),
            ),
            "fp_mode": _normalize_fp_mode(self.fp_mode, default=str(preset["fp_mode"])),
            "enable_enzyme": _canonical_bool_or_default(
                self.enable_enzyme,
                default=bool(preset["enable_enzyme"]),
                name="enable_enzyme",
            ),
            "enable_native_optimizations": _canonical_bool_or_default(
                self.enable_native_optimizations,
                default=bool(preset["enable_native_optimizations"]),
                name="enable_native_optimizations",
            ),
            "enable_thin_lto": _canonical_bool_or_default(
                self.enable_thin_lto,
                default=bool(preset["enable_thin_lto"]),
                name="enable_thin_lto",
            ),
            "analysis": _canonical_bool_or_default(
                self.analysis,
                default=bool(preset["analysis"]),
                name="analysis",
            ),
            "enzyme_jacobian_batch_width": _canonical_enzyme_jacobian_batch_width(
                self.enzyme_jacobian_batch_width
            ),
            "layout_code": LAYOUT_CODES[layout],
            "layout_profile_first": layout == "family",
        }
        for name, value in normalized_values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class KernelBoundary:
    """Runtime boundary scalars and Fourier offsets.

    ``c_offsets`` is indexed by cosine order and starts at c0. ``s_offsets`` is
    public input for physical sine modes only and starts at s1; backend runtime
    lowering adds the structural s0=0 slot.
    """

    a: float | None = None
    R0: float | None = None
    Z0: float | None = None
    B0: float | None = None
    ka: float | None = None
    c_offsets: np.ndarray | tuple[float, ...] | list[float] | None = None
    s_offsets: tuple[float, ...] | list[float] | np.ndarray | None = None
    R_boundary: InitVar[Any | None] = None
    Z_boundary: InitVar[Any | None] = None
    c_order: InitVar[int | None] = None
    s_order: InitVar[int | None] = None
    fit_maxtol: InitVar[float] = 1.0e-2
    fit_rms: float | None = field(init=False)
    fit_max_curve_error: float | None = field(init=False)
    fit_c_order: int | None = field(init=False)
    fit_s_order: int | None = field(init=False)
    _raw_R_boundary: np.ndarray | None = field(init=False, repr=False, compare=False)
    _raw_Z_boundary: np.ndarray | None = field(init=False, repr=False, compare=False)
    _raw_c_order: int | None = field(init=False, repr=False, compare=False)
    _raw_s_order: int | None = field(init=False, repr=False, compare=False)
    _raw_fit_maxtol: float = field(init=False, repr=False, compare=False)
    _s_offsets_with_s0: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(
        self,
        R_boundary: Any | None,
        Z_boundary: Any | None,
        c_order: int | None,
        s_order: int | None,
        fit_maxtol: float,
    ) -> None:
        if self.B0 is None:
            raise ValueError("B0 is required")
        uses_boundary_points = any(
            value is not None for value in (R_boundary, Z_boundary, c_order, s_order)
        )
        if uses_boundary_points:
            raw_R, raw_Z, raw_c_order, raw_s_order, raw_fit_maxtol = self._coerce_boundary_points(
                R_boundary=R_boundary,
                Z_boundary=Z_boundary,
                c_order=c_order,
                s_order=s_order,
                fit_maxtol=fit_maxtol,
            )
            c_offsets_input = None
            s_offsets_input = None
            object.__setattr__(self, "fit_rms", None)
            object.__setattr__(self, "fit_max_curve_error", None)
            object.__setattr__(self, "fit_c_order", None)
            object.__setattr__(self, "fit_s_order", None)
            object.__setattr__(self, "_raw_R_boundary", raw_R)
            object.__setattr__(self, "_raw_Z_boundary", raw_Z)
            object.__setattr__(self, "_raw_c_order", raw_c_order)
            object.__setattr__(self, "_raw_s_order", raw_s_order)
            object.__setattr__(self, "_raw_fit_maxtol", raw_fit_maxtol)
        else:
            missing = [
                name
                for name, value in (("a", self.a), ("R0", self.R0), ("Z0", self.Z0))
                if value is None
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"{joined} must be provided when RZ boundary points are absent")
            c_offsets_input = self.c_offsets
            s_offsets_input = self.s_offsets
            object.__setattr__(self, "a", float(self.a))
            object.__setattr__(self, "R0", float(self.R0))
            object.__setattr__(self, "Z0", float(self.Z0))
            object.__setattr__(self, "ka", 1.0 if self.ka is None else float(self.ka))
            object.__setattr__(self, "fit_rms", None)
            object.__setattr__(self, "fit_max_curve_error", None)
            object.__setattr__(self, "fit_c_order", None)
            object.__setattr__(self, "fit_s_order", None)
            object.__setattr__(self, "_raw_R_boundary", None)
            object.__setattr__(self, "_raw_Z_boundary", None)
            object.__setattr__(self, "_raw_c_order", None)
            object.__setattr__(self, "_raw_s_order", None)
            object.__setattr__(self, "_raw_fit_maxtol", float(fit_maxtol))
        object.__setattr__(self, "B0", float(self.B0))
        object.__setattr__(self, "c_offsets", _readonly_1d_or_default(c_offsets_input, "c_offsets"))
        s_offsets = _float_tuple_or_default(s_offsets_input, "s_offsets", default=())
        object.__setattr__(self, "s_offsets", s_offsets)
        object.__setattr__(self, "_s_offsets_with_s0", _s_offsets_runtime_array(s_offsets))

    def _coerce_boundary_points(
        self,
        *,
        R_boundary: Any | None,
        Z_boundary: Any | None,
        c_order: int | None,
        s_order: int | None,
        fit_maxtol: float,
    ) -> tuple[np.ndarray, np.ndarray, int, int, float]:
        if R_boundary is None or Z_boundary is None or c_order is None or s_order is None:
            raise ValueError(
                "R_boundary, Z_boundary, c_order, and s_order must be provided together"
            )
        mixed = {
            "a": self.a,
            "R0": self.R0,
            "Z0": self.Z0,
            "ka": self.ka,
            "c_offsets": self.c_offsets,
            "s_offsets": self.s_offsets,
        }
        mixed_names = [name for name, value in mixed.items() if value is not None]
        if mixed_names:
            joined = ", ".join(mixed_names)
            raise ValueError(f"RZ boundary input cannot be mixed with {joined}")
        raw_R = _readonly_1d(R_boundary, "R_boundary")
        raw_Z = _readonly_1d(Z_boundary, "Z_boundary")
        if raw_R.shape != raw_Z.shape:
            raise ValueError(
                "R_boundary and Z_boundary must have the same shape, "
                f"got {raw_R.shape} and {raw_Z.shape}"
            )
        if raw_R.size < 4:
            raise ValueError("R_boundary and Z_boundary must contain at least four points")
        if not np.all(np.isfinite(raw_R)) or not np.all(np.isfinite(raw_Z)):
            raise ValueError("R_boundary and Z_boundary must contain only finite values")
        normalized_maxtol = float(fit_maxtol)
        if normalized_maxtol <= 0.0:
            raise ValueError(f"fit_maxtol must be positive, got {fit_maxtol!r}")
        return (
            raw_R,
            raw_Z,
            _nonnegative_int(c_order, "c_order"),
            _nonnegative_int(s_order, "s_order"),
            normalized_maxtol,
        )


def kernel_boundary_s_offsets_with_s0(boundary: KernelBoundary) -> np.ndarray:
    """Return backend runtime sine offsets indexed directly by Fourier order."""

    return boundary._s_offsets_with_s0


def kernel_boundary_has_raw_points(boundary: KernelBoundary) -> bool:
    """Return whether a KernelBoundary stores R/Z scatter input for fitting."""

    return boundary._raw_R_boundary is not None


def kernel_boundary_raw_fit_spec(
    boundary: KernelBoundary,
) -> tuple[np.ndarray, np.ndarray, int, int, float] | None:
    """Return raw R/Z fit input for internal Kernel materialization."""

    if boundary._raw_R_boundary is None or boundary._raw_Z_boundary is None:
        return None
    assert boundary._raw_c_order is not None
    assert boundary._raw_s_order is not None
    return (
        boundary._raw_R_boundary,
        boundary._raw_Z_boundary,
        boundary._raw_c_order,
        boundary._raw_s_order,
        boundary._raw_fit_maxtol,
    )


def kernel_boundary_shape_orders(boundary: KernelBoundary) -> tuple[int, int]:
    """Return highest c/s boundary orders without forcing R/Z fitting."""

    raw = kernel_boundary_raw_fit_spec(boundary)
    if raw is not None:
        _, _, c_order, s_order, _ = raw
        return c_order, s_order
    return int(np.asarray(boundary.c_offsets, dtype=np.float64).size) - 1, len(boundary.s_offsets)


@dataclass(frozen=True, slots=True)
class KernelTopology:
    """Cxx topology independent from artifact recipe."""

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
    nodes: str
    ip_constraint: bool = False
    beta_constraint: bool = False
    sample_count: int | None = None
    quadrature: str = "legendre"
    calculus: str = "spectral"
    L_max: int | None = None
    M_max: int | None = None
    K_max: int | None = None
    key: str | None = None
    active_profiles: tuple[tuple[str, int], ...] = field(init=False)
    x_size: int = field(init=False)
    source_route_key: tuple[str, str, str] = field(init=False)
    source_route_code: int = field(init=False)
    source_coordinate_code: int = field(init=False)
    constraint_label: str = field(init=False)
    source_constraint_code: int = field(init=False)
    source_nodes_code: int = field(init=False)
    source_active_family: str = field(init=False)
    source_active_family_code: int = field(init=False)
    source_parameterization: str = field(init=False)
    source_parameterization_code: int = field(init=False)
    source_supported_constraints: tuple[str, ...] = field(init=False)
    source_uses_ip_constraint: bool = field(init=False)
    source_uses_beta_constraint: bool = field(init=False)

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
        if route not in SOURCE_ROUTE_CODES:
            raise TopologyError(f"unsupported route {route!r}")
        coordinate = _normalize_token(self.coordinate, "coordinate").lower()
        if coordinate not in SOURCE_COORDINATE_CODES:
            raise TopologyError(f"unsupported coordinate {coordinate!r}")
        nodes = _normalize_token(self.nodes, "nodes").lower()
        if nodes not in SOURCE_NODES_CODES:
            raise TopologyError(f"unsupported source nodes {nodes!r}")
        ip_constraint = _canonical_bool(self.ip_constraint, "ip_constraint")
        beta_constraint = _canonical_bool(self.beta_constraint, "beta_constraint")
        _validate_source_constraint(route, ip_constraint, beta_constraint)
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
        source_parameterization = _source_parameterization(route, coordinate, nodes)
        constraint_flags = (ip_constraint, beta_constraint)
        constraint_label = SOURCE_CONSTRAINT_LABELS_BY_FLAGS[constraint_flags]
        active_profiles = _active_profiles_tuple(profile_counts, c_counts, s_counts)
        normalized_values: dict[str, Any] = {
            **profile_counts,
            "c_counts": c_counts,
            "s_counts": s_counts,
            "Nr": nr,
            "Nt": nt,
            "route": route,
            "coordinate": coordinate,
            "nodes": nodes,
            "ip_constraint": ip_constraint,
            "beta_constraint": beta_constraint,
            "sample_count": sample_count,
            "quadrature": quadrature,
            "calculus": calculus,
            "L_max": l_max,
            "M_max": m_max,
            "K_max": k_max,
            "active_profiles": active_profiles,
            "x_size": sum(count for _, count in active_profiles),
            "source_route_key": (route, coordinate, nodes),
            "source_route_code": SOURCE_ROUTE_CODES[route],
            "source_coordinate_code": SOURCE_COORDINATE_CODES[coordinate],
            "constraint_label": constraint_label,
            "source_constraint_code": SOURCE_CONSTRAINT_CODES_BY_FLAGS[constraint_flags],
            "source_nodes_code": SOURCE_NODES_CODES[nodes],
            "source_active_family": source_active_family,
            "source_active_family_code": SOURCE_ACTIVE_FAMILY_CODES[source_active_family],
            "source_parameterization": source_parameterization,
            "source_parameterization_code": SOURCE_PARAMETERIZATION_CODES[source_parameterization],
            "source_supported_constraints": _supported_constraint_labels(route),
            "source_uses_ip_constraint": ip_constraint,
            "source_uses_beta_constraint": beta_constraint,
        }
        for name, value in normalized_values.items():
            object.__setattr__(self, name, value)
        expected_key = compute_topology_key(self)
        if self.key is not None and self.key != expected_key:
            raise TopologyError(
                "key does not match canonical kernel topology: "
                f"got {self.key!r}, expected {expected_key!r}"
            )
        object.__setattr__(self, "key", expected_key)


@dataclass(frozen=True, slots=True)
class KernelSource:
    """Runtime source and physical constraints for one Kernel solve."""

    heat_profile: np.ndarray | list[float] | tuple[float, ...]
    current_profile: np.ndarray | list[float] | tuple[float, ...]
    Ip: float = np.nan
    beta: float = np.nan
    case_name: str | None = None

    def __post_init__(self) -> None:
        heat = _readonly_1d(self.heat_profile, "heat_profile")
        current = _readonly_1d(self.current_profile, "current_profile")
        if heat.shape != current.shape:
            raise ValueError(
                "heat_profile and current_profile must share the same shape, "
                f"got {heat.shape} and {current.shape}"
            )
        object.__setattr__(self, "heat_profile", heat)
        object.__setattr__(self, "current_profile", current)
        object.__setattr__(self, "Ip", float(self.Ip))
        object.__setattr__(self, "beta", float(self.beta))
        case_name = None if self.case_name is None else str(self.case_name)
        object.__setattr__(self, "case_name", case_name)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Runtime configuration for one Kernel invocation."""

    method: str = "powell"
    max_residual: float = 1.0e-6
    max_evaluations: int | None = None
    accepted_residual_factor: float = 10.0
    accepted_residual_floor: float = 1.0e-5
    initial: str = "cold"
    continuation: str = "warm"
    norm: str = "fast"
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e6
    residual_normalization_huber_tau: float = 3.0
    residual_normalization_probe_count: int = 4
    residual_normalization_probe_step: float = 1.0e-6
    residual_normalization_sensitivity_lambda: float = 0.5
    method_code: int = field(init=False)
    initial_code: int = field(init=False)
    continuation_code: int = field(init=False)
    norm_code: int = field(init=False)

    def __post_init__(self) -> None:
        method = normalize_solver_method(self.method)
        initial = normalize_initial_policy(self.initial)
        continuation = normalize_continue_policy(self.continuation)
        norm = normalize_residual_normalization(self.norm)
        max_evaluations = None
        if self.max_evaluations is not None:
            max_evaluations = _nonnegative_int(self.max_evaluations, "max_evaluations")
        normalized_values: dict[str, object | None] = {
            "method": method,
            "max_residual": float(self.max_residual),
            "max_evaluations": max_evaluations,
            "accepted_residual_factor": float(self.accepted_residual_factor),
            "accepted_residual_floor": float(self.accepted_residual_floor),
            "initial": initial,
            "continuation": continuation,
            "norm": norm,
            "residual_normalization_floor": float(self.residual_normalization_floor),
            "residual_normalization_max_ratio": float(self.residual_normalization_max_ratio),
            "residual_normalization_huber_tau": float(self.residual_normalization_huber_tau),
            "residual_normalization_probe_count": _nonnegative_int(
                self.residual_normalization_probe_count,
                "residual_normalization_probe_count",
            ),
            "residual_normalization_probe_step": float(self.residual_normalization_probe_step),
            "residual_normalization_sensitivity_lambda": float(
                self.residual_normalization_sensitivity_lambda
            ),
            "method_code": solver_method_code(method),
            "initial_code": initial_policy_code(initial),
            "continuation_code": continue_policy_code(continuation),
            "norm_code": residual_normalization_code(norm),
        }
        for name, value in normalized_values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class KernelPrepareResult:
    """Public preparation snapshot returned by Kernel handles for all backends."""

    backend: str
    topology: KernelTopology
    recipe: KernelRecipe
    x_size: int
    residual_size: int
    prepared: bool
    dry_run: bool
    artifact: object | None = None
    warmed: bool = False
    raw_norm: float = np.nan


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Python-owned snapshot of one Kernel solve result."""

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
    preprocess_ms: float = 0.0
    solver_ms: float = 0.0
    postprocess_ms: float = 0.0


def config_with_overrides(config: KernelConfig, **overrides: Any) -> KernelConfig:
    field_names = {item.name for item in fields(config) if item.init}
    unknown = sorted(name for name in overrides if name not in field_names)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unsupported KernelConfig override(s): {names}")
    return replace(config, **overrides)



def _normalize_token(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TopologyError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TopologyError(f"{name} must not be empty")
    return normalized


def _normalize_build(value: str) -> str:
    normalized = _normalize_token(value, "build").lower()
    if normalized in _BUILD_PRESET_KWARGS:
        return normalized
    raise TopologyError("build must be one of fastmath, fastmath-enzyme, release, or debug")


def _validate_source_constraint(route: str, ip_constraint: bool, beta_constraint: bool) -> None:
    flags = (ip_constraint, beta_constraint)
    if flags not in SOURCE_CONSTRAINT_FLAGS_BY_ROUTE[route]:
        label = SOURCE_CONSTRAINT_LABELS_BY_FLAGS[flags]
        raise TopologyError(f"{route} source topology does not support constraint {label!r}")


def _normalize_layout(value: str) -> str:
    normalized = _normalize_token(value, "layout").lower()
    if normalized in LAYOUT_CODES:
        return normalized
    raise TopologyError("layout must be degree or family")


def _normalize_backend(value: str) -> str:
    normalized = _normalize_token(value, "backend").lower()
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise TopologyError("backend must be cxx or numba")


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


def _supported_constraint_labels(route: str) -> tuple[str, ...]:
    supported = SOURCE_CONSTRAINT_FLAGS_BY_ROUTE[route]
    return tuple(
        SOURCE_CONSTRAINT_LABELS_BY_FLAGS[flags]
        for flags in SOURCE_CONSTRAINT_FLAG_ORDER
        if flags in supported
    )


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


def _canonical_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TopologyError(f"{name} must be a bool")
    return value


def _canonical_bool_or_default(value: bool | None, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    return _canonical_bool(value, name)


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


def _active_profiles_tuple(
    profile_counts: dict[str, int],
    c_counts: tuple[int, ...],
    s_counts: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    active: list[tuple[str, int]] = []
    for field_name, profile_name in (
        ("h_count", "h"),
        ("v_count", "v"),
        ("kappa_count", "k"),
    ):
        count = profile_counts[field_name]
        if count > 0:
            active.append((profile_name, count))
    for order, count in enumerate(c_counts):
        if count > 0:
            active.append((f"c{order}", count))
    for order, count in enumerate(s_counts, start=1):
        if count > 0:
            active.append((f"s{order}", count))
    for field_name, profile_name in (("psin_count", "psin"), ("F_count", "F")):
        count = profile_counts[field_name]
        if count > 0:
            active.append((profile_name, count))
    return tuple(active)


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


def _float_tuple_or_default(
    value: np.ndarray | list[float] | tuple[float, ...] | None,
    name: str,
    *,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return tuple(float(item) for item in arr)


def _s_offsets_runtime_array(s_offsets: tuple[float, ...]) -> np.ndarray:
    out = np.zeros(len(s_offsets) + 1, dtype=np.float64)
    if s_offsets:
        out[1:] = np.asarray(s_offsets, dtype=np.float64)
    out.setflags(write=False)
    return out


def _readonly_1d(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    out = arr.copy()
    out.setflags(write=False)
    return out
