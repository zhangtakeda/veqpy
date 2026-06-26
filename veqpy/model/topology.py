from __future__ import annotations

import base64
import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any

_TOPOLOGY_KEY_LENGTH = 32
_SOURCE_ROUTE_CODES = {
    "PF": 1,
    "PP": 2,
    "PI": 3,
    "PJ1": 4,
    "PJ2": 5,
    "PQ": 6,
}
_SOURCE_COORDINATE_CODES = {
    "rho": 1,
    "psin": 2,
}
_SOURCE_CONSTRAINT_CODES = {
    "null": 0,
    "Ip": 1,
    "beta": 2,
    "Ip_beta": 3,
}
_SOURCE_NODES_CODES = {
    "uniform": 1,
    "grid": 2,
}
_SOURCE_ACTIVE_FAMILY_CODES = {
    "none": 0,
    "psin": 1,
    "F": 2,
}
_SOURCE_PARAMETERIZATION_CODES = {
    "identity": 0,
    "sqrt_psin": 1,
}
_SOURCE_CONSTRAINTS_BY_ROUTE = {
    "PF": frozenset({"null", "Ip", "beta"}),
    "PP": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PI": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PJ1": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PJ2": frozenset({"null", "Ip", "beta", "Ip_beta"}),
    "PQ": frozenset({"null", "Ip", "beta", "Ip_beta"}),
}
_VEQLIB_MVP_ROUTE_CONSTRAINTS = {
    ("PF", "rho", "uniform"): frozenset({"null", "Ip", "beta"}),
    ("PF", "rho", "grid"): frozenset({"null", "Ip", "beta"}),
    ("PF", "psin", "uniform"): frozenset({"null", "Ip", "beta"}),
    ("PF", "psin", "grid"): frozenset({"null", "Ip", "beta"}),
    ("PP", "rho", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PP", "rho", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PP", "psin", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PP", "psin", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PI", "rho", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PI", "rho", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PI", "psin", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PI", "psin", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ1", "rho", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ1", "rho", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ1", "psin", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ1", "psin", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ2", "rho", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ2", "rho", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ2", "psin", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PJ2", "psin", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PQ", "rho", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PQ", "rho", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PQ", "psin", "uniform"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
    ("PQ", "psin", "grid"): frozenset({"null", "Ip", "beta", "Ip_beta"}),
}
_VEQLIB_MVP_ROUTE_KEYS = frozenset(_VEQLIB_MVP_ROUTE_CONSTRAINTS)
_VEQLIB_MVP_ROUTES = frozenset(
    {
        route_key[0] for route_key in _VEQLIB_MVP_ROUTE_KEYS
    }
)
_LAYOUT_CODES = {
    "degree": 0,
    "family": 1,
}
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
    """Raised when a VEQlib topology cannot be canonicalized."""


@dataclass(frozen=True, slots=True)
class Topology:
    """Canonical kernel topology for the experimental VEQlib Solver path."""

    # profiles: ProfileTopology
    h_count: int
    v_count: int
    kappa_count: int
    psin_count: int
    F_count: int
    c_counts: tuple[int, ...]  # c0,c1,...
    s_counts: tuple[int, ...]  # s1,s2,...

    # grid: GridTopology
    Nr: int
    Nt: int

    # source: SourceTopology
    route: str  # "PF", "PP", "PI", "PJ1", "PJ2", "PQ"
    coordinate: str  # "psin", "rho"
    constraint: str  # "Ip", "beta", "Ip_beta", "null"
    nodes: str  # "uniform", "grid"
    sample_count: int | None = None  # grid 默认 Nr；uniform 必须由 case/topology 明确给出

    # kernel/source policies
    quadrature: str = "legendre"
    calculus: str = "spectral"
    L_max: int | None = None  # 从 profile count 推断；显式值必须一致
    M_max: int | None = None  # 默认从 c_counts/s_counts 推断；允许显式更高
    K_max: int | None = None  # 默认 max(2, M_max)；允许显式更高

    # artifact/cache metadata
    build: str = "fastmath"  # fastmath, fastmath-enzyme, release, debug
    layout: str = "degree"  # degree or family/profile-first
    cmake_build_type: str | None = None
    fp_mode: str | None = None
    enable_enzyme: bool | None = None
    enable_native_optimizations: bool | None = None
    enable_thin_lto: bool | None = None
    analysis: bool | None = None
    enzyme_jacobian_batch_width: int | None = None  # 0 lets C++ choose from x_size
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
        if route not in {"PF", "PP", "PI", "PJ1", "PJ2", "PQ"}:
            raise TopologyError(f"unsupported route {route!r}")

        coordinate = _normalize_token(self.coordinate, "coordinate").lower()
        if coordinate not in {"rho", "psin"}:
            raise TopologyError(f"unsupported coordinate {coordinate!r}")

        nodes = _normalize_token(self.nodes, "nodes").lower()
        if nodes not in {"uniform", "grid"}:
            raise TopologyError(f"unsupported source nodes {nodes!r}")

        constraint = _normalize_constraint(self.constraint)
        _validate_source_constraint(route, constraint)
        quadrature = _normalize_token(self.quadrature, "quadrature").lower()
        if quadrature != "legendre":
            raise TopologyError("only legendre quadrature is supported by the topology schema v1")
        calculus = _normalize_token(self.calculus, "calculus").lower()
        if calculus != "spectral":
            raise TopologyError("only spectral calculus is supported by the topology schema v1")
        build = _normalize_token(self.build, "build")
        if build not in _BUILD_PRESET_KWARGS:
            raise TopologyError(
                "build must be one of fastmath, fastmath-enzyme, release, or debug"
            )
        layout = _normalize_layout(self.layout)
        sample_count = self._canonical_sample_count(nodes, nr)
        inferred_l = _infer_l_max((*profile_counts.values(), *c_counts, *s_counts))
        l_max = _canonical_exact_or_inferred(self.L_max, inferred_l, "L_max")
        inferred_m = _infer_m_max(c_counts, s_counts)
        m_max = _canonical_at_least(self.M_max, inferred_m, "M_max")
        k_max = _canonical_at_least(self.K_max, max(2, m_max), "K_max")
        source_active_family = _source_active_family(route, coordinate, nodes)
        _validate_source_active_family(
            source_active_family,
            psin_count=profile_counts["psin_count"],
            f_count=profile_counts["F_count"],
        )
        _warn_source_profile_ownership(
            route,
            source_active_family=source_active_family,
            psin_count=profile_counts["psin_count"],
            f_count=profile_counts["F_count"],
        )
        build_preset = _BUILD_PRESET_KWARGS[build]
        cmake_build_type = _normalize_cmake_build_type(
            self.cmake_build_type,
            default=str(build_preset["cmake_build_type"]),
        )
        fp_mode = _normalize_fp_mode(self.fp_mode, default=str(build_preset["fp_mode"]))
        enable_enzyme = _canonical_bool(
            self.enable_enzyme,
            default=bool(build_preset["enable_enzyme"]),
            name="enable_enzyme",
        )
        enable_native_optimizations = _canonical_bool(
            self.enable_native_optimizations,
            default=bool(build_preset["enable_native_optimizations"]),
            name="enable_native_optimizations",
        )
        enable_thin_lto = _canonical_bool(
            self.enable_thin_lto,
            default=bool(build_preset["enable_thin_lto"]),
            name="enable_thin_lto",
        )
        analysis = _canonical_bool(
            self.analysis,
            default=bool(build_preset["analysis"]),
            name="analysis",
        )
        enzyme_jacobian_batch_width = _canonical_enzyme_jacobian_batch_width(
            self.enzyme_jacobian_batch_width
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
            "build": build,
            "layout": layout,
            "cmake_build_type": cmake_build_type,
            "fp_mode": fp_mode,
            "enable_enzyme": enable_enzyme,
            "enable_native_optimizations": enable_native_optimizations,
            "enable_thin_lto": enable_thin_lto,
            "analysis": analysis,
            "enzyme_jacobian_batch_width": enzyme_jacobian_batch_width,
        }
        for key, value in normalized_values.items():
            object.__setattr__(self, key, value)

        expected_key = self.compute_key()
        if self.key is not None and self.key != expected_key:
            raise TopologyError(
                "key does not match canonical topology: "
                f"got {self.key!r}, expected {expected_key!r}"
            )
        object.__setattr__(self, "key", expected_key)

    def _canonical_sample_count(self, nodes: str, nr: int) -> int:
        if nodes == "grid":
            if self.sample_count is None:
                return nr
            sample_count = _positive_int(self.sample_count, "sample_count")
            if sample_count != nr:
                raise TopologyError("grid source nodes require sample_count == Nr")
            return sample_count
        if self.sample_count is None:
            raise TopologyError("uniform source nodes require an explicit sample_count")
        return _positive_int(self.sample_count, "sample_count")

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the deterministic payload used for artifact indexing."""

        return {
            "build": self.build,
            "build_options": self.build_options_dict(),
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
            "source": {
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
            },
            "layout": {
                "packed": self.layout,
                "profile_first": self.layout_profile_first,
                "code": self.layout_code,
            },
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the canonical payload with stable key and whitespace rules."""

        return json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def compute_key(self) -> str:
        """Return the deterministic key for the canonical topology payload."""

        digest = hashlib.sha256(self.to_json_bytes()).digest()
        encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
        return encoded[:_TOPOLOGY_KEY_LENGTH]

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
        ordered_constraints = ("Ip_beta", "Ip", "beta", "null")
        supported_constraints = _SOURCE_CONSTRAINTS_BY_ROUTE[self.route]
        return tuple(
            constraint for constraint in ordered_constraints if constraint in supported_constraints
        )

    @property
    def source_uses_ip_constraint(self) -> bool:
        return self.constraint in {"Ip", "Ip_beta"}

    @property
    def source_uses_beta_constraint(self) -> bool:
        return self.constraint in {"beta", "Ip_beta"}

    def source_policy_dict(self) -> dict[str, object]:
        """Return source-route policy metadata derived at topology setup time."""

        return {
            "route_key": list(self.source_route_key),
            "active_family": self.source_active_family,
            "active_family_code": self.source_active_family_code,
            "parameterization": self.source_parameterization,
            "parameterization_code": self.source_parameterization_code,
            "supported_constraints": list(self.source_supported_constraints),
            "uses_Ip": self.source_uses_ip_constraint,
            "uses_beta": self.source_uses_beta_constraint,
        }

    @property
    def layout_code(self) -> int:
        return _LAYOUT_CODES[self.layout]

    @property
    def layout_profile_first(self) -> bool:
        return self.layout == "family"

    def build_options_dict(self) -> dict[str, object]:
        """Return preset-expanded CMake/kernel build options."""

        return {
            "preset": self.build,
            "cmake_build_type": self.cmake_build_type,
            "fp_mode": self.fp_mode,
            "enable_enzyme": self.enable_enzyme,
            "enable_native_optimizations": self.enable_native_optimizations,
            "enable_thin_lto": self.enable_thin_lto,
            "analysis": self.analysis,
            "enzyme_jacobian_batch_width": self.enzyme_jacobian_batch_width,
        }

    def validate_supported_for_veqlib_mvp(self) -> None:
        """Reject topology combinations not yet implemented by the VEQlib MVP backend."""

        expected = {
            "quadrature": "legendre",
            "calculus": "spectral",
            "layout": "degree",
        }
        actual = {
            "quadrature": self.quadrature,
            "calculus": self.calculus,
            "layout": self.layout,
        }
        mismatches = [
            f"{name}={actual[name]!r} (expected {value!r})"
            for name, value in expected.items()
            if actual[name] != value
        ]
        if self.route not in _VEQLIB_MVP_ROUTES:
            supported = ", ".join(sorted(_VEQLIB_MVP_ROUTES))
            mismatches.append(f"route={self.route!r} (expected one of {supported})")
        if self.source_route_key not in _VEQLIB_MVP_ROUTE_KEYS:
            supported = ", ".join(
                "/".join(route_key) for route_key in sorted(_VEQLIB_MVP_ROUTE_KEYS)
            )
            mismatches.append(
                f"route_key={self.source_route_key!r} (expected one of {supported})"
            )
        supported_constraints = _VEQLIB_MVP_ROUTE_CONSTRAINTS.get(self.source_route_key)
        if supported_constraints is None or self.constraint not in supported_constraints:
            supported = (
                "unavailable"
                if supported_constraints is None
                else ", ".join(sorted(supported_constraints))
            )
            mismatches.append(
                f"constraint={self.constraint!r} (expected one of {supported})"
            )
        if mismatches:
            raise TopologyError(
                "VEQlib MVP backend currently supports PF, PP, PI, PJ1, PJ2, "
                "and PQ source topology slices for the implemented constraint "
                "sets only; got "
                + ", ".join(mismatches)
            )
        if self.source_active_family != "F" and self.F_count > 0:
            raise TopologyError("VEQlib MVP backend only accepts F_count > 0 for PJ2")
        if self.source_active_family == "F" and self.F_count <= 0:
            raise TopologyError("VEQlib MVP PJ2 topology requires F_count > 0")
        if self.source_active_family != "psin" and self.psin_count > 0:
            raise TopologyError(
                "VEQlib MVP source-owned topology does not accept psin_count > 0"
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
    supported_constraints = _SOURCE_CONSTRAINTS_BY_ROUTE[route]
    if constraint not in supported_constraints:
        supported = ", ".join(sorted(supported_constraints))
        raise TopologyError(
            f"{route} source topology does not support constraint {constraint!r}; "
            f"supported constraints: {supported}"
        )


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


def _warn_source_profile_ownership(
    route: str,
    *,
    source_active_family: str,
    psin_count: int,
    f_count: int,
) -> None:
    if source_active_family == "F" and psin_count > 0:
        warnings.warn(
            "PJ2 source topology uses active F ownership; psin_count is ignored by "
            "the source kernel and should normally be 0",
            UserWarning,
            stacklevel=3,
        )
    if route != "PJ2" and f_count > 0:
        warnings.warn(
            "Only PJ2 source topology uses active F ownership; F_count is ignored by "
            f"{route} source kernels and should normally be 0",
            UserWarning,
            stacklevel=3,
        )


def _normalize_cmake_build_type(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    normalized = _normalize_token(value, "cmake_build_type").lower()
    mapping = {
        "debug": "Debug",
        "release": "Release",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise TopologyError("cmake_build_type must be Debug or Release") from exc


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
    highest_count = max(profile_counts, default=0)
    if highest_count < 1:
        raise TopologyError(
            "derived L_max requires at least one active profile count"
        )
    return max(1, highest_count - 1)


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
