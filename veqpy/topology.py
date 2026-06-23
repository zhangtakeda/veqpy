from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

_TOPOLOGY_KEY_LENGTH = 32


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
    build: str = "fastmath"  # fastmath, fastmath-enzyme, release, release-enzyme, debug
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
        quadrature = _normalize_token(self.quadrature, "quadrature").lower()
        if quadrature != "legendre":
            raise TopologyError("only legendre quadrature is supported by the topology schema v1")
        calculus = _normalize_token(self.calculus, "calculus").lower()
        if calculus != "spectral":
            raise TopologyError("only spectral calculus is supported by the topology schema v1")
        build = _normalize_token(self.build, "build")
        if build not in {"fastmath", "fastmath-enzyme", "release", "release-enzyme", "debug"}:
            raise TopologyError(
                "build must be one of fastmath, fastmath-enzyme, release, release-enzyme, or debug"
            )
        sample_count = self._canonical_sample_count(nodes, nr)
        inferred_l = _infer_l_max((*profile_counts.values(), *c_counts, *s_counts))
        l_max = _canonical_exact_or_inferred(self.L_max, inferred_l, "L_max")
        inferred_m = _infer_m_max(c_counts, s_counts)
        m_max = _canonical_at_least(self.M_max, inferred_m, "M_max")
        k_max = _canonical_at_least(self.K_max, max(2, m_max), "K_max")

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
                "route": self.route,
                "coordinate": self.coordinate,
                "constraint": self.constraint,
                "nodes": self.nodes,
                "sample_count": self.sample_count,
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

    def validate_supported_for_veqlib_mvp(self) -> None:
        """Reject topology combinations not yet implemented by the VEQlib MVP backend."""

        expected = {
            "route": "PF",
            "coordinate": "psin",
            "constraint": "Ip",
            "nodes": "uniform",
            "quadrature": "legendre",
            "calculus": "spectral",
        }
        actual = {
            "route": self.route,
            "coordinate": self.coordinate,
            "constraint": self.constraint,
            "nodes": self.nodes,
            "quadrature": self.quadrature,
            "calculus": self.calculus,
        }
        mismatches = [
            f"{name}={actual[name]!r} (expected {value!r})"
            for name, value in expected.items()
            if actual[name] != value
        ]
        if mismatches:
            raise TopologyError(
                "VEQlib MVP backend currently supports PF/psin/uniform/Ip only; got "
                + ", ".join(mismatches)
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
    inferred = highest_count - 1
    if inferred < 1:
        raise TopologyError(
            "derived L_max must be at least 1; at least one profile count must be >= 2"
        )
    return inferred


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
