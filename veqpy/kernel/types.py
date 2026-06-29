from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from veqpy.cpp.options import (
    initial_policy_code,
    residual_normalization_code,
    solver_method_code,
)
from veqpy.model import Boundary, Problem, Topology
from veqpy.operator.packed_layout import (
    build_profile_layout,
    build_profile_names,
    packed_size,
)
from veqpy.operator.source_plan import _scaled_source_inputs

_KERNEL_TOPOLOGY_KEY_LENGTH = 32


@dataclass(frozen=True, slots=True)
class KernelBuild:
    """Artifact build and packed-layout configuration for one VEQlib kernel."""

    layout: str = "degree"
    build: str = "fastmath"
    cmake_build_type: str | None = None
    fp_mode: str | None = None
    enable_enzyme: bool | None = None
    enable_native_optimizations: bool | None = None
    enable_thin_lto: bool | None = None
    analysis: bool | None = None
    enzyme_jacobian_batch_width: int | None = None

    def topology_kwargs(self) -> dict[str, object]:
        """Return keyword arguments consumed by the legacy topology builder."""

        return {
            "layout": self.layout,
            "build": self.build,
            "cmake_build_type": self.cmake_build_type,
            "fp_mode": self.fp_mode,
            "enable_enzyme": self.enable_enzyme,
            "enable_native_optimizations": self.enable_native_optimizations,
            "enable_thin_lto": self.enable_thin_lto,
            "analysis": self.analysis,
            "enzyme_jacobian_batch_width": self.enzyme_jacobian_batch_width,
        }


@dataclass(frozen=True, slots=True)
class KernelTopology:
    """Compile-time VEQlib kernel topology, excluding build/layout policy."""

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
        legacy = self.to_legacy_topology(KernelBuild())
        values: dict[str, object] = {
            "h_count": legacy.h_count,
            "v_count": legacy.v_count,
            "kappa_count": legacy.kappa_count,
            "psin_count": legacy.psin_count,
            "F_count": legacy.F_count,
            "c_counts": legacy.c_counts,
            "s_counts": legacy.s_counts,
            "Nr": legacy.Nr,
            "Nt": legacy.Nt,
            "route": legacy.route,
            "coordinate": legacy.coordinate,
            "constraint": legacy.constraint,
            "nodes": legacy.nodes,
            "sample_count": legacy.sample_count,
            "quadrature": legacy.quadrature,
            "calculus": legacy.calculus,
            "L_max": legacy.L_max,
            "M_max": legacy.M_max,
            "K_max": legacy.K_max,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        expected_key = self.compute_key()
        if self.key is not None and self.key != expected_key:
            raise ValueError(
                "key does not match canonical kernel topology: "
                f"got {self.key!r}, expected {expected_key!r}"
            )
        object.__setattr__(self, "key", expected_key)

    @classmethod
    def from_legacy_topology(cls, topology: Topology) -> Self:
        """Create a build/layout-free topology view from the legacy topology type."""

        return cls(
            h_count=topology.h_count,
            v_count=topology.v_count,
            kappa_count=topology.kappa_count,
            psin_count=topology.psin_count,
            F_count=topology.F_count,
            c_counts=topology.c_counts,
            s_counts=topology.s_counts,
            Nr=topology.Nr,
            Nt=topology.Nt,
            route=topology.route,
            coordinate=topology.coordinate,
            constraint=topology.constraint,
            nodes=topology.nodes,
            sample_count=topology.sample_count,
            quadrature=topology.quadrature,
            calculus=topology.calculus,
            L_max=topology.L_max,
            M_max=topology.M_max,
            K_max=topology.K_max,
        )

    def to_legacy_topology(self, build: KernelBuild | None = None) -> Topology:
        """Lower the split topology/build pair to the current VEQlib artifact type."""

        build = KernelBuild() if build is None else build
        return Topology(
            h_count=self.h_count,
            v_count=self.v_count,
            kappa_count=self.kappa_count,
            psin_count=self.psin_count,
            F_count=self.F_count,
            c_counts=self.c_counts,
            s_counts=self.s_counts,
            Nr=self.Nr,
            Nt=self.Nt,
            route=self.route,
            coordinate=self.coordinate,
            constraint=self.constraint,
            nodes=self.nodes,
            sample_count=self.sample_count,
            quadrature=self.quadrature,
            calculus=self.calculus,
            L_max=self.L_max,
            M_max=self.M_max,
            K_max=self.K_max,
            **build.topology_kwargs(),
        )

    def active_profiles(self) -> dict[str, int]:
        """Return active packed profile lengths keyed by profile name."""

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

    def packed_size(self, *, build: KernelBuild | None = None) -> int:
        """Return the packed unknown length for this topology/build layout."""

        legacy = self.to_legacy_topology(build)
        _profile_l, coeff_index, _order_offsets = build_profile_layout(
            self.active_profiles(),
            profile_names=build_profile_names(legacy.M_max),
            profile_first=legacy.layout_profile_first,
        )
        return packed_size(coeff_index)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a build/layout-free canonical topology payload."""

        legacy_payload = self.to_legacy_topology(KernelBuild()).to_canonical_dict()
        return {
            "profiles": legacy_payload["profiles"],
            "grid": legacy_payload["grid"],
            "source": legacy_payload["source"],
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


@dataclass(frozen=True, slots=True)
class KernelInput:
    """Runtime physical input lowered for one VEQlib kernel solve."""

    boundary: Boundary
    scaled_heat: np.ndarray
    scaled_current: np.ndarray
    scaled_Ip: float = np.nan
    beta: float = np.nan
    fix_rho: float = 0.05
    case_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary", _coerce_boundary(self.boundary))
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
        object.__setattr__(self, "fix_rho", float(self.fix_rho))
        case_name = None if self.case_name is None else str(self.case_name)
        object.__setattr__(self, "case_name", case_name)

    @classmethod
    def from_problem(
        cls,
        problem: Problem,
        *,
        fix_rho: float = 0.05,
        case_name: str | None = None,
    ) -> Self:
        """Lower a user-facing ``Problem`` into kernel-ready runtime input."""

        scaled_heat, scaled_current, scaled_Ip, beta = _scaled_source_inputs(problem)
        return cls(
            boundary=problem.boundary,
            scaled_heat=scaled_heat,
            scaled_current=scaled_current,
            scaled_Ip=scaled_Ip,
            beta=beta,
            fix_rho=fix_rho,
            case_name=case_name,
        )

    def to_payload_dict(self) -> dict[str, Any]:
        boundary = self.boundary
        payload: dict[str, Any] = {
            "boundary": {
                "a": boundary.a,
                "R0": boundary.R0,
                "Z0": boundary.Z0,
                "B0": boundary.B0,
                "ka": boundary.ka,
                "c_offsets": np.asarray(boundary.c_offsets, dtype=np.float64).tolist(),
                "s_offsets": np.asarray(boundary.s_offsets, dtype=np.float64).tolist(),
            },
            "source": {
                "scaled_heat": self.scaled_heat.tolist(),
                "scaled_current": self.scaled_current.tolist(),
            },
            "constraints": {"fix_rho": self.fix_rho},
        }
        if self.case_name is not None:
            payload["case_name"] = self.case_name
        if np.isfinite(self.scaled_Ip):
            payload["constraints"]["scaled_Ip"] = self.scaled_Ip
        if np.isfinite(self.beta):
            payload["constraints"]["beta"] = self.beta
        return payload


@dataclass(frozen=True, slots=True)
class KernelSolve:
    """Runtime solve policy for one VEQlib kernel invocation."""

    method: str | int = "powell"
    max_residual: float = 1.0e-6
    max_evaluations: int | None = None
    accepted_residual_factor: float = 10.0
    accepted_residual_floor: float = 1.0e-5
    initial: str | int = "cold"
    norm: str | int = "fast"
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e6
    residual_normalization_huber_tau: float = 3.0
    residual_normalization_probe_count: int = 4
    residual_normalization_probe_step: float = 1.0e-6
    residual_normalization_sensitivity_lambda: float = 0.5

    def to_payload_dict(self, *, x_size: int) -> dict[str, Any]:
        max_evaluations = (
            x_size * x_size if self.max_evaluations is None else int(self.max_evaluations)
        )
        if max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        return {
            "method_code": solver_method_code(self.method),
            "max_residual": float(self.max_residual),
            "max_evaluations": max_evaluations,
            "accepted_residual_factor": float(self.accepted_residual_factor),
            "accepted_residual_floor": float(self.accepted_residual_floor),
            "initial_policy_code": initial_policy_code(self.initial),
            "residual_normalization_code": residual_normalization_code(self.norm),
            "residual_normalization_floor": float(self.residual_normalization_floor),
            "residual_normalization_max_ratio": float(self.residual_normalization_max_ratio),
            "residual_normalization_huber_tau": float(self.residual_normalization_huber_tau),
            "residual_normalization_probe_count": int(self.residual_normalization_probe_count),
            "residual_normalization_probe_step": float(self.residual_normalization_probe_step),
            "residual_normalization_sensitivity_lambda": float(
                self.residual_normalization_sensitivity_lambda
            ),
        }


def _coerce_boundary(boundary: Boundary | dict[str, object]) -> Boundary:
    if isinstance(boundary, Boundary):
        return boundary
    if isinstance(boundary, dict):
        return Boundary(**boundary)
    raise TypeError(f"boundary must be Boundary or dict, got {type(boundary).__name__}")


def _readonly_1d(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    out = arr.copy()
    out.setflags(write=False)
    return out
