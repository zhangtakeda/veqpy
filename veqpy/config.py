"""User-facing VEQ configuration normalization for the VEQ Module.

The public boundary accepts ordinary mappings only.  This module is the one
place where topology fields, solver tokens, and build-only options are
validated before the private four-buffer ABI is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .kernels.abi.enums import SUPPORTED_BACKENDS
from .kernels.abi.options import (
    CONTINUE_POLICY_CODES,
    INITIAL_POLICY_CODES,
    RESIDUAL_NORMALIZATION_CODES,
    SOLVER_METHOD_CODES,
)
from .kernels.contracts import KernelConfig, KernelTopology

TOPOLOGY_FIELDS = frozenset(
    {
        "Nr",
        "Nt",
        "route",
        "coordinate",
        "constraint",
        "h_count",
        "v_count",
        "kappa_count",
        "psin_count",
        "F_count",
        "c_counts",
        "s_counts",
        "quadrature",
        "calculus",
        "L_max",
        "M_max",
        "K_max",
    }
)
SOLVER_FIELDS = frozenset(
    {
        "method",
        "max_residual",
        "max_evaluations",
        "accepted_residual_factor",
        "accepted_residual_floor",
        "initial",
        "continuation",
        "norm",
        "residual_normalization_floor",
        "residual_normalization_max_ratio",
        "residual_normalization_huber_tau",
        "residual_normalization_probe_count",
        "residual_normalization_probe_step",
        "residual_normalization_sensitivity_lambda",
    }
)
SOLVER_DEFAULTS: dict[str, Any] = {
    "method": "powell",
    "max_residual": 1.0e-6,
    "max_evaluations": None,
    "accepted_residual_factor": 10.0,
    "accepted_residual_floor": 1.0e-5,
    "initial": "cold",
    "continuation": "warm",
    "norm": "fast",
    "residual_normalization_floor": 1.0,
    "residual_normalization_max_ratio": 1.0e6,
    "residual_normalization_huber_tau": 3.0,
    "residual_normalization_probe_count": 4,
    "residual_normalization_probe_step": 1.0e-6,
    "residual_normalization_sensitivity_lambda": 0.5,
}


def normalize_topology(value: Mapping[str, Any]) -> KernelTopology:
    """Validate and lower one public topology mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"topology must be a dict-like mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - TOPOLOGY_FIELDS)
    if unknown:
        raise TypeError(f"unsupported topology field(s): {', '.join(unknown)}")
    required = {"Nr", "Nt", "route", "coordinate"}
    missing = sorted(required - set(value))
    if missing:
        raise TypeError(f"topology is missing required field(s): {', '.join(missing)}")
    defaults: dict[str, Any] = {
        "constraint": "none",
        "h_count": 0,
        "v_count": 0,
        "kappa_count": 0,
        "psin_count": 0,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (),
        "quadrature": "legendre",
        "calculus": "spectral",
        "L_max": None,
        "M_max": None,
        "K_max": None,
    }
    defaults.update(dict(value))
    return KernelTopology(**defaults)


def merge_solver(
    build: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge solver defaults with build and one-call overrides."""

    merged = dict(SOLVER_DEFAULTS)
    for label, values in (("solver", build), ("solver override", override)):
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} must be a dict-like mapping")
        unknown = sorted(set(values) - SOLVER_FIELDS)
        if unknown:
            raise TypeError(f"unsupported {label} field(s): {', '.join(unknown)}")
        merged.update(values)
    return merged


def solver_config(value: Mapping[str, Any] | None = None) -> KernelConfig:
    """Lower a normalized solver mapping to numeric internal policy codes."""

    values = merge_solver(value)
    try:
        return KernelConfig(
            method_code=SOLVER_METHOD_CODES[str(values["method"]).strip().lower()],
            max_residual=values["max_residual"],
            max_evaluations=values["max_evaluations"],
            accepted_residual_factor=values["accepted_residual_factor"],
            accepted_residual_floor=values["accepted_residual_floor"],
            initial_code=INITIAL_POLICY_CODES[str(values["initial"]).strip().lower()],
            continuation_code=CONTINUE_POLICY_CODES[str(values["continuation"]).strip().lower()],
            norm_code=RESIDUAL_NORMALIZATION_CODES[str(values["norm"]).strip().lower()],
            residual_normalization_floor=values["residual_normalization_floor"],
            residual_normalization_max_ratio=values["residual_normalization_max_ratio"],
            residual_normalization_huber_tau=values["residual_normalization_huber_tau"],
            residual_normalization_probe_count=values["residual_normalization_probe_count"],
            residual_normalization_probe_step=values["residual_normalization_probe_step"],
            residual_normalization_sensitivity_lambda=values[
                "residual_normalization_sensitivity_lambda"
            ],
        )
    except KeyError as error:
        raise ValueError(f"unsupported solver token {error.args[0]!r}") from error


def normalize_backend(value: str) -> str:
    """Normalize the five supported backend tokens and the ``cxx`` alias."""

    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"backend must be one of {choices}")
    return "cxx-relaxed" if normalized == "cxx" else normalized


def normalize_cpu_affinity(value: bool | int | None) -> bool | int | None:
    """Validate the optional build-only CPU affinity request."""

    if value is None or type(value) is bool:
        return value
    if type(value) is int and value >= 0:
        return value
    raise TypeError("cpu_affinity must be None, bool, or a non-negative int")


def normalize_artifact_dir(value: str | Path | None) -> Path | None:
    """Normalize the optional build-only artifact directory."""

    return None if value is None else Path(value).expanduser().resolve()


def normalize_rebuild(value: bool) -> bool:
    """Validate the build-only rebuild switch."""

    if type(value) is not bool:
        raise TypeError("rebuild must be a bool")
    return value


__all__ = [
    "SOLVER_DEFAULTS",
    "SOLVER_FIELDS",
    "TOPOLOGY_FIELDS",
    "merge_solver",
    "normalize_artifact_dir",
    "normalize_backend",
    "normalize_cpu_affinity",
    "normalize_rebuild",
    "normalize_topology",
    "solver_config",
]
