"""
Module: operator.operator_case

Role:
- Normalize case inputs into a stable case configuration object.

Public API:
- OperatorCase

Notes:
- `OperatorCase` only stores case inputs.
- Does not build layouts, compute residuals, or manage solver policy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Self

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.model.boundary import Boundary

ProfileCoeffInput = list[float] | np.ndarray | int | None
ProfileCoeff = np.ndarray | None

MU0 = 4.0e-7 * np.pi
SETUP_NORMALIZED_ABS_MIN = 1.0e-3
SETUP_NORMALIZED_ABS_MAX = 1.0e3
SETUP_PHYSICAL_ABS_MIN = SETUP_NORMALIZED_ABS_MIN / MU0
SETUP_PHYSICAL_ABS_MAX = SETUP_NORMALIZED_ABS_MAX / MU0
CURRENT_PROFILE_ROUTES = frozenset({"PI", "PJ1", "PJ2"})


@dataclass(slots=True)
class OperatorCase:
    """Describe the static case inputs required for one operator evaluation."""

    route: str
    coordinate: str
    profile_coeffs: dict[str, ProfileCoeffInput]
    boundary: Boundary
    heat_input: np.ndarray
    current_input: np.ndarray
    nodes: str = "uniform"
    Ip: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """Normalize fields into stable runtime representations after construction."""
        object.__setattr__(self, "route", _normalize_case_value("route", self.route))
        object.__setattr__(self, "coordinate", _normalize_case_value("coordinate", self.coordinate))
        object.__setattr__(self, "nodes", _normalize_case_value("nodes", self.nodes))
        object.__setattr__(
            self, "profile_coeffs", _normalize_case_value("profile_coeffs", self.profile_coeffs)
        )
        object.__setattr__(self, "boundary", _normalize_case_value("boundary", self.boundary))
        object.__setattr__(self, "Ip", _normalize_case_value("Ip", self.Ip))
        object.__setattr__(self, "beta", _normalize_case_value("beta", self.beta))
        object.__setattr__(self, "heat_input", _normalize_case_value("heat_input", self.heat_input))
        object.__setattr__(
            self, "current_input", _normalize_case_value("current_input", self.current_input)
        )
        if self.heat_input.shape != self.current_input.shape:
            raise ValueError(
                f"heat_input and current_input must share the same shape, "
                f"got {self.heat_input.shape} and {self.current_input.shape}"
            )
        _normalize_setup_inputs(self)

    def __setattr__(self, name: str, value) -> None:
        if name in (
            "profile_coeffs",
            "route",
            "boundary",
            "coordinate",
            "nodes",
            "Ip",
            "beta",
            "heat_input",
            "current_input",
        ):
            value = _normalize_case_value(name, value)
        object.__setattr__(self, name, value)

    def __rich__(self) -> Tree:
        tree = Tree("[bold blue]OperatorCase[/]")
        tree.add(f"route: {self.route}")
        tree.add(f"coordinate: {self.coordinate}")
        tree.add(f"nodes: {self.nodes}")
        tree.add(
            f"heat_input: shape={self.heat_input.shape}, "
            f"min={float(np.min(self.heat_input)):.3f}, max={float(np.max(self.heat_input)):.3f}"
        )
        tree.add(
            f"current_input: shape={self.current_input.shape}, "
            f"min={float(np.min(self.current_input)):.3f}, "
            f"max={float(np.max(self.current_input)):.3f}"
        )
        if np.isfinite(self.Ip):
            tree.add(f"Ip(mu0-scaled): {self.Ip:.3e}")
        if np.isfinite(self.beta):
            tree.add(f"beta: {self.beta:.3e}")
        tree.add(self.boundary)
        return tree

    def __str__(self) -> str:
        console = Console(
            color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False
        )
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> Self:
        """Create a copy independent from the current case."""
        clone = object.__new__(type(self))
        object.__setattr__(clone, "route", self.route)
        object.__setattr__(clone, "coordinate", self.coordinate)
        object.__setattr__(clone, "nodes", self.nodes)
        object.__setattr__(clone, "profile_coeffs", _copy_coeffs(self.profile_coeffs))
        object.__setattr__(
            clone,
            "boundary",
            Boundary(
                a=self.a,
                R0=self.R0,
                Z0=self.Z0,
                B0=self.B0,
                ka=self.ka,
                c_offsets=self.c_offsets.copy(),
                s_offsets=self.s_offsets.copy(),
            ),
        )
        object.__setattr__(clone, "heat_input", self.heat_input.copy())
        object.__setattr__(clone, "current_input", self.current_input.copy())
        object.__setattr__(clone, "Ip", self.Ip)
        object.__setattr__(clone, "beta", self.beta)
        return clone

    @property
    def a(self) -> float:
        """Minor radius from the associated boundary."""
        return self.boundary.a

    @property
    def R0(self) -> float:
        """Major-radius reference point from the associated boundary."""
        return self.boundary.R0

    @property
    def Z0(self) -> float:
        """Vertical reference point from the associated boundary."""
        return self.boundary.Z0

    @property
    def B0(self) -> float:
        """Reference toroidal field from the associated boundary."""
        return self.boundary.B0

    @property
    def ka(self) -> float:
        """Elongation from the associated boundary."""
        return self.boundary.ka

    @property
    def c_offsets(self) -> np.ndarray:
        """Cosine-side boundary offsets from the associated boundary."""
        return self.boundary.c_offsets

    @property
    def s_offsets(self) -> np.ndarray:
        """Sine-side boundary offsets from the associated boundary."""
        return self.boundary.s_offsets


def _normalize_coeffs(
    profile_coeffs: dict[str, ProfileCoeffInput],
) -> dict[str, ProfileCoeff]:
    return {name: _normalize_profile_coeff(name, coeff) for name, coeff in profile_coeffs.items()}


def _normalize_profile_coeff(name: str, coeff: ProfileCoeffInput) -> ProfileCoeff:
    if coeff is None:
        return None
    if isinstance(coeff, bool):
        raise TypeError(f"{name} coeff length indicator must be an integer, got bool")
    if isinstance(coeff, Integral):
        length = int(coeff)
        if length <= 0:
            raise ValueError(f"{name} coeff length indicator must be positive, got {coeff}")
        return np.zeros(length, dtype=np.float64)
    if isinstance(coeff, (list, np.ndarray)):
        return _as_1d_array(coeff, name=f"{name} coeff").astype(np.float64, copy=True)
    raise TypeError(f"Invalid {name} coeff type {type(coeff).__name__}")


def _copy_coeffs(profile_coeffs: dict[str, ProfileCoeffInput]) -> dict[str, ProfileCoeff]:
    return {name: _normalize_profile_coeff(name, coeff) for name, coeff in profile_coeffs.items()}


def _as_1d_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _normalize_case_value(name: str, value):
    if name == "route":
        return str(value).upper()
    if name == "profile_coeffs":
        return _normalize_coeffs(value)
    if name == "boundary":
        if isinstance(value, Boundary):
            return value
        if isinstance(value, dict):
            return Boundary(**value)
        raise TypeError(f"boundary must be Boundary or dict, got {type(value).__name__}")
    if name == "coordinate":
        coord = str(value).lower()
        if coord not in ("rho", "psin"):
            raise ValueError(f"coordinate must be one of ('rho', 'psin'), got {value!r}")
        return coord
    if name == "nodes":
        nodes = str(value).lower()
        if nodes not in ("uniform", "grid"):
            raise ValueError(f"nodes must be one of ('uniform', 'grid'), got {value!r}")
        return nodes
    if name in ("Ip", "beta"):
        return np.nan if value is None else float(value)
    if name in ("heat_input", "current_input"):
        return _as_1d_array(value, name=name).copy()
    return value


def _normalize_setup_inputs(case: OperatorCase) -> None:
    rejected_inputs: list[str] = []

    _normalize_setup_profile(
        case,
        field_name="heat_input",
        requires_mu0_scaling=_heat_input_requires_mu0_scaling(case),
        rejected_inputs=rejected_inputs,
    )
    _normalize_setup_profile(
        case,
        field_name="current_input",
        requires_mu0_scaling=_current_input_requires_mu0_scaling(case),
        rejected_inputs=rejected_inputs,
    )
    if np.isfinite(case.Ip):
        _normalize_setup_ip(case, rejected_inputs=rejected_inputs)

    if rejected_inputs:
        message = (
            "Rejected setup input magnitude: "
            + "; ".join(rejected_inputs)
            + ". Expected non-current/pressure setup profiles in "
            f"[{SETUP_NORMALIZED_ABS_MIN:.0e}, {SETUP_NORMALIZED_ABS_MAX:.0e}], "
            "and physical current/pressure setup inputs in "
            f"[{SETUP_PHYSICAL_ABS_MIN:.3e}, {SETUP_PHYSICAL_ABS_MAX:.3e}] before mu0 "
            "scaling. Pass unnormalized setup values; accepted physical current/pressure "
            "inputs are scaled by mu0 during construction."
        )
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=3,
        )
        raise ValueError(message)


def _heat_input_requires_mu0_scaling(case: OperatorCase) -> bool:
    return True


def _current_input_requires_mu0_scaling(case: OperatorCase) -> bool:
    return case.route in CURRENT_PROFILE_ROUTES


def _normalize_setup_profile(
    case: OperatorCase,
    *,
    field_name: str,
    requires_mu0_scaling: bool,
    rejected_inputs: list[str],
) -> None:
    values = getattr(case, field_name)
    max_abs = _profile_max_abs(values)
    if max_abs == 0.0:
        return
    if requires_mu0_scaling:
        if _in_closed_range(max_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
            values *= MU0
        else:
            rejected_inputs.append(
                f"{field_name} max_abs={max_abs:.3e} outside physical current/pressure setup range"
            )
        return
    if not _in_closed_range(max_abs, SETUP_NORMALIZED_ABS_MIN, SETUP_NORMALIZED_ABS_MAX):
        rejected_inputs.append(
            f"{field_name} max_abs={max_abs:.3e} outside non-current/pressure setup range"
        )


def _normalize_setup_ip(case: OperatorCase, *, rejected_inputs: list[str]) -> None:
    ip_abs = abs(float(case.Ip))
    if ip_abs == 0.0:
        return
    if _in_closed_range(ip_abs, SETUP_PHYSICAL_ABS_MIN, SETUP_PHYSICAL_ABS_MAX):
        object.__setattr__(case, "Ip", float(case.Ip) * MU0)
    else:
        rejected_inputs.append(f"Ip abs={ip_abs:.3e} outside physical current setup range")


def _profile_max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _in_closed_range(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper
