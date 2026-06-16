"""
Module: model.problem

Role:
- Define the validated fixed-boundary equilibrium problem consumed by Operator.
- Normalize source-route spelling, profile coefficients, boundary, source-array
  shape, and optional constraint sentinels without applying runtime unit scaling.

Public API:
- Problem

Notes:
- `Problem` stores user-facing problem inputs in one canonical form.
- It does not build packed layouts, allocate runtime memory, compute residuals,
  apply engine-unit scaling, or manage solver policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Self

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.base import Serial
from veqpy.model.boundary import Boundary
from veqpy.model.profile import Profile


@dataclass(frozen=True, slots=True)
class Problem(Serial):
    """Fixed-boundary equilibrium problem definition for one operator topology."""

    route: str
    coordinate: str
    profiles: dict[str, Profile]
    boundary: Boundary
    heat_input: np.ndarray
    current_input: np.ndarray
    nodes: str = "uniform"
    Ip: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """Normalize construction inputs into the canonical internal form."""

        route = str(self.route).upper()
        coordinate = str(self.coordinate).lower()
        nodes = str(self.nodes).lower()

        if coordinate not in ("rho", "psin"):
            raise ValueError(f"coordinate must be one of ('rho', 'psin'), got {self.coordinate!r}")
        if nodes not in ("uniform", "grid"):
            raise ValueError(f"nodes must be one of ('uniform', 'grid'), got {self.nodes!r}")

        object.__setattr__(self, "route", route)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "nodes", nodes)

        object.__setattr__(self, "profiles", _normalize_profiles(self.profiles))
        object.__setattr__(self, "boundary", _coerce_boundary(self.boundary))
        object.__setattr__(
            self, "heat_input", _readonly_array(_as_1d_array(self.heat_input, name="heat_input"))
        )
        object.__setattr__(
            self,
            "current_input",
            _readonly_array(_as_1d_array(self.current_input, name="current_input")),
        )
        object.__setattr__(self, "Ip", np.nan if self.Ip is None else float(self.Ip))
        object.__setattr__(self, "beta", np.nan if self.beta is None else float(self.beta))

        if self.heat_input.shape != self.current_input.shape:
            raise ValueError(
                f"heat_input and current_input must share the same shape, "
                f"got {self.heat_input.shape} and {self.current_input.shape}"
            )
        # Unit scaling is intentionally not done here.  Problem is the user-facing
        # definition; SourcePlan materializes engine-ready scaled arrays later.

    def __rich__(self) -> Tree:
        tree = Tree("[bold blue]Problem[/]")
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
            tree.add(f"Ip: {self.Ip:.3e}")
        if np.isfinite(self.beta):
            tree.add(f"beta: {self.beta:.3e}")
        tree.add(self.boundary)
        return tree

    def __str__(self) -> str:
        console = Console(
            color_system=None,
            force_terminal=False,
            width=120,
            record=True,
            soft_wrap=False,
        )
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> Self:
        """Create an equivalent detached problem definition."""

        return self.replace()

    def replace(self, **kwargs) -> Self:
        """Return a new problem with selected raw input fields replaced."""

        return dataclass_replace(self, **kwargs)

    @property
    def profile_coeffs(self) -> dict[str, np.ndarray | None]:
        """Compatibility view of active profile coefficients keyed by profile name."""

        return {name: profile.coeff for name, profile in self.profiles.items()}

    @property
    def a(self) -> float:
        return self.boundary.a

    @property
    def R0(self) -> float:
        return self.boundary.R0

    @property
    def Z0(self) -> float:
        return self.boundary.Z0

    @property
    def B0(self) -> float:
        return self.boundary.B0

    @property
    def ka(self) -> float:
        return self.boundary.ka

    @property
    def c_offsets(self) -> np.ndarray:
        return self.boundary.c_offsets

    @property
    def s_offsets(self) -> np.ndarray:
        return self.boundary.s_offsets


def _coerce_boundary(boundary: Boundary | dict[str, object]) -> Boundary:
    if isinstance(boundary, Boundary):
        return boundary
    boundary_type = type(boundary)
    if (
        boundary_type.__name__ == Boundary.__name__
        and boundary_type.__module__ == Boundary.__module__
    ):
        return Boundary(
            a=boundary.a,
            R0=boundary.R0,
            Z0=boundary.Z0,
            B0=boundary.B0,
            ka=boundary.ka,
            c_offsets=boundary.c_offsets,
            s_offsets=boundary.s_offsets,
        )
    if isinstance(boundary, dict):
        return Boundary(**boundary)
    raise TypeError(f"boundary must be Boundary or dict, got {type(boundary).__name__}")


def _normalize_profiles(profiles: dict[str, Profile]) -> dict[str, Profile]:
    normalized: dict[str, Profile] = {}
    for name, profile in profiles.items():
        if not isinstance(name, str):
            raise TypeError(f"profile names must be str, got {type(name).__name__}")
        if not isinstance(profile, Profile):
            raise TypeError(f"{name} profile must be Profile, got {type(profile).__name__}")
        normalized[name] = profile.copy()
    return normalized


def _as_1d_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _readonly_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.setflags(write=False)
    return arr
