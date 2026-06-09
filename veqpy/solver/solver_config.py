"""
Module: solver.solver_config

Role:
- Describe the configuration for one nonlinear solve.
- Register mappings from solver method names to SciPy optimize callables.

Public API:
- SolverConfig
- ROOT_METHODS
- LEAST_SQUARES_METHODS
- SUPPORTED_METHODS

Notes:
- `SolverConfig` only stores configuration and does not execute solves.
- Does not own history storage or residual evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from rich.console import Console
from rich.tree import Tree
from scipy.optimize import least_squares, root

from veqpy.solver.residual_scale import DEFAULT_RESIDUAL_NORMALIZATION

OptimizeMethod = Callable[..., Any]


def _root_method(method: str) -> OptimizeMethod:
    # Store thin wrappers instead of scipy callables directly so the registry
    # presents one uniform signature to Solver.
    def run(fun, x0, **kwargs):
        return root(fun, x0, method=method, **kwargs)

    return run


def _least_squares_method(method: str) -> OptimizeMethod:
    # least_squares uses a different SciPy entrypoint but remains a Solver
    # method choice once wrapped with the same callable shape.
    def run(fun, x0, **kwargs):
        return least_squares(fun, x0, method=method, **kwargs)

    return run


ROOT_METHODS: dict[str, OptimizeMethod] = {
    "hybr": _root_method("hybr"),
}

LEAST_SQUARES_METHODS: dict[str, OptimizeMethod] = {
    "lm": _least_squares_method("lm"),
    "trf": _least_squares_method("trf"),
}

SUPPORTED_METHODS: dict[str, OptimizeMethod] = {
    **ROOT_METHODS,
    **LEAST_SQUARES_METHODS,
}

DEFAULT_VARIATIONAL_METHOD = "hybr"
DEFAULT_COLLOCATION_METHOD = "lm"
DEFAULT_VARIATIONAL_FALLBACK_METHODS = ("lm",)
SUPPORTED_INITIAL_POLICIES = frozenset(("zeros", "warm", "homothetic"))


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Describe Solver defaults and per-solve overrides."""

    method: str | None = None
    max_residual: float = 1e-6
    max_evaluations: int = 1000
    initial_policy: str | None = None
    enable_fallback: bool = True
    fallback_methods: tuple[str, ...] | list[str] | None = field(default=None)
    enable_verbose: bool = False
    enable_history: bool = True
    # Passing None resolves to the same package default; no second normalization
    # default is hidden behind the nullable form.
    residual_normalization: str | None = DEFAULT_RESIDUAL_NORMALIZATION
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e6
    residual_normalization_huber_tau: float = 3.0
    residual_normalization_probe_count: int = 4
    residual_normalization_probe_step: float = 1.0e-6
    residual_normalization_sensitivity_lambda: float = 0.5

    enable_collocation: bool = False
    collocation_method: str = DEFAULT_COLLOCATION_METHOD
    collocation_weight: float = 1.0
    collocation_max_residual: float | None = None
    collocation_max_evaluations: int | None = None

    def __post_init__(self) -> None:
        """Validate method names and fallback-related parameters."""

        method = DEFAULT_VARIATIONAL_METHOD if self.method is None else str(self.method)
        fallback_methods = (
            DEFAULT_VARIATIONAL_FALLBACK_METHODS
            if self.fallback_methods is None
            else self.fallback_methods
        )
        fallback_methods = tuple(str(method_name) for method_name in fallback_methods)
        deduped_fallback_methods: list[str] = []
        seen: set[str] = set()
        for method_name in fallback_methods:
            if method_name in seen:
                continue
            seen.add(method_name)
            deduped_fallback_methods.append(method_name)
        # Fallback order is user-visible in SolverResult messages, so de-dupe
        # without sorting.

        if method not in SUPPORTED_METHODS:
            supported = ", ".join(SUPPORTED_METHODS)
            raise ValueError(f"Unsupported solver method {method!r}; supported: {supported}.")
        collocation_method = str(self.collocation_method)
        if collocation_method not in LEAST_SQUARES_METHODS:
            supported = ", ".join(LEAST_SQUARES_METHODS)
            raise ValueError(
                f"Unsupported collocation_method {collocation_method!r}; supported: {supported}."
            )
        collocation_weight = float(self.collocation_weight)
        if not isfinite(collocation_weight) or collocation_weight < 0.0 or collocation_weight > 1.0:
            raise ValueError(
                "collocation_weight must be a finite float in [0, 1]; "
                f"got {self.collocation_weight!r}."
            )
        unsupported_fallbacks = [
            method_name
            for method_name in deduped_fallback_methods
            if method_name not in SUPPORTED_METHODS
        ]
        if unsupported_fallbacks:
            supported = ", ".join(SUPPORTED_METHODS)
            unsupported = ", ".join(repr(method_name) for method_name in unsupported_fallbacks)
            raise ValueError(
                f"Unsupported fallback solver method(s): {unsupported}. "
                f"Supported methods are: {supported}."
            )
        max_residual = float(self.max_residual)
        max_evaluations = int(self.max_evaluations)
        initial_policy = None if self.initial_policy is None else str(self.initial_policy).lower()
        if initial_policy == "zero":
            initial_policy = "zeros"
        if initial_policy == "warmstart":
            initial_policy = "warm"
        # Accept the common aliases above, then store only the canonical policy
        # strings consumed by Solver._build_initial_state.
        if initial_policy is not None and initial_policy not in SUPPORTED_INITIAL_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_INITIAL_POLICIES))
            raise ValueError(
                f"Unsupported initial_policy {self.initial_policy!r}; supported: {supported}."
            )
        if not isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError(
                f"SolverConfig.max_residual must be a positive finite float, "
                f"got {self.max_residual!r}."
            )
        if max_evaluations < 0:
            raise ValueError(
                f"SolverConfig.max_evaluations must be non-negative; got {self.max_evaluations!r}."
            )
        collocation_max_residual = (
            None if self.collocation_max_residual is None else float(self.collocation_max_residual)
        )
        collocation_max_evaluations = (
            None
            if self.collocation_max_evaluations is None
            else int(self.collocation_max_evaluations)
        )
        if collocation_max_residual is not None and (
            not isfinite(collocation_max_residual) or collocation_max_residual <= 0.0
        ):
            raise ValueError(
                "collocation_max_residual must be positive finite; "
                f"got {self.collocation_max_residual!r}."
            )
        if collocation_max_evaluations is not None and collocation_max_evaluations < 0:
            raise ValueError(
                "collocation_max_evaluations must be non-negative; "
                f"got {self.collocation_max_evaluations!r}."
            )
        residual_normalization = (
            DEFAULT_RESIDUAL_NORMALIZATION
            if self.residual_normalization is None
            else str(self.residual_normalization).strip().lower().replace("_", "-")
        )
        # Hyphen-normalized mode names keep CLI/config spelling flexible while
        # residual_scale receives a stable token.
        residual_normalization_floor = float(self.residual_normalization_floor)
        residual_normalization_max_ratio = float(self.residual_normalization_max_ratio)
        residual_normalization_huber_tau = float(self.residual_normalization_huber_tau)
        residual_normalization_probe_count = int(self.residual_normalization_probe_count)
        residual_normalization_probe_step = float(self.residual_normalization_probe_step)
        residual_normalization_sensitivity_lambda = float(
            self.residual_normalization_sensitivity_lambda
        )
        if not isfinite(residual_normalization_floor) or residual_normalization_floor <= 0.0:
            raise ValueError(
                "SolverConfig.residual_normalization_floor must be positive finite; "
                f"got {self.residual_normalization_floor!r}."
            )
        if not isfinite(residual_normalization_max_ratio) or residual_normalization_max_ratio < 1.0:
            raise ValueError(
                "SolverConfig.residual_normalization_max_ratio must be finite and >= 1; "
                f"got {self.residual_normalization_max_ratio!r}."
            )
        if not isfinite(residual_normalization_huber_tau) or residual_normalization_huber_tau < 0.0:
            raise ValueError(
                "SolverConfig.residual_normalization_huber_tau must be finite and >= 0; "
                f"got {self.residual_normalization_huber_tau!r}."
            )
        if residual_normalization_probe_count < 0:
            raise ValueError(
                "SolverConfig.residual_normalization_probe_count must be non-negative; "
                f"got {self.residual_normalization_probe_count!r}."
            )
        if (
            not isfinite(residual_normalization_probe_step)
            or residual_normalization_probe_step <= 0.0
        ):
            raise ValueError(
                "SolverConfig.residual_normalization_probe_step must be positive finite; "
                f"got {self.residual_normalization_probe_step!r}."
            )
        if not isfinite(residual_normalization_sensitivity_lambda) or (
            residual_normalization_sensitivity_lambda < 0.0
        ):
            raise ValueError(
                "SolverConfig.residual_normalization_sensitivity_lambda must be finite and >= 0; "
                f"got {self.residual_normalization_sensitivity_lambda!r}."
            )
        object.__setattr__(self, "method", method)
        # The dataclass is frozen to make solve configs snapshot-safe for history
        # records.  __post_init__ writes validated canonical values exactly once.
        object.__setattr__(self, "enable_collocation", bool(self.enable_collocation))
        object.__setattr__(self, "collocation_method", collocation_method)
        object.__setattr__(self, "collocation_weight", collocation_weight)
        object.__setattr__(self, "collocation_max_residual", collocation_max_residual)
        object.__setattr__(self, "collocation_max_evaluations", collocation_max_evaluations)
        object.__setattr__(self, "max_residual", max_residual)
        object.__setattr__(self, "max_evaluations", max_evaluations)
        object.__setattr__(self, "initial_policy", initial_policy)
        object.__setattr__(self, "enable_fallback", bool(self.enable_fallback))
        object.__setattr__(self, "fallback_methods", tuple(deduped_fallback_methods))
        object.__setattr__(self, "residual_normalization", residual_normalization)
        object.__setattr__(self, "residual_normalization_floor", residual_normalization_floor)
        object.__setattr__(
            self, "residual_normalization_max_ratio", residual_normalization_max_ratio
        )
        object.__setattr__(
            self, "residual_normalization_huber_tau", residual_normalization_huber_tau
        )
        object.__setattr__(
            self, "residual_normalization_probe_count", residual_normalization_probe_count
        )
        object.__setattr__(
            self, "residual_normalization_probe_step", residual_normalization_probe_step
        )
        object.__setattr__(
            self,
            "residual_normalization_sensitivity_lambda",
            residual_normalization_sensitivity_lambda,
        )

    def __rich__(self) -> Tree:
        tree = Tree("[bold blue]SolverConfig[/]")
        tree.add(f"method: {self.method}")
        tree.add(f"enable_collocation: {self.enable_collocation}")
        if self.enable_collocation:
            tree.add(f"collocation_method: {self.collocation_method}")
            tree.add(f"collocation_weight: {self.collocation_weight:.6g}")
            if self.collocation_max_residual is not None:
                tree.add(f"collocation_max_residual: {self.collocation_max_residual:.6g}")
            if self.collocation_max_evaluations is not None:
                tree.add(f"collocation_max_evaluations: {self.collocation_max_evaluations}")
        tree.add(f"max_residual: {self.max_residual:.6g}")
        tree.add(f"max_evaluations: {self.max_evaluations}")
        tree.add(f"initial_policy: {self.initial_policy}")
        tree.add(f"enable_fallback: {self.enable_fallback}")
        if self.enable_fallback:
            tree.add(f"fallback_methods: {list(self.fallback_methods)}")
        tree.add(f"enable_verbose: {self.enable_verbose}")
        tree.add(f"enable_history: {self.enable_history}")
        tree.add(f"residual_normalization: {self.residual_normalization}")
        if self.residual_normalization != "none":
            tree.add(f"residual_normalization_floor: {self.residual_normalization_floor:.6g}")
            tree.add(
                f"residual_normalization_max_ratio: {self.residual_normalization_max_ratio:.6g}"
            )
            tree.add(
                f"residual_normalization_huber_tau: {self.residual_normalization_huber_tau:.6g}"
            )
            tree.add(
                f"residual_normalization_probe_count: {self.residual_normalization_probe_count}"
            )
            tree.add(
                f"residual_normalization_probe_step: {self.residual_normalization_probe_step:.6g}"
            )
            tree.add(
                "residual_normalization_sensitivity_lambda: "
                f"{self.residual_normalization_sensitivity_lambda:.6g}"
            )
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
