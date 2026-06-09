"""
Module: solver.solver_result

Role:
- Hold the result snapshot and statistics for one solve.

Public API:
- SolverResult

Notes:
- `SolverResult` is decoupled from caller input arrays.
- Does not own history management, case replacement, or equilibrium reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.text import Text
from rich.tree import Tree


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Describe the final result snapshot for one solve call."""

    x0: np.ndarray
    x: np.ndarray
    success: bool
    message: str
    residual_norm_final: float
    function_evaluations: int
    jacobian_evaluations: int
    iterations: int
    elapsed: float

    def __post_init__(self) -> None:
        """Copy packed state arrays to avoid sharing mutable memory with callers."""

        object.__setattr__(self, "x0", _as_1d_array(self.x0, name="x0"))
        object.__setattr__(self, "x", _as_1d_array(self.x, name="x"))

    def __rich__(self) -> Tree:
        tree = Tree("[bold blue]SolverResult[/]")
        tree.add(f"success: {self.success}")
        tree.add(f"message: {self.message}")
        tree.add(f"residual_norm: {self.residual_norm_final:.6e}")
        tree.add(f"function_evaluations: {self.function_evaluations}")
        tree.add(f"jacobian_evaluations: {self.jacobian_evaluations}")
        tree.add(f"iterations: {self.iterations}")
        tree.add(Text(f"elapsed: {(self.elapsed / 1000):.3f} [ms]"))
        tree.add(
            f"x0: shape={self.x0.shape}, min={float(np.min(self.x0)):.3f}, "
            f"max={float(np.max(self.x0)):.3f}"
        )
        tree.add(
            f"x: shape={self.x.shape}, min={float(np.min(self.x)):.3f}, "
            f"max={float(np.max(self.x)):.3f}"
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


def _as_1d_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize input into an independent one-dimensional float64 array copy."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy()
