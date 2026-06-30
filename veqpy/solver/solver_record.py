"""
Module: solver.solver_record

Role:
- Hold the history snapshot associated with one solve.

Public API:
- SolverRecord

Notes:
- `SolverRecord` only packages problem, config, and result snapshots.
- Does not execute solves, packed codecs, or numerical kernel updates.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.tree import Tree

from veqpy.model import Problem
from veqpy.solver.solver_config import SolverConfig
from veqpy.solver.solver_result import SolverResult


@dataclass(frozen=True, slots=True)
class SolverRecord:
    """Describe the immutable history snapshot after one completed solve."""

    problem_snapshot: Problem
    config_snapshot: SolverConfig
    result_snapshot: SolverResult

    def __rich__(self) -> Tree:
        tree = Tree("[bold blue]SolverRecord[/]")
        tree.add(self.problem_snapshot)
        tree.add(self.config_snapshot)
        tree.add(self.result_snapshot)
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
