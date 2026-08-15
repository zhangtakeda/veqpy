"""High-level VEQ build and one-shot solve entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fusionprime_base import Plasma

from .module import VEQ, VEQRecord

__all__ = ["build", "solve"]


def build(
    *,
    topology: Mapping[str, Any],
    solver: Mapping[str, Any] | None = None,
    backend: str = "numba",
    artifact_dir: str | Path | None = None,
    cpu_affinity: bool | int | None = None,
    rebuild: bool = False,
    materialize: bool = True,
    verbose: bool = True,
    report: bool = False,
    report_dir: str | Path | None = None,
) -> VEQ:
    """Build and prepare a reusable VEQ Module from ordinary mappings."""

    return VEQ(
        topology=topology,
        solver=solver,
        backend=backend,
        artifact_dir=artifact_dir,
        cpu_affinity=cpu_affinity,
        rebuild=rebuild,
        materialize=materialize,
        verbose=verbose,
        report=report,
        report_dir=report_dir,
    )


def solve(
    *,
    plasma: Plasma,
    topology: Mapping[str, Any],
    solver: Mapping[str, Any] | None = None,
    backend: str = "numba",
    artifact_dir: str | Path | None = None,
    cpu_affinity: bool | int | None = None,
    rebuild: bool = False,
    materialize: bool = True,
    verbose: bool = True,
    report: bool = False,
    report_dir: str | Path | None = None,
    solver_override: Mapping[str, Any] | None = None,
) -> VEQRecord:
    """Build a short-lived Module, solve one frozen Plasma, and close it."""

    module = build(
        topology=topology,
        solver=solver,
        backend=backend,
        artifact_dir=artifact_dir,
        cpu_affinity=cpu_affinity,
        rebuild=rebuild,
        materialize=materialize,
        verbose=verbose,
        report=report,
        report_dir=report_dir,
    )
    try:
        return module.solve(
            plasma=plasma,
            solver=solver_override,
            materialize=materialize,
            verbose=verbose,
            report=report,
            report_dir=report_dir,
        )
    finally:
        module.close()
