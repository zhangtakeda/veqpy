"""
Module: veqpy.api

Role:
- Provide function-style ``build`` and ``solve`` entrypoints over ``veqpy.Kernel``.

Notes:
- This module imports only the public Kernel contract, not concrete backend modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veqpy.kernels import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)

__all__ = ["build", "solve"]


def build(
    *,
    topology: KernelTopology,
    recipe: KernelRecipe | None = None,
    config: KernelConfig | None = None,
    registry: object | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Kernel:
    """Create a kernel handle, cache its default config, and prepare its artifact."""

    kernel = Kernel(
        topology=topology,
        recipe=recipe,
        config=config,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    kernel.prepare(force=force, dry_run=dry_run)
    return kernel


def solve(
    boundary: KernelBoundary,
    source: KernelSource,
    *,
    topology: KernelTopology,
    config: KernelConfig | None = None,
    recipe: KernelRecipe | None = None,
    registry: object | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    case_name: str | None = None,
    **config_overrides: Any,
) -> SolveResult:
    """Prepare a short-lived kernel, solve one case, and close its private workspace."""

    kernel = Kernel(
        topology=topology,
        recipe=recipe,
        config=config,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    try:
        kernel.prepare(force=force, dry_run=False)
        return kernel.solve(boundary, source, case_name=case_name, **config_overrides)
    finally:
        kernel.close()
