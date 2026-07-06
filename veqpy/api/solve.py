"""Solve one Kernel case through a short-lived handle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veqpy.kernels import Kernel
from veqpy.types import (
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)


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
