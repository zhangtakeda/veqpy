"""
Module: veqpy.api

Role:
- Provide function-style entrypoints over ``veqpy.Kernel`` and ``KernelBoundary``.

Notes:
- This module imports only the public Kernel contract, not concrete backend modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from veqpy.kernels import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    ParetoResult,
    SolveResult,
)

__all__ = ["build", "fit", "pareto", "solve"]


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


def fit(
    boundary: KernelBoundary,
    *,
    backend: str = "numba",
    method: str | None = None,
    c_order: int | None = None,
    s_order: int | None = None,
    maxtol: float | None = None,
) -> KernelBoundary:
    """Return a parameterized boundary by explicitly fitting stored R/Z points."""

    return boundary.fit(
        backend=backend,
        method=method,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
    )


def pareto(
    boundary: KernelBoundary,
    source: KernelSource,
    *,
    topology: KernelTopology,
    candidates: Sequence[object] | object,
    config: KernelConfig | None = None,
    recipe: KernelRecipe | None = None,
    registry: object | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    case_name: str | None = None,
    reference: SolveResult | None = None,
    target: str = "counts",
    metric: str = "rms",
    **config_overrides: Any,
) -> ParetoResult:
    """Prepare a short-lived kernel, evaluate Pareto candidates, and close it."""

    kernel_recipe = KernelRecipe(backend="numba", layout="degree") if recipe is None else recipe
    kernel = Kernel(
        topology=topology,
        recipe=kernel_recipe,
        config=config,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    try:
        kernel.prepare(force=force, dry_run=False)
        return kernel.pareto(
            boundary,
            source,
            candidates=candidates,
            config=config,
            case_name=case_name,
            reference=reference,
            target=target,
            metric=metric,
            **config_overrides,
        )
    finally:
        kernel.close()
