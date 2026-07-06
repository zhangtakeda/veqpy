"""Build a reusable Kernel handle."""

from __future__ import annotations

from pathlib import Path

from veqpy.kernels import Kernel
from veqpy.types import KernelConfig, KernelRecipe, KernelTopology


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
