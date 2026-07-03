"""Explicit Numba backend handle for the VEQPy facade."""

from __future__ import annotations

from veqlib.facade import KernelConfig, KernelRecipe, KernelTopology
from veqpy.kernel import NumbaKernel

Kernel = NumbaKernel


def build(
    *,
    topology: KernelTopology,
    recipe: KernelRecipe | None = None,
    config: KernelConfig | None = None,
) -> NumbaKernel:
    """Create an explicit Numba ``Kernel`` handle."""

    return NumbaKernel(topology=topology, recipe=recipe, config=config)
