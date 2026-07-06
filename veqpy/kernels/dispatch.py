"""Private Kernel backend dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veqpy.kernels.types import KernelConfig, KernelRecipe, KernelTopology


def _make_kernel_impl(
    *,
    topology: KernelTopology,
    recipe: KernelRecipe | None,
    config: KernelConfig | None,
    registry: object | None,
    cache_root: Path | None,
    source_dir: Path | None,
    pin_cpu: bool | int | None,
) -> Any:
    kernel_recipe = KernelRecipe() if recipe is None else recipe
    if not isinstance(kernel_recipe, KernelRecipe):
        raise TypeError(f"recipe must be KernelRecipe, got {type(kernel_recipe).__name__}")
    if kernel_recipe.backend == "cxx":
        from veqpy.kernels.cxx_kernel.kernel import _CxxKernelImpl

        return _CxxKernelImpl(
            topology=topology,
            recipe=kernel_recipe,
            config=config,
            registry=registry,
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
    if kernel_recipe.backend == "numba":
        _validate_numba_options(
            registry=registry,
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        from veqpy.kernels.numba_kernel.kernel import _NumbaKernelImpl

        return _NumbaKernelImpl(topology=topology, recipe=kernel_recipe, config=config)
    raise ValueError("KernelRecipe backend selection supports backend='cxx' or backend='numba'")


def _validate_numba_options(
    *,
    registry: object | None,
    cache_root: Path | None,
    source_dir: Path | None,
    pin_cpu: bool | int | None,
) -> None:
    invalid = []
    if registry is not None:
        invalid.append("registry")
    if cache_root is not None:
        invalid.append("cache_root")
    if source_dir is not None:
        invalid.append("source_dir")
    if pin_cpu is not None:
        invalid.append("pin_cpu")
    if invalid:
        names = ", ".join(invalid)
        raise ValueError(f"backend='numba' does not accept native-only option(s): {names}")
