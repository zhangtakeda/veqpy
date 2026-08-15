"""
Module: veqpy.kernels.dispatch

Role:
- Select the private backend implementation requested by ``_BuildPolicy.backend``.

Notes:
- Dispatch stays behind ``veqpy.kernels.Kernel`` and is not part of the public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veqpy.kernels.types import KernelTopology, _BackendConfig, _BuildPolicy


def _make_kernel_impl(
    *,
    topology: KernelTopology,
    recipe: _BuildPolicy | None,
    config: _BackendConfig | None,
    registry: object | None,
    cache_root: Path | None,
    source_dir: Path | None,
    pin_cpu: bool | int | None,
) -> Any:
    kernel_recipe = _BuildPolicy() if recipe is None else recipe
    if not isinstance(kernel_recipe, _BuildPolicy):
        raise TypeError(f"recipe must be _BuildPolicy, got {type(kernel_recipe).__name__}")
    if kernel_recipe.backend in {"cxx-strict", "cxx-relaxed"}:
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
        from veqpy.kernels.numba_kernel.kernel import _NumbaKernelImpl

        return _NumbaKernelImpl(topology=topology, recipe=kernel_recipe, config=config)
    raise ValueError("_BuildPolicy backend selection supports numba, cxx-strict, or cxx-relaxed")
