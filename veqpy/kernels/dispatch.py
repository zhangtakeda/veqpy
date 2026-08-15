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
    if kernel_recipe.backend == "cxx":
        if topology.nodes == "explicit":
            raise NotImplementedError(
                "nodes='explicit' is currently supported only by backend='numba'"
            )
        if topology.coordinate == "rho":
            raise NotImplementedError(
                "coordinate='rho' is currently supported only by backend='numba'"
            )
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
    raise ValueError("_BuildPolicy backend selection supports backend='cxx' or backend='numba'")


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
