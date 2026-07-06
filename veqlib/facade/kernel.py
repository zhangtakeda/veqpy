"""High-level Python handle for topology-specific Kernel backends."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.kernels.cxx_kernel.kernel import _CxxKernelImpl
from veqpy.kernels.cxx_kernel.registry import KernelRegistry
from veqpy.types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)

if TYPE_CHECKING:
    from veqpy.model import Equilibrium


class Kernel:
    """Backend-neutral public Kernel wrapper selected by ``KernelRecipe.backend``."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        recipe: KernelRecipe | None = None,
        config: KernelConfig | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
    ) -> None:
        self._impl = _make_kernel_impl(
            topology=topology,
            recipe=recipe,
            config=config,
            registry=registry,
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )

    @property
    def topology(self) -> KernelTopology:
        return self._impl.topology

    @property
    def recipe(self) -> KernelRecipe:
        return self._impl.recipe

    @property
    def config(self) -> KernelConfig:
        return self._impl.config

    @property
    def history(self) -> list[SolveResult]:
        return self._impl.history

    @property
    def result(self) -> SolveResult | None:
        return self._impl.result

    @property
    def x_size(self) -> int:
        return self._impl.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> KernelPrepareResult:
        return self._impl.prepare(force=force, dry_run=dry_run)

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        **config_overrides: Any,
    ) -> SolveResult:
        return self._impl.solve(
            boundary,
            source,
            config=config,
            case_name=case_name,
            **config_overrides,
        )

    def residual(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        return self._impl.residual(x, boundary, source)

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        self._impl.residual_into(out, x, boundary, source)

    def jvp(self, x: Any, v: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        return self._impl.jvp(x, v, boundary, source)

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        self._impl.jvp_into(out, x, v, boundary, source)

    def jacobian(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        return self._impl.jacobian(x, boundary, source)

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        self._impl.jacobian_into(out, x, boundary, source)

    def build_equilibrium(self, x: Any | None = None) -> Equilibrium:
        return self._impl.build_equilibrium(x)

    def clear(self) -> None:
        self._impl.clear()

    def close(self) -> None:
        self._impl.close()

    def pinned(self) -> AbstractContextManager[None, bool | None]:
        return self._impl.pinned()


def _make_kernel_impl(
    *,
    topology: KernelTopology,
    recipe: KernelRecipe | None,
    config: KernelConfig | None,
    registry: KernelRegistry | None,
    cache_root: Path | None,
    source_dir: Path | None,
    pin_cpu: bool | int | None,
) -> Any:
    kernel_recipe = KernelRecipe() if recipe is None else recipe
    if not isinstance(kernel_recipe, KernelRecipe):
        raise TypeError(f"recipe must be KernelRecipe, got {type(kernel_recipe).__name__}")
    if kernel_recipe.backend == "cxx":
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
    registry: KernelRegistry | None,
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


def build(
    *,
    topology: KernelTopology,
    recipe: KernelRecipe | None = None,
    config: KernelConfig | None = None,
    registry: KernelRegistry | None = None,
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
    registry: KernelRegistry | None = None,
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
