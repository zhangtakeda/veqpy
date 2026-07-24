"""
Module: veqpy.kernels.kernel

Role:
- Provide the backend-neutral public ``Kernel`` handle.

Notes:
- The wrapper owns lifecycle, validation, and result shape; private backends own execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.kernels.initial import KernelInitial
from veqpy.kernels.pareto import ParetoResult
from veqpy.kernels.types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)

from .dispatch import _make_kernel_impl

if TYPE_CHECKING:
    from veqpy.model import Equilibrium, Grid


class Kernel:
    """Backend-neutral public Kernel wrapper selected by ``KernelRecipe.backend``."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        recipe: KernelRecipe | None = None,
        config: KernelConfig | None = None,
        registry: object | None = None,
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

    def variant(
        self,
        *,
        h_count: int | None = None,
        v_count: int | None = None,
        kappa_count: int | None = None,
        psin_count: int | None = None,
        F_count: int | None = None,
        c_counts: tuple[int, ...] | None = None,
        s_counts: tuple[int, ...] | None = None,
    ) -> "Kernel":
        self._impl.variant(
            h_count=h_count,
            v_count=v_count,
            kappa_count=kappa_count,
            psin_count=psin_count,
            F_count=F_count,
            c_counts=c_counts,
            s_counts=s_counts,
        )
        return self

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        x0: KernelInitial | None = None,
        **config_overrides: Any,
    ) -> SolveResult:
        return self._impl.solve(
            boundary,
            source,
            config=config,
            case_name=case_name,
            x0=x0,
            **config_overrides,
        )

    def pareto(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        candidates: Sequence[object] | object,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        reference: SolveResult | None = None,
        target: str = "counts",
        metric: str = "rms",
        **config_overrides: Any,
    ) -> ParetoResult:
        return self._impl.pareto(
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

    def build_equilibrium(
        self,
        x: Any | None = None,
        *,
        grid: Grid | None = None,
    ) -> Equilibrium:
        """Materialize the latest or supplied state on the requested output grid."""
        return self._impl.build_equilibrium(x, grid=grid)

    def clear(self) -> None:
        self._impl.clear()

    def close(self) -> None:
        self._impl.close()

    def pinned(self) -> AbstractContextManager[None, bool | None]:
        return self._impl.pinned()
