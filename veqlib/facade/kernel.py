"""High-level Python handle for topology-specific VEQlib kernels.

This module owns the user-facing ``Kernel`` lifecycle: artifact resolution,
typed ``KernelBoundary``/``KernelSource``/``KernelConfig`` runtime calls, and
Python-owned result snapshots. External ``Operator`` adapters and benchmark
harnesses stay outside this ABI boundary.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veqlib.cxx_core.abi import (
    boundary_runtime_args,
    config_runtime_args,
    solve_result_from_native,
    source_runtime_args,
)
from veqlib.cxx_core.affinity import pinned_cpu
from veqlib.cxx_core.registry import KernelRegistry
from veqlib.cxx_core.solver import VEQlibSolver

from .source_semantics import MaterializedKernelSource, materialize_kernel_source
from .types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    config_with_overrides,
)

if TYPE_CHECKING:
    from veqpy.model import Equilibrium


class _CxxKernelImpl:
    """Stateful VEQlib kernel handle backed by one topology-specific artifact."""

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
        self.topology = topology
        self.recipe = KernelRecipe() if recipe is None else recipe
        if not isinstance(self.recipe, KernelRecipe):
            raise TypeError(f"recipe must be KernelRecipe, got {type(self.recipe).__name__}")
        self._validate_native_recipe(self.recipe)
        self.config = KernelConfig() if config is None else self._kernel_config(config)
        self.pin_cpu = pin_cpu
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self._solver: VEQlibSolver | None = None
        self.history: list[SolveResult] = []
        self.result: SolveResult | None = None
        self._last_boundary: KernelBoundary | None = None
        self._last_source: KernelSource | None = None

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> KernelPrepareResult:
        artifact = self._veqlib_solver().prepare(force=force, dry_run=dry_run)
        return KernelPrepareResult(
            backend=self.recipe.backend,
            topology=self.topology,
            recipe=self.recipe,
            x_size=self.x_size,
            residual_size=self.x_size,
            prepared=not dry_run,
            dry_run=dry_run,
            artifact=artifact,
        )

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        **config_overrides: Any,
    ) -> SolveResult:
        kernel_config = self._runtime_config(config, config_overrides)
        solver = self._set_runtime(boundary, source, kernel_config, case_name=case_name)
        self.result = solve_result_from_native(solver.solve_direct())
        self.history.append(self.result)
        return self.result

    # Raw numerical APIs use the handle default config to install the native
    # current-case context required before residual/JVP/Jacobian kernels run.
    def residual(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, source)
        return out

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._veqlib_solver().residual_var_into(packed_out, packed_x)

    def jvp(self, x: Any, v: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v, boundary, source)
        return out

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._veqlib_solver().jvp_into(packed_out, packed_x, packed_v)

    def jacobian(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x, boundary, source)
        return out

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        matrix_out = self._packed_output(out, (self.x_size, self.x_size), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._veqlib_solver().jacobian_into(matrix_out, packed_x)

    def build_equilibrium(self, x: Any | None = None) -> Equilibrium:
        if self._last_boundary is None or self._last_source is None:
            raise RuntimeError("build_equilibrium requires a previous Kernel runtime case")
        if x is None:
            if self.result is None:
                raise RuntimeError("build_equilibrium(x=None) requires a previous solve result")
            packed_x = self.result.x
        else:
            packed_x = self._packed_input(x, "x")
        from veqpy.kernels.numba_kernel.runtime import NumbaRuntime

        runtime = NumbaRuntime(self.topology)
        return runtime.build_equilibrium(packed_x, self._last_boundary, self._last_source)

    def clear(self) -> None:
        self.history.clear()
        self.result = None
        self._last_boundary = None
        self._last_source = None

    def close(self) -> None:
        if self._solver is not None:
            self._solver.close()
            self._solver = None

    def pinned(self) -> AbstractContextManager[None, bool | None]:
        """Return a scoped CPU pinning context for high-volume solve loops."""

        return pinned_cpu(self.pin_cpu)

    def _veqlib_solver(self) -> VEQlibSolver:
        if self._solver is None:
            self._solver = VEQlibSolver(
                self.topology,
                recipe=self.recipe,
                registry=self.registry,
                pin_cpu=self.pin_cpu,
            )
        return self._solver

    def _packed_input(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (self.x_size,):
            raise ValueError(f"{name} must have shape ({self.x_size},), got {array.shape}")
        return np.ascontiguousarray(array, dtype=np.float64)

    @staticmethod
    def _packed_output(out: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        if not isinstance(out, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if out.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if out.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
        if not out.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return out

    def _runtime_config(
        self,
        config: KernelConfig | None,
        overrides: dict[str, Any],
    ) -> KernelConfig:
        kernel_config = self.config if config is None else self._kernel_config(config)
        if overrides:
            kernel_config = config_with_overrides(kernel_config, **overrides)
        return kernel_config

    def _set_runtime(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
        *,
        case_name: str | None,
    ) -> VEQlibSolver:
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        materialized_source = materialize_kernel_source(self.topology, kernel_source)
        self._validate_runtime_case_adaptability(kernel_boundary, materialized_source)
        solver = self._veqlib_solver()
        solver.set_kernel_runtime(
            "" if materialized_source.case_name is None else materialized_source.case_name,
            *boundary_runtime_args(kernel_boundary),
            *source_runtime_args(materialized_source),
            *config_runtime_args(config, x_size=self.x_size),
        )
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        return solver

    @staticmethod
    def _validate_native_recipe(recipe: KernelRecipe) -> None:
        if recipe.backend != "cxx":
            raise ValueError("VEQlib native Kernel requires KernelRecipe backend='cxx'")

    @staticmethod
    def _kernel_config(config: KernelConfig) -> KernelConfig:
        if not isinstance(config, KernelConfig):
            raise TypeError(f"config must be KernelConfig, got {type(config).__name__}")
        return config

    @staticmethod
    def _kernel_boundary(boundary: KernelBoundary) -> KernelBoundary:
        if not isinstance(boundary, KernelBoundary):
            raise TypeError(f"boundary must be KernelBoundary, got {type(boundary).__name__}")
        return boundary

    @staticmethod
    def _kernel_source(source: KernelSource, *, case_name: str | None) -> KernelSource:
        if not isinstance(source, KernelSource):
            raise TypeError(f"source must be KernelSource, got {type(source).__name__}")
        if case_name is None:
            return source
        return KernelSource(
            heat_profile=source.heat_profile,
            current_profile=source.current_profile,
            Ip=source.Ip,
            beta=source.beta,
            case_name=case_name,
        )

    def _validate_runtime_case_adaptability(
        self,
        boundary: KernelBoundary,
        source: MaterializedKernelSource,
    ) -> None:
        topology = self.topology
        max_offsets = topology.M_max + 1
        for name, values in (
            ("c_offsets", boundary.c_offsets),
            ("s_offsets", boundary.s_offsets),
        ):
            if values.size > max_offsets:
                raise ValueError(
                    "case does not match kernel topology: "
                    f"{name} length must be at most M_max + 1 ({max_offsets}), "
                    f"got {values.size}"
                )


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
