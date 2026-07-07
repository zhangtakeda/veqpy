"""
Module: veqpy.kernels.cxx_kernel.kernel

Role:
- Implement the private Cxx backend behind the public ``Kernel`` wrapper.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.kernels.abi.source_semantics import MaterializedKernelSource, materialize_kernel_source
from veqpy.kernels.types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    config_with_overrides,
)

from .affinity import pinned_cpu
from .native_abi import (
    boundary_runtime_args,
    config_runtime_args,
    solve_result_from_native,
    source_runtime_args,
)
from .registry import KernelRegistry
from .solver import CxxSolver

if TYPE_CHECKING:
    from veqpy.model import Equilibrium


class _CxxKernelImpl:
    """Stateful Kernel handle backed by one topology-specific artifact."""

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
        self._solver: CxxSolver | None = None
        self.history: list[SolveResult] = []
        self.result: SolveResult | None = None
        self._last_boundary: KernelBoundary | None = None
        self._last_source: KernelSource | None = None

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> KernelPrepareResult:
        artifact = self._cxx_solver().prepare(force=force, dry_run=dry_run)
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
        self._cxx_solver().residual_var_into(packed_out, packed_x)

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
        self._cxx_solver().jvp_into(packed_out, packed_x, packed_v)

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
        self._cxx_solver().jacobian_into(matrix_out, packed_x)

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

    def _cxx_solver(self) -> CxxSolver:
        if self._solver is None:
            self._solver = CxxSolver(
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
    ) -> CxxSolver:
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        materialized_source = materialize_kernel_source(self.topology, kernel_source)
        self._validate_runtime_case_adaptability(kernel_boundary, materialized_source)
        solver = self._cxx_solver()
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
            raise ValueError("Cxx Kernel requires KernelRecipe backend='cxx'")

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
        max_cosine_offsets = topology.M_max + 1
        if boundary.c_offsets.size > max_cosine_offsets:
            raise ValueError(
                "case does not match kernel topology: "
                f"c_offsets length must be at most M_max + 1 ({max_cosine_offsets}), "
                f"got {boundary.c_offsets.size}"
            )
        max_sine_offsets = topology.M_max
        if len(boundary.s_offsets) > max_sine_offsets:
            raise ValueError(
                "case does not match kernel topology: "
                f"s_offsets length must be at most M_max ({max_sine_offsets}), "
                f"got {len(boundary.s_offsets)}"
            )
