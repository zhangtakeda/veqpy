"""
Module: veqpy.kernels.cxx_kernel.kernel

Role:
- Implement the private Cxx backend behind the public ``Kernel`` wrapper.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE, source_driver_for
from veqpy.kernels.abi.source_semantics import _MaterializedSource, materialize_kernel_source
from veqpy.kernels.boundary_materialization import materialize_kernel_boundary
from veqpy.kernels.types import (
    KernelTopology,
    _BackendConfig,
    _BoundaryCase,
    _BuildPolicy,
    _PrepareDiagnostics,
    _SolveSnapshot,
    _SourceCase,
    config_with_overrides,
)

from ..numba_kernel.state import coerce_initial_state
from .affinity import pinned_cpu
from .native_abi import (
    boundary_runtime_args,
    config_runtime_args,
    solve_result_from_native,
    source_runtime_args,
)
from .registry import KernelRegistry
from .solver import CxxSolver
from .validation import validate_supported_for_cxx_backend

if TYPE_CHECKING:
    from fusionprime_base import Equilibrium

    from veqpy.numerics.grid import Grid


class _CxxKernelImpl:
    """Stateful Kernel handle backed by one topology-specific artifact."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        recipe: _BuildPolicy | None = None,
        config: _BackendConfig | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
    ) -> None:
        self.topology = topology
        self.native_topology = _native_topology(topology)
        self.recipe = (
            _BuildPolicy(backend="cxx-relaxed", build="release-relaxed")
            if recipe is None
            else recipe
        )
        if not isinstance(self.recipe, _BuildPolicy):
            raise TypeError(f"recipe must be _BuildPolicy, got {type(self.recipe).__name__}")
        self._validate_native_recipe(self.recipe)
        validate_supported_for_cxx_backend(self.native_topology)
        self.config = _BackendConfig() if config is None else self._kernel_config(config)
        self.pin_cpu = pin_cpu
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self._solver: CxxSolver | None = None
        self._last_snapshot: _SolveSnapshot | None = None
        self._last_boundary: _BoundaryCase | None = None
        self._last_source: _SourceCase | None = None
        self._last_native_source: _SourceCase | None = None

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> _PrepareDiagnostics:
        artifact = self._cxx_solver().prepare(force=force, dry_run=dry_run)
        return _PrepareDiagnostics(
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
        boundary: _BoundaryCase,
        source: _SourceCase,
        *,
        config: _BackendConfig | None = None,
        case_name: str | None = None,
        x0: np.ndarray | None = None,
        **config_overrides: Any,
    ) -> _SolveSnapshot:
        elapsed_started = perf_counter()
        kernel_config = self._runtime_config(config, config_overrides)
        solver = self._set_runtime(boundary, source, kernel_config, case_name=case_name)
        if x0 is not None:
            solver.set_initial_state(coerce_initial_state(x0, self.x_size))
        preprocess_ms = (perf_counter() - elapsed_started) * 1000.0
        native_value = solver.solve_direct()
        postprocess_started = perf_counter()
        result = solve_result_from_native(
            native_value,
            preprocess_ms=preprocess_ms,
            solver_ms=float(native_value[0]),
        )
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
        self._last_snapshot = replace(
            result,
            elapsed_ms=(perf_counter() - elapsed_started) * 1000.0,
            postprocess_ms=postprocess_ms,
        )
        return self._last_snapshot

    # Raw numerical APIs use the handle default config to install the native
    # current-case context required before residual/JVP/Jacobian kernels run.
    def residual(self, x: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, source)
        return out

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._cxx_solver().residual_var_into(packed_out, packed_x)

    def jvp(self, x: Any, v: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v, boundary, source)
        return out

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._cxx_solver().jvp_into(packed_out, packed_x, packed_v)

    def jacobian(self, x: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x, boundary, source)
        return out

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
    ) -> None:
        matrix_out = self._packed_output(out, (self.x_size, self.x_size), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, source, self.config, case_name=None)
        self._cxx_solver().jacobian_into(matrix_out, packed_x)

    def build_equilibrium(
        self,
        x: Any | None = None,
        *,
        grid: Grid | None = None,
    ) -> Equilibrium:
        if self._last_boundary is None or self._last_source is None:
            raise RuntimeError("build_equilibrium requires a previous Kernel runtime case")
        if x is None:
            if self._last_snapshot is None:
                raise RuntimeError("build_equilibrium(x=None) requires a previous solve result")
            packed_x = self._last_snapshot.x
        else:
            packed_x = self._packed_input(x, "x")
        from veqpy.kernels.numba_kernel.runtime import NumbaRuntime

        runtime = NumbaRuntime(self.topology)
        return runtime.build_equilibrium(
            packed_x,
            self._last_boundary,
            self._last_source,
            grid=grid,
        )

    def clear(self) -> None:
        self._last_snapshot = None
        self._last_boundary = None
        self._last_source = None
        self._last_native_source = None

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
                self.native_topology,
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
        config: _BackendConfig | None,
        overrides: dict[str, Any],
    ) -> _BackendConfig:
        kernel_config = self.config if config is None else self._kernel_config(config)
        if overrides:
            kernel_config = config_with_overrides(kernel_config, **overrides)
        return kernel_config

    def _set_runtime(
        self,
        boundary: _BoundaryCase,
        source: _SourceCase,
        config: _BackendConfig,
        *,
        case_name: str | None,
    ) -> CxxSolver:
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        source_count = int(kernel_source.pressure_profile.size)
        native_source = _native_source_case(
            self.topology,
            kernel_source,
            native_topology=self.native_topology,
        )
        materialized_source = materialize_kernel_source(self.native_topology, native_source)
        self._validate_runtime_case_adaptability(kernel_boundary, materialized_source)
        solver = self._cxx_solver()
        solver.set_kernel_runtime(
            "" if materialized_source.case_name is None else materialized_source.case_name,
            *boundary_runtime_args(kernel_boundary),
            *source_runtime_args(materialized_source, source_count),
            *config_runtime_args(config, x_size=self.x_size),
        )
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        self._last_native_source = native_source
        return solver

    @staticmethod
    def _validate_native_recipe(recipe: _BuildPolicy) -> None:
        if recipe.backend not in {"cxx-strict", "cxx-relaxed"}:
            raise ValueError("Cxx Kernel requires a strict or relaxed Cxx recipe")

    @staticmethod
    def _kernel_config(config: _BackendConfig) -> _BackendConfig:
        if not isinstance(config, _BackendConfig):
            raise TypeError(f"config must be _BackendConfig, got {type(config).__name__}")
        return config

    @staticmethod
    def _kernel_boundary(boundary: _BoundaryCase) -> _BoundaryCase:
        if not isinstance(boundary, _BoundaryCase):
            raise TypeError(f"boundary must be _BoundaryCase, got {type(boundary).__name__}")
        return materialize_kernel_boundary(boundary, fit_backend="cxx").boundary

    @staticmethod
    def _kernel_source(source: _SourceCase, *, case_name: str | None) -> _SourceCase:
        if not isinstance(source, _SourceCase):
            raise TypeError(f"source must be _SourceCase, got {type(source).__name__}")
        if case_name is None:
            return source
        return replace(source, case_name=case_name)

    def _validate_runtime_case_adaptability(
        self,
        boundary: _BoundaryCase,
        source: _MaterializedSource,
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


def _native_topology(topology: KernelTopology) -> KernelTopology:
    """Lower rho to the native-r operator coordinate without changing source count."""

    coordinate = "r" if topology.coordinate == "rho" else topology.coordinate
    return topology if coordinate == topology.coordinate else replace(topology, coordinate=coordinate)


def _native_source_case(
    topology: KernelTopology,
    source: _SourceCase,
    *,
    native_topology: KernelTopology | None = None,
) -> _SourceCase:
    """Pack one explicit source case into the fixed native workspace.

    Cxx uses the r operator coordinate for rho topologies.  The Adapter keeps
    the corresponding r nodes and d rho / dr in private source buffers, so
    coordinate derivatives receive the same chain rule as the Numba path.
    """

    native_topology = topology if native_topology is None else native_topology
    source_nodes = (
        np.asarray(
            source.native_source_nodes
            if topology.coordinate == "rho" and source.native_source_nodes is not None
            else source.source_nodes,
            dtype=np.float64,
        )
        if source.source_nodes is not None
        else np.linspace(0.0, 1.0, source.pressure_profile.size, dtype=np.float64)
    )
    source_count = int(source.pressure_profile.size)
    pressure_profile = np.asarray(source.pressure_profile, dtype=np.float64)
    driver_profile = np.asarray(source.driver_profile, dtype=np.float64)
    if topology.coordinate == "rho":
        jacobian = source.source_coordinate_jacobian
        if jacobian is None:
            raise ValueError("rho Cxx source lowering requires d rho / dr source data")
        jacobian = np.asarray(jacobian, dtype=np.float64)
        if source.pressure_name == "P_rho":
            pressure_profile = pressure_profile * jacobian
        if source.driver_name == "FF_rho":
            driver_profile = driver_profile * jacobian
    pressure_name = PRESSURE_DERIVATIVE_BY_COORDINATE[native_topology.coordinate]
    driver_name = source_driver_for(native_topology.route, native_topology.coordinate)
    values: dict[str, object] = {
        "Ip": source.Ip,
        "beta": source.beta,
        "case_name": source.case_name,
        # Keep the explicit source coordinate and values intact.  The native
        # runtime receives matching PCHIP coefficients and evaluates them at
        # its changing physical queries; resampling to a fictitious uniform
        # grid here changes the source function before Cxx sees it.
        "source_nodes": np.ascontiguousarray(source_nodes[:source_count], dtype=np.float64),
        driver_name: np.ascontiguousarray(driver_profile[:source_count], dtype=np.float64),
    }
    if source.pressure_name == "p":
        values["p"] = np.ascontiguousarray(pressure_profile[:source_count], dtype=np.float64)
    else:
        values[pressure_name] = np.ascontiguousarray(pressure_profile[:source_count], dtype=np.float64)
        values["p0"] = source.p0
    return _SourceCase(**values)
