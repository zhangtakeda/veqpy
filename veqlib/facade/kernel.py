"""High-level Python handle for topology-specific VEQlib kernels.

This module owns the user-facing ``Kernel`` lifecycle: artifact resolution,
typed ``KernelBoundary``/``KernelInput``/``KernelConfig`` runtime calls, and
Python-owned result snapshots. It deliberately does not translate external
``Operator`` objects or make benchmark adapters part of the VEQlib ABI.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from .affinity import pinned_cpu
from .builder import KernelArtifact
from .registry import KernelRegistry
from .solver import VEQlibSolver
from .types import (
    KernelBoundary,
    KernelBuild,
    KernelConfig,
    KernelInput,
    KernelResult,
    KernelTopology,
)


class Kernel:
    """Stateful VEQlib kernel handle backed by one topology/build artifact."""

    def __init__(
        self,
        topology: KernelTopology,
        *,
        build: KernelBuild | None = None,
        config: KernelConfig | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
        backend: str = "cxx",
    ) -> None:
        self.topology = topology
        self.build_config = KernelBuild() if build is None else build
        self.build_topology = topology.with_build(self.build_config)
        self.config = KernelConfig() if config is None else self._kernel_config(config)
        self.backend = backend
        self.pin_cpu = pin_cpu
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self._solver: VEQlibSolver | None = None
        self.history: list[KernelResult] = []
        self.result: KernelResult | None = None

    @property
    def x_size(self) -> int:
        return self.build_topology.packed_size()

    def build(self, *, force: bool = False, dry_run: bool = False) -> KernelArtifact:
        return self._veqlib_solver().build(force=force, dry_run=dry_run)

    def solve(
        self,
        boundary: KernelBoundary,
        input: KernelInput,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        **config_overrides: Any,
    ) -> KernelResult:
        kernel_config = self._runtime_config(config, config_overrides)
        solver = self._set_runtime(boundary, input, kernel_config, case_name=case_name)
        self.result = KernelResult.from_solve_direct(solver.solve_direct())
        self.history.append(self.result)
        return self.result

    # Raw numerical APIs intentionally do not expose per-call solve policy.  The
    # handle default config only installs the native current-case context required
    # before residual/JVP/Jacobian kernels run.
    def residual(self, x: Any, boundary: KernelBoundary, input: KernelInput) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, input)
        return out

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        input: KernelInput,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, input, self.config, case_name=None)
        self._veqlib_solver().residual_var_into(packed_out, packed_x)

    def jvp(self, x: Any, v: Any, boundary: KernelBoundary, input: KernelInput) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v, boundary, input)
        return out

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: KernelBoundary,
        input: KernelInput,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        self._set_runtime(boundary, input, self.config, case_name=None)
        self._veqlib_solver().jvp_into(packed_out, packed_x, packed_v)

    def jacobian(self, x: Any, boundary: KernelBoundary, input: KernelInput) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x, boundary, input)
        return out

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        input: KernelInput,
    ) -> None:
        matrix_out = self._packed_output(out, (self.x_size, self.x_size), "out")
        packed_x = self._packed_input(x, "x")
        self._set_runtime(boundary, input, self.config, case_name=None)
        self._veqlib_solver().jacobian_into(matrix_out, packed_x)

    def clear(self) -> None:
        self.history.clear()
        self.result = None

    def close(self) -> None:
        if self._solver is not None:
            self._solver.close()
            self._solver = None

    def pinned(self) -> AbstractContextManager[None, bool | None]:
        """Return a scoped CPU pinning context for high-volume solve loops."""

        return pinned_cpu(self.pin_cpu)

    def _veqlib_solver(self) -> VEQlibSolver:
        if self.backend != "cxx":
            raise ValueError("VEQlib facade currently only supports backend='cxx'")
        if self._solver is None:
            self._solver = VEQlibSolver(
                self.build_topology,
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
            kernel_config = kernel_config.with_overrides(**overrides)
        return kernel_config

    def _set_runtime(
        self,
        boundary: KernelBoundary,
        input: KernelInput,
        config: KernelConfig,
        *,
        case_name: str | None,
    ) -> VEQlibSolver:
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_input = self._kernel_input(input, case_name=case_name)
        self._validate_runtime_case_adaptability(kernel_boundary, kernel_input)
        solver = self._veqlib_solver()
        solver.set_kernel_runtime(
            "" if kernel_input.case_name is None else kernel_input.case_name,
            *kernel_boundary.runtime_args(),
            *kernel_input.runtime_args(),
            *config.runtime_args(x_size=self.x_size),
        )
        return solver

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
    def _kernel_input(input: KernelInput, *, case_name: str | None) -> KernelInput:
        if not isinstance(input, KernelInput):
            raise TypeError(f"input must be KernelInput, got {type(input).__name__}")
        if case_name is None:
            return input
        return KernelInput(
            scaled_heat=input.scaled_heat,
            scaled_current=input.scaled_current,
            scaled_Ip=input.scaled_Ip,
            beta=input.beta,
            case_name=case_name,
        )

    def _validate_runtime_case_adaptability(
        self,
        boundary: KernelBoundary,
        input: KernelInput,
    ) -> None:
        topology = self.build_topology
        expected_samples = topology.sample_count
        heat_length = input.scaled_heat.size
        current_length = input.scaled_current.size
        if heat_length != expected_samples or current_length != expected_samples:
            raise ValueError(
                "case does not match kernel topology: scaled_heat and scaled_current "
                f"must have length {expected_samples} for "
                f"route={topology.route}/{topology.coordinate}/{topology.nodes}, "
                f"got {heat_length} and {current_length}"
            )

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


def build(
    topology: KernelTopology,
    *,
    build: KernelBuild | None = None,
    config: KernelConfig | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Kernel:
    """Create a kernel handle, cache its default config, and resolve its artifact."""

    kernel = Kernel(
        topology,
        build=build,
        config=config,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    kernel.build(force=force, dry_run=dry_run)
    return kernel


def solve(
    topology: KernelTopology,
    boundary: KernelBoundary,
    input: KernelInput,
    *,
    config: KernelConfig | None = None,
    build: KernelBuild | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    case_name: str | None = None,
    **config_overrides: Any,
) -> KernelResult:
    """Build a short-lived kernel, solve one case, and close its private workspace."""

    kernel = Kernel(
        topology,
        build=build,
        config=config,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    try:
        kernel.build(force=force, dry_run=False)
        return kernel.solve(boundary, input, case_name=case_name, **config_overrides)
    finally:
        kernel.close()
