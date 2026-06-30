from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from .affinity import pinned_cpu
from .builder import KernelArtifact
from .registry import KernelRegistry
from .solver import VEQlibSolver
from .types import KernelBuild, KernelConfig, KernelInput, KernelResult, KernelTopology


class Kernel:
    """Stateful VEQlib kernel handle backed by one topology/build artifact."""

    def __init__(
        self,
        topology: KernelTopology,
        *,
        build: KernelBuild | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
        backend: str = "cxx",
    ) -> None:
        self.topology = topology
        self.build_config = KernelBuild() if build is None else build
        self.build_topology = topology.with_build(self.build_config)
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

    def payload_json(
        self,
        case: KernelInput,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
    ) -> str:
        kernel_case = self._kernel_input(case, case_name=case_name)
        kernel_config = KernelConfig() if config is None else config
        payload = kernel_case.to_payload_dict()
        payload["solver"] = kernel_config.to_payload_dict(x_size=self.x_size)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def solve(
        self,
        case: KernelInput,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
    ) -> KernelResult:
        solver = self._veqlib_solver()
        kernel_case = self._kernel_input(case, case_name=case_name)
        kernel_config = KernelConfig() if config is None else config
        try:
            solver.set_kernel_runtime(
                *kernel_case.runtime_args(),
                *kernel_config.runtime_args(x_size=self.x_size),
            )
        except AttributeError:
            solver.set_case_json(self.payload_json(kernel_case, config=kernel_config))
        self.result = KernelResult.from_solve_direct(solver.solve_direct())
        self.history.append(self.result)
        return self.result

    def residual(self, x: Any) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x)
        return out

    def residual_into(self, out: np.ndarray, x: Any) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        self._veqlib_solver().residual_var_into(packed_out, packed_x)

    def jvp(self, x: Any, v: Any) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v)
        return out

    def jvp_into(self, out: np.ndarray, x: Any, v: Any) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        self._veqlib_solver().jvp_into(packed_out, packed_x, packed_v)

    def jacobian(self, x: Any) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x)
        return out

    def jacobian_into(self, out: np.ndarray, x: Any) -> None:
        matrix_out = self._packed_output(out, (self.x_size, self.x_size), "out")
        packed_x = self._packed_input(x, "x")
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

    @staticmethod
    def _kernel_input(case: KernelInput, *, case_name: str | None) -> KernelInput:
        if not isinstance(case, KernelInput):
            raise TypeError(f"case must be KernelInput, got {type(case).__name__}")
        if case_name is None:
            return case
        return KernelInput(
            boundary=case.boundary,
            scaled_heat=case.scaled_heat,
            scaled_current=case.scaled_current,
            scaled_Ip=case.scaled_Ip,
            beta=case.beta,
            case_name=case_name,
        )


def build(
    topology: KernelTopology,
    *,
    build: KernelBuild | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Kernel:
    """Create a kernel handle and resolve its artifact plan/build."""

    kernel = Kernel(
        topology,
        build=build,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    kernel.build(force=force, dry_run=dry_run)
    return kernel


def solve(
    topology: KernelTopology,
    case: KernelInput,
    *,
    config: KernelConfig | None = None,
    build: KernelBuild | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
    force: bool = False,
    case_name: str | None = None,
) -> KernelResult:
    """Build a short-lived kernel, solve one case, and close its private workspace."""

    kernel = Kernel(
        topology,
        build=build,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    try:
        kernel.build(force=force, dry_run=False)
        return kernel.solve(case, config=config, case_name=case_name)
    finally:
        kernel.close()
