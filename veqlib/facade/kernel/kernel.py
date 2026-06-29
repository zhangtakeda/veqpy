from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..affinity import CpuPinning
from .builder import KernelArtifact
from .registry import KernelRegistry
from .solver import VEQlibSolver
from .types import KernelBuild, KernelInput, KernelResult, KernelSolve, KernelTopology


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
        cxx: str = "clang++",
        pin_cpu: CpuPinning = None,
    ) -> None:
        self.topology = topology
        self.build_config = KernelBuild() if build is None else build
        self.build_topology = topology.with_build(self.build_config)
        self.pin_cpu = pin_cpu
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            cxx=cxx,
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
        input: KernelInput,
        *,
        solve: KernelSolve | None = None,
        case_name: str | None = None,
    ) -> str:
        kernel_input = self._kernel_input(input, case_name=case_name)
        kernel_solve = KernelSolve() if solve is None else solve
        payload = kernel_input.to_payload_dict()
        payload["solver"] = kernel_solve.to_payload_dict(x_size=self.x_size)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def solve(
        self,
        input: KernelInput,
        *,
        solve: KernelSolve | None = None,
        case_name: str | None = None,
    ) -> KernelResult:
        solver = self._veqlib_solver()
        kernel_input = self._kernel_input(input, case_name=case_name)
        kernel_solve = KernelSolve() if solve is None else solve
        try:
            solver.set_kernel_runtime(
                *kernel_input.runtime_args(),
                *kernel_solve.runtime_args(x_size=self.x_size),
            )
        except AttributeError:
            solver.set_case_json(self.payload_json(kernel_input, solve=kernel_solve))
        self.result = KernelResult.from_solve_direct(solver.solve_direct())
        self.history.append(self.result)
        return self.result

    def clear(self) -> None:
        self.history.clear()
        self.result = None

    def close(self) -> None:
        self._solver = None

    def metadata(self) -> Any:
        return self._veqlib_solver().metadata()

    def metadata_json(self) -> str:
        return self._veqlib_solver().metadata_json()

    def _veqlib_solver(self) -> VEQlibSolver:
        if self._solver is None:
            self._solver = VEQlibSolver(
                self.build_topology,
                registry=self.registry,
                pin_cpu=self.pin_cpu,
            )
        return self._solver

    @staticmethod
    def _kernel_input(input: KernelInput, *, case_name: str | None) -> KernelInput:
        if not isinstance(input, KernelInput):
            raise TypeError(f"input must be KernelInput, got {type(input).__name__}")
        if case_name is None:
            return input
        return KernelInput(
            boundary=input.boundary,
            scaled_heat=input.scaled_heat,
            scaled_current=input.scaled_current,
            scaled_Ip=input.scaled_Ip,
            beta=input.beta,
            fix_rho=input.fix_rho,
            case_name=case_name,
        )


def build(
    topology: KernelTopology,
    *,
    build: KernelBuild | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    cxx: str = "clang++",
    pin_cpu: CpuPinning = None,
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
        cxx=cxx,
        pin_cpu=pin_cpu,
    )
    kernel.build(force=force, dry_run=dry_run)
    return kernel
