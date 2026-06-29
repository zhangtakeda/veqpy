from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veqpy.cpp import KernelArtifact, KernelRegistry, VEQlibSolver
from veqpy.model import Problem

from .types import KernelBuild, KernelInput, KernelSolve, KernelTopology


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
    ) -> None:
        self.topology = topology
        self.build_config = KernelBuild() if build is None else build
        self.legacy_topology = topology.to_legacy_topology(self.build_config)
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            cxx=cxx,
        )
        self._solver: VEQlibSolver | None = None
        self.history: list[Any] = []
        self.result: Any | None = None

    @property
    def x_size(self) -> int:
        return self.topology.packed_size(build=self.build_config)

    def build(self, *, force: bool = False, dry_run: bool = False) -> KernelArtifact:
        return self._veqlib_solver().build(force=force, dry_run=dry_run)

    def payload_json(
        self,
        input: KernelInput | Problem,
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
        input: KernelInput | Problem,
        *,
        solve: KernelSolve | None = None,
        case_name: str | None = None,
    ) -> Any:
        solver = self._veqlib_solver()
        solver.set_case_json(self.payload_json(input, solve=solve, case_name=case_name))
        self.result = solver.solve_direct()
        self.history.append(self.result)
        return self.result

    def clear(self) -> None:
        self.history.clear()
        self.result = None

    def metadata(self) -> Any:
        return self._veqlib_solver().metadata()

    def metadata_json(self) -> str:
        return self._veqlib_solver().metadata_json()

    def _veqlib_solver(self) -> VEQlibSolver:
        if self._solver is None:
            self._solver = VEQlibSolver(
                self.legacy_topology,
                registry=self.registry,
            )
        return self._solver

    @staticmethod
    def _kernel_input(input: KernelInput | Problem, *, case_name: str | None) -> KernelInput:
        if isinstance(input, KernelInput):
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
        if isinstance(input, Problem):
            return KernelInput.from_problem(input, case_name=case_name)
        raise TypeError(f"input must be KernelInput or Problem, got {type(input).__name__}")


def build(
    topology: KernelTopology,
    *,
    build: KernelBuild | None = None,
    registry: KernelRegistry | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    cxx: str = "clang++",
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
    )
    kernel.build(force=force, dry_run=dry_run)
    return kernel
