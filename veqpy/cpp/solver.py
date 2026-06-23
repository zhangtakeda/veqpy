from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from veqpy.topology import Topology

from .kernel_builder import KernelArtifact
from .kernel_registry import KernelRegistry, SolverThreadError, ThreadOwnedKernelSolver
from .options import solver_method_code


class VEQlibSolver:
    """Experimental Solver facade for the new VEQlib/nanobind architecture."""

    def __init__(
        self,
        topology: Topology,
        *,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        cxx: str = "clang++",
        solver: str | int = "powell",
        enzyme_width: int = 1,
    ) -> None:
        topology.validate_supported_for_veqlib_mvp()
        self.topology = topology
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            cxx=cxx,
        )
        self.solver_code = solver_method_code(solver)
        self.enzyme_width = enzyme_width
        self._owner_thread_id = threading.get_ident()
        self._cpp_solver: ThreadOwnedKernelSolver | None = None

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def check_thread(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise SolverThreadError(
                "VEQlibSolver cannot be used across threads; create one solver per thread "
                f"or use KernelRegistry.get_thread_solver(). owner={self._owner_thread_id}, "
                f"current={current}"
            )

    def build(self, *, force: bool = False, dry_run: bool = False) -> KernelArtifact:
        self.check_thread()
        return self.registry.get_or_build(self.topology, force=force, dry_run=dry_run)

    def _solver(self) -> ThreadOwnedKernelSolver:
        self.check_thread()
        if self._cpp_solver is None:
            self._cpp_solver = self.registry.get_thread_solver(
                self.topology,
                solver=self.solver_code,
                enzyme_width=self.enzyme_width,
            )
        return self._cpp_solver

    def metadata(self) -> Any:
        return self._solver().metadata()

    def metadata_json(self) -> str:
        return self._solver().metadata_json()

    def warmup(self, count: int = 1) -> None:
        self._solver().warmup(count)

    def set_case_json(self, payload: str) -> None:
        self._solver().set_case_json(payload)

    def solve_json(self) -> str:
        return self._solver().solve_json()

    def solve_direct(self) -> Any:
        return self._solver().solve_direct()

    def residual_var_into(self, x: Any, out: Any) -> None:
        self._solver().residual_var_into(x, out)

    @property
    def last_elapsed_ms(self) -> float:
        return self._solver().last_elapsed_ms
