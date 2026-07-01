"""Low-level VEQlib solver facade around a thread-owned C++ workspace.

Most callers should use ``veqlib.facade.Kernel``. This layer stays close to the
nanobind surface for benchmark harnesses and lifecycle tests that need explicit
``set_kernel_runtime`` / ``solve_direct`` control.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .builder import KernelArtifact
from .options import solver_method_code
from .registry import KernelRegistry, SolverThreadError, ThreadOwnedKernelSolver
from .types import KernelTopology as Topology


class VEQlibSolver:
    """Experimental Solver facade for the new VEQlib/nanobind architecture."""

    def __init__(
        self,
        topology: Topology,
        *,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        solver: str | int = "powell",
        pin_cpu: bool | int | None = None,
    ) -> None:
        topology.validate_supported_for_veqlib_native()
        self.topology = topology
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self.solver_code = solver_method_code(solver)
        self.pin_cpu = pin_cpu
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
            self._cpp_solver = self.registry.create_solver(
                self.topology,
                solver=self.solver_code,
                pin_cpu=self.pin_cpu,
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

    def set_kernel_runtime(self, *args: Any) -> None:
        self._solver().set_kernel_runtime(*args)

    def solve_json(self) -> str:
        return self._solver().solve_json()

    def solve_direct(self) -> Any:
        return self._solver().solve_direct()

    def adopt_last_solution_as_initial(self) -> None:
        self._solver().adopt_last_solution_as_initial()

    def residual_var_into(self, out: Any, x: Any) -> None:
        self._solver().residual_var_into(out, x)

    def jvp_into(self, out: Any, x: Any, v: Any) -> None:
        self._solver().jvp_into(out, x, v)

    def jacobian_into(self, out: Any, x: Any) -> None:
        self._solver().jacobian_into(out, x)

    def close(self) -> None:
        self.check_thread()
        if self._cpp_solver is not None:
            self._cpp_solver.close()
            self._cpp_solver = None

    @property
    def last_elapsed_ms(self) -> float:
        return self._solver().last_elapsed_ms
