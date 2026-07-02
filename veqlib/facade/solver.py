"""Low-level VEQlib solver facade around a thread-owned C++ workspace.

Most callers should use ``veqlib.facade.Kernel``. This layer stays close to the
nanobind surface for benchmark harnesses and lifecycle tests that need explicit
``set_kernel_runtime`` / ``solve_direct`` control.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .builder import CompileResult
from .options import solver_method_code
from .registry import KernelRegistry, SolverThreadError, ThreadOwnedNativeSolver
from .types import KernelRecipe as Recipe
from .types import KernelTopology as Topology


class VEQlibSolver:
    """Experimental Solver facade for the new VEQlib/nanobind architecture."""

    def __init__(
        self,
        topology: Topology,
        *,
        recipe: Recipe | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        solver: str | int = "powell",
        pin_cpu: bool | int | None = None,
    ) -> None:
        topology.validate_supported_for_veqlib_native()
        self.topology = topology
        self.recipe = Recipe() if recipe is None else recipe
        if not isinstance(self.recipe, Recipe):
            raise TypeError(f"recipe must be KernelRecipe, got {type(self.recipe).__name__}")
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self.solver_code = solver_method_code(solver)
        self.pin_cpu = pin_cpu
        self._owner_thread_id = threading.get_ident()
        self._native_solver: ThreadOwnedNativeSolver | None = None

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

    def compile(self, *, force: bool = False, dry_run: bool = False) -> CompileResult:
        self.check_thread()
        return self.registry.get_or_compile(
            self.topology,
            recipe=self.recipe,
            force=force,
            dry_run=dry_run,
        )

    def _solver(self) -> ThreadOwnedNativeSolver:
        self.check_thread()
        if self._native_solver is None:
            self._native_solver = self.registry.create_solver(
                self.topology,
                recipe=self.recipe,
                solver=self.solver_code,
                pin_cpu=self.pin_cpu,
            )
        return self._native_solver

    def metadata(self) -> Any:
        return self._solver().metadata()

    def warmup(self, count: int = 1) -> None:
        self._solver().warmup(count)

    def set_kernel_runtime(self, *args: Any) -> None:
        self._solver().set_kernel_runtime(*args)

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
        if self._native_solver is not None:
            self._native_solver.close()
            self._native_solver = None

    @property
    def last_elapsed_ms(self) -> float:
        return self._solver().last_elapsed_ms
