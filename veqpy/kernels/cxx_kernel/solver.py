"""
Module: veqpy.kernels.cxx_kernel.solver

Role:
- Provide a low-level Cxx solver wrapper around a thread-owned native workspace.

Notes:
- Most callers should use ``veqpy.Kernel``. This layer stays close to the
  nanobind surface for benchmark harnesses and lifecycle tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from veqpy.kernels.abi.options import normalize_solver_method, solver_method_code
from veqpy.kernels.types import KernelTopology, _BuildPolicy

from .builder import _ArtifactPreparation
from .registry import KernelRegistry, SolverThreadError, ThreadOwnedNativeSolver
from .validation import validate_supported_for_cxx_backend


class CxxSolver:
    """Experimental solver wrapper for the Cxx/nanobind architecture."""

    def __init__(
        self,
        topology: KernelTopology,
        *,
        recipe: _BuildPolicy | None = None,
        registry: KernelRegistry | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        solver: str = "powell",
        pin_cpu: bool | int | None = None,
    ) -> None:
        validate_supported_for_cxx_backend(topology)
        self.topology = topology
        self.recipe = (
            _BuildPolicy(backend="cxx-relaxed", build="release-relaxed")
            if recipe is None
            else recipe
        )
        if not isinstance(self.recipe, _BuildPolicy):
            raise TypeError(f"recipe must be _BuildPolicy, got {type(self.recipe).__name__}")
        self.registry = registry or KernelRegistry(
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self.solver = normalize_solver_method(solver)
        self.solver_code = solver_method_code(self.solver)
        self.pin_cpu = pin_cpu
        self._owner_thread_id = threading.get_ident()
        self._native_solver: ThreadOwnedNativeSolver | None = None
        self._artifact: _ArtifactPreparation | None = None

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def check_thread(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise SolverThreadError(
                "CxxSolver cannot be used across threads; create one solver per thread "
                f"or use KernelRegistry.get_thread_solver(). owner={self._owner_thread_id}, "
                f"current={current}"
            )

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> _ArtifactPreparation:
        self.check_thread()
        if self._artifact is not None and not force and not dry_run:
            return self._artifact
        artifact = self.registry.prepare_artifact(
            self.topology,
            recipe=self.recipe,
            force=force,
            dry_run=dry_run,
        )
        if dry_run:
            return artifact
        if self._native_solver is not None:
            self._native_solver.close()
        self._native_solver = self.registry.create_solver_from_artifact(
            artifact,
            solver=self.solver,
            force=force,
            pin_cpu=self.pin_cpu,
        )
        self._artifact = artifact
        return artifact

    def _solver(self) -> ThreadOwnedNativeSolver:
        self.check_thread()
        if self._native_solver is None:
            self.prepare()
        assert self._native_solver is not None
        return self._native_solver

    def metadata(self) -> Any:
        return self._solver().metadata()

    def source_state(self) -> Any:
        return self._solver().source_state()

    def warmup(self, count: int = 1) -> None:
        self._solver().warmup(count)

    def set_kernel_runtime(self, *args: Any) -> None:
        self._solver().set_kernel_runtime(*args)

    def solve_direct(self) -> Any:
        return self._solver().solve_direct()

    def adopt_last_solution_as_initial(self) -> None:
        self._solver().adopt_last_solution_as_initial()

    def set_initial_state(self, x0: Any) -> None:
        self._solver().set_initial_state(x0)

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
        self._artifact = None

    @property
    def last_elapsed_ms(self) -> float:
        return self._solver().last_elapsed_ms
