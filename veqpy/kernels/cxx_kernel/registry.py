"""
Module: veqpy.kernels.cxx_kernel.registry

Role:
- Cache native Cxx modules and manage thread-owned solver guards.

Notes:
- Loaded topology artifacts are process-scoped; each ``NativeSolver`` guard owns
  mutable C++ workspace and is bound to the creating Python thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from fusionprime_base.native import NativeArtifactLoadError, load_native_artifact

from veqpy.kernels.abi.options import solver_method_code
from veqpy.kernels.types import KernelTopology, _BuildPolicy

from .affinity import cpu_pin_scope_active, pinned_cpu
from .builder import _ArtifactPreparation, _native_artifact_reference
from .builder import prepare as prepare_kernel


class KernelLoadError(ImportError):
    """Raised when a built Kernel artifact cannot be imported."""


class SolverThreadError(RuntimeError):
    """Raised when a thread-owned Cxx solver is used from another thread."""


class SolverClosedError(RuntimeError):
    """Raised when a closed Cxx solver guard is used again."""


@dataclass(frozen=True, slots=True)
class LoadedKernel:
    """A process-cached nanobind module and its artifact metadata."""

    artifact: _ArtifactPreparation
    module: ModuleType


class ThreadOwnedNativeSolver:
    """Python guard around the mutable C++ NativeSolver workspace."""

    def __init__(self, solver: Any, *, pin_cpu: bool | int | None = None) -> None:
        self._solver: Any | None = solver
        self._pin_cpu = pin_cpu
        self._owner_thread_id = threading.get_ident()

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    @property
    def closed(self) -> bool:
        return self._solver is None

    def check_thread(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise SolverThreadError(
                "Cxx NativeSolver owns mutable C++ workspace and cannot be used "
                f"from thread {current}; owner thread is {self._owner_thread_id}"
            )

    def metadata(self) -> Any:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.metadata)

    def source_state(self) -> Any:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.source_state)

    def set_kernel_runtime(self, *args: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.set_kernel_runtime, *args)

    def warmup(self, count: int) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.warmup, count)

    def solve_direct(self) -> Any:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.solve_direct)

    def adopt_last_solution_as_initial(self) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.adopt_last_solution_as_initial)

    def set_initial_state(self, x0: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.set_initial_state, x0)

    def residual_var_into(self, out: Any, x: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.residual_var_into, out, x)

    def jvp_into(self, out: Any, x: Any, v: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.jvp_into, out, x, v)

    def jacobian_into(self, out: Any, x: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.jacobian_into, out, x)

    def close(self) -> None:
        self.check_thread()
        self._solver = None

    @property
    def last_elapsed_ms(self) -> float:
        self.check_thread()
        solver = self._require_solver()
        return float(self._call_native(lambda: solver.last_elapsed_ms))

    def _require_solver(self) -> Any:
        if self._solver is None:
            raise SolverClosedError("Cxx NativeSolver guard is closed")
        return self._solver

    def _call_native(self, method: Any, *args: Any) -> Any:
        if self._pin_cpu is False or cpu_pin_scope_active():
            return method(*args)
        with pinned_cpu(self._pin_cpu):
            return method(*args)


class KernelRegistry:
    """Process cache for Cxx nanobind modules and per-thread C++ solvers."""

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.source_dir = source_dir
        self.pin_cpu = pin_cpu
        self._modules: dict[str, LoadedKernel] = {}
        self._thread_local = threading.local()
        self._lock = threading.RLock()

    def prepare_artifact(
        self,
        topology: KernelTopology,
        *,
        recipe: _BuildPolicy | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> _ArtifactPreparation:
        return prepare_kernel(
            topology,
            recipe=recipe,
            cache_root=self.cache_root,
            source_dir=self.source_dir,
            force=force,
            dry_run=dry_run,
        )

    def load_kernel(
        self,
        topology: KernelTopology,
        *,
        recipe: _BuildPolicy | None = None,
        force: bool = False,
    ) -> LoadedKernel:
        artifact = self.prepare_artifact(topology, recipe=recipe, force=force, dry_run=False)
        return self.load_artifact(artifact, force=force)

    def load_artifact(
        self,
        artifact: _ArtifactPreparation,
        *,
        force: bool = False,
    ) -> LoadedKernel:
        """Load one already-resolved artifact without repeating preparation."""

        with self._lock:
            cached = self._modules.get(artifact.artifact_id)
            if cached is not None and not force:
                return cached
            module = _load_artifact_module(artifact)
            loaded = LoadedKernel(artifact=artifact, module=module)
            self._modules[artifact.artifact_id] = loaded
            return loaded

    def create_solver(
        self,
        topology: KernelTopology,
        *,
        recipe: _BuildPolicy | None = None,
        solver: str = "powell",
        force: bool = False,
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedNativeSolver:
        loaded = self.load_kernel(topology, recipe=recipe, force=force)
        return self.create_solver_from_loaded(loaded, solver=solver, pin_cpu=pin_cpu)

    def create_solver_from_artifact(
        self,
        artifact: _ArtifactPreparation,
        *,
        solver: str = "powell",
        force: bool = False,
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedNativeSolver:
        loaded = self.load_artifact(artifact, force=force)
        return self.create_solver_from_loaded(loaded, solver=solver, pin_cpu=pin_cpu)

    def create_solver_from_loaded(
        self,
        loaded: LoadedKernel,
        *,
        solver: str = "powell",
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedNativeSolver:
        solver_code = solver_method_code(solver)
        pin_policy = self.pin_cpu if pin_cpu is None else pin_cpu
        cpp_solver = loaded.module.NativeSolver(
            solver_code=solver_code,
        )
        return ThreadOwnedNativeSolver(cpp_solver, pin_cpu=pin_policy)

    def get_thread_solver(
        self,
        topology: KernelTopology,
        *,
        recipe: _BuildPolicy | None = None,
        solver: str = "powell",
        force: bool = False,
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedNativeSolver:
        loaded = self.load_kernel(topology, recipe=recipe, force=force)
        solvers = self._thread_solver_cache()
        solver_code = solver_method_code(solver)
        pin_policy = self.pin_cpu if pin_cpu is None else pin_cpu
        key = (loaded.artifact.artifact_id, solver_code, _pinning_cache_key(pin_policy))
        cached = solvers.get(key)
        if cached is not None and not cached.closed:
            return cached
        cpp_solver = loaded.module.NativeSolver(
            solver_code=solver_code,
        )
        wrapped = ThreadOwnedNativeSolver(cpp_solver, pin_cpu=pin_policy)
        solvers[key] = wrapped
        return wrapped

    def _thread_solver_cache(self) -> dict[tuple[str, int, object], ThreadOwnedNativeSolver]:
        solvers = getattr(self._thread_local, "solvers", None)
        if solvers is None:
            solvers = {}
            self._thread_local.solvers = solvers
        return solvers


def _load_artifact_module(artifact: _ArtifactPreparation) -> ModuleType:
    try:
        return load_native_artifact(_native_artifact_reference(artifact))
    except NativeArtifactLoadError as error:
        raise KernelLoadError(str(error)) from error


def _pinning_cache_key(policy: bool | int | None) -> object:
    if isinstance(policy, bool) or policy is None:
        return policy
    return int(policy)
