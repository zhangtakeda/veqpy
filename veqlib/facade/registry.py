from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .affinity import cpu_pin_scope_active, pinned_cpu
from .builder import KernelArtifact, build_kernel
from .options import solver_method_code
from .types import KernelTopology as Topology


class KernelLoadError(ImportError):
    """Raised when a built VEQlib kernel artifact cannot be imported."""


class SolverThreadError(RuntimeError):
    """Raised when a thread-owned VEQlib solver is used from another thread."""


class SolverClosedError(RuntimeError):
    """Raised when a closed VEQlib solver wrapper is used again."""


@dataclass(frozen=True, slots=True)
class LoadedKernel:
    """A process-cached nanobind module and its artifact metadata."""

    artifact: KernelArtifact
    module: ModuleType


class ThreadOwnedKernelSolver:
    """Small Python guard around the mutable C++ KernelSolver workspace."""

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
                "VEQlib KernelSolver owns mutable C++ workspace and cannot be used "
                f"from thread {current}; owner thread is {self._owner_thread_id}"
            )

    def metadata(self) -> Any:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.metadata)

    def metadata_json(self) -> str:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.metadata_json)

    def set_case_json(self, payload: str) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.set_case_json, payload)

    def set_kernel_runtime(self, *args: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.set_kernel_runtime, *args)

    def warmup(self, count: int) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.warmup, count)

    def solve_json(self) -> str:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.solve_json)

    def solve_direct(self) -> Any:
        self.check_thread()
        solver = self._require_solver()
        return self._call_native(solver.solve_direct)

    def adopt_last_solution_as_initial(self) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.adopt_last_solution_as_initial)

    def residual_var_into(self, x: Any, out: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.residual_var_into, x, out)

    def jvp_into(self, x: Any, v: Any, out: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.jvp_into, x, v, out)

    def jacobian_into(self, x: Any, out: Any) -> None:
        self.check_thread()
        solver = self._require_solver()
        self._call_native(solver.jacobian_into, x, out)

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
            raise SolverClosedError("VEQlib KernelSolver wrapper is closed")
        return self._solver

    def _call_native(self, method: Any, *args: Any) -> Any:
        if self._pin_cpu is False or cpu_pin_scope_active():
            return method(*args)
        with pinned_cpu(self._pin_cpu):
            return method(*args)


class KernelRegistry:
    """Process cache for VEQlib nanobind modules and per-thread C++ solvers."""

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
        self._topology_modules: dict[str, LoadedKernel] = {}
        self._thread_local = threading.local()
        self._lock = threading.RLock()

    def get_or_build(
        self,
        topology: Topology,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> KernelArtifact:
        return build_kernel(
            topology,
            cache_root=self.cache_root,
            source_dir=self.source_dir,
            force=force,
            dry_run=dry_run,
        )

    def load_kernel(self, topology: Topology, *, force: bool = False) -> LoadedKernel:
        topology_key = topology.key or topology.compute_key()
        with self._lock:
            cached_by_topology = self._topology_modules.get(topology_key)
            if cached_by_topology is not None and not force:
                return cached_by_topology

        artifact = self.get_or_build(topology, force=force, dry_run=False)
        with self._lock:
            cached = self._modules.get(artifact.artifact_id)
            if cached is not None and not force:
                self._topology_modules[topology_key] = cached
                return cached
            module = _load_artifact_module(artifact)
            loaded = LoadedKernel(artifact=artifact, module=module)
            self._modules[artifact.artifact_id] = loaded
            self._topology_modules[topology_key] = loaded
            return loaded

    def create_solver(
        self,
        topology: Topology,
        *,
        solver: str | int = "powell",
        force: bool = False,
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedKernelSolver:
        loaded = self.load_kernel(topology, force=force)
        solver_code = solver_method_code(solver)
        pin_policy = self.pin_cpu if pin_cpu is None else pin_cpu
        cpp_solver = loaded.module.KernelSolver(
            solver_code=solver_code,
        )
        return ThreadOwnedKernelSolver(cpp_solver, pin_cpu=pin_policy)

    def get_thread_solver(
        self,
        topology: Topology,
        *,
        solver: str | int = "powell",
        force: bool = False,
        pin_cpu: bool | int | None = None,
    ) -> ThreadOwnedKernelSolver:
        loaded = self.load_kernel(topology, force=force)
        solvers = self._thread_solver_cache()
        solver_code = solver_method_code(solver)
        pin_policy = self.pin_cpu if pin_cpu is None else pin_cpu
        key = (loaded.artifact.artifact_id, solver_code, _pinning_cache_key(pin_policy))
        cached = solvers.get(key)
        if cached is not None and not cached.closed:
            return cached
        cpp_solver = loaded.module.KernelSolver(
            solver_code=solver_code,
        )
        wrapped = ThreadOwnedKernelSolver(cpp_solver, pin_cpu=pin_policy)
        solvers[key] = wrapped
        return wrapped

    def _thread_solver_cache(self) -> dict[tuple[str, int, object], ThreadOwnedKernelSolver]:
        solvers = getattr(self._thread_local, "solvers", None)
        if solvers is None:
            solvers = {}
            self._thread_local.solvers = solvers
        return solvers


def load_kernel(
    topology: Topology,
    *,
    registry: KernelRegistry | None = None,
    force: bool = False,
) -> LoadedKernel:
    """Load a VEQlib kernel through ``registry`` or a short-lived default registry."""

    return (registry or KernelRegistry()).load_kernel(topology, force=force)


def _load_artifact_module(artifact: KernelArtifact) -> ModuleType:
    if not artifact.shared_library_path.exists():
        raise KernelLoadError(f"VEQlib shared library is missing: {artifact.shared_library_path}")
    module_name = _module_name_for_artifact(artifact.artifact_id)
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, artifact.shared_library_path)
    if spec is None or spec.loader is None:
        raise KernelLoadError(f"cannot create import spec for {artifact.shared_library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _module_name_for_artifact(artifact_id: str) -> str:
    safe_id = artifact_id.replace("-", "_")
    return f"veqlib._kernel_cache.k_{safe_id}.veqlib_ext"


def _pinning_cache_key(policy: bool | int | None) -> object:
    if isinstance(policy, bool) or policy is None:
        return policy
    return int(policy)
