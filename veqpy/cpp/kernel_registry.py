from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from veqpy.topology import Topology

from .kernel_builder import KernelArtifact, build_kernel


class KernelLoadError(ImportError):
    """Raised when a built VEQlib kernel artifact cannot be imported."""


class SolverThreadError(RuntimeError):
    """Raised when a thread-owned VEQlib solver is used from another thread."""


@dataclass(frozen=True, slots=True)
class LoadedKernel:
    """A process-cached nanobind module and its artifact metadata."""

    artifact: KernelArtifact
    module: ModuleType


class ThreadOwnedKernelSolver:
    """Small Python guard around the mutable C++ KernelSolver workspace."""

    def __init__(self, solver: Any) -> None:
        self._solver = solver
        self._owner_thread_id = threading.get_ident()

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def check_thread(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise SolverThreadError(
                "VEQlib KernelSolver owns mutable C++ workspace and cannot be used "
                f"from thread {current}; owner thread is {self._owner_thread_id}"
            )

    def metadata(self) -> Any:
        self.check_thread()
        return self._solver.metadata()

    def metadata_json(self) -> str:
        self.check_thread()
        return self._solver.metadata_json()

    def set_case_json(self, payload: str) -> None:
        self.check_thread()
        self._solver.set_case_json(payload)

    def warmup(self, count: int) -> None:
        self.check_thread()
        self._solver.warmup(count)

    def solve_json(self) -> str:
        self.check_thread()
        return self._solver.solve_json()

    def solve_direct(self) -> Any:
        self.check_thread()
        return self._solver.solve_direct()

    def residual_var_into(self, x: Any, out: Any) -> None:
        self.check_thread()
        self._solver.residual_var_into(x, out)

    @property
    def last_elapsed_ms(self) -> float:
        self.check_thread()
        return float(self._solver.last_elapsed_ms)


class KernelRegistry:
    """Process cache for VEQlib nanobind modules and per-thread C++ solvers."""

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        cxx: str = "clang++",
    ) -> None:
        self.cache_root = cache_root
        self.source_dir = source_dir
        self.cxx = cxx
        self._modules: dict[str, LoadedKernel] = {}
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
            cxx=self.cxx,
            force=force,
            dry_run=dry_run,
        )

    def load_kernel(self, topology: Topology, *, force: bool = False) -> LoadedKernel:
        artifact = self.get_or_build(topology, force=force, dry_run=False)
        with self._lock:
            cached = self._modules.get(artifact.artifact_id)
            if cached is not None:
                return cached
            module = _load_artifact_module(artifact)
            loaded = LoadedKernel(artifact=artifact, module=module)
            self._modules[artifact.artifact_id] = loaded
            return loaded

    def get_thread_solver(
        self,
        topology: Topology,
        *,
        solver: str = "residual",
        enzyme_width: int = 1,
        force: bool = False,
    ) -> ThreadOwnedKernelSolver:
        loaded = self.load_kernel(topology, force=force)
        solvers = self._thread_solver_cache()
        key = (loaded.artifact.artifact_id, solver, enzyme_width)
        cached = solvers.get(key)
        if cached is not None:
            return cached
        cpp_solver = loaded.module.KernelSolver(solver=solver, enzyme_width=enzyme_width)
        wrapped = ThreadOwnedKernelSolver(cpp_solver)
        solvers[key] = wrapped
        return wrapped

    def _thread_solver_cache(self) -> dict[tuple[str, str, int], ThreadOwnedKernelSolver]:
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
    return f"veqpy._kernel_cache.k_{safe_id}.veqlib_ext"
