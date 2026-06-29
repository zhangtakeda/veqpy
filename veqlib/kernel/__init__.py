"""Pure Python-facing VEQlib kernel API.

This package is intentionally aligned with the native C++/nanobind runtime and
must not depend on VEQPy model/operator internals.
"""

from __future__ import annotations

from .builder import KernelArtifact, KernelBuildError, build_kernel, default_kernel_cache_root
from .kernel import Kernel, build
from .options import (
    INITIAL_POLICY_COLD,
    INITIAL_POLICY_COLD_GEOMETRIC,
    INITIAL_POLICY_COLD_ZEROS,
    INITIAL_POLICY_WARM_CLONE,
    RESIDUAL_NORMALIZATION_BALANCED,
    RESIDUAL_NORMALIZATION_FAST,
    RESIDUAL_NORMALIZATION_NONE,
    RESIDUAL_NORMALIZATION_SAFE,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_NEWTON_KRYLOV,
    SOLVER_METHOD_NEWTON_RAPHSON,
    SOLVER_METHOD_POWELL,
    initial_policy_code,
    residual_normalization_code,
    solver_method_code,
)
from .registry import KernelLoadError, KernelRegistry, LoadedKernel, SolverThreadError
from .solver import VEQlibSolver
from .types import (
    KernelBoundary,
    KernelBuild,
    KernelInput,
    KernelResult,
    KernelSolve,
    KernelTopology,
    TopologyError,
)

__all__ = [
    "INITIAL_POLICY_COLD",
    "INITIAL_POLICY_COLD_GEOMETRIC",
    "INITIAL_POLICY_COLD_ZEROS",
    "INITIAL_POLICY_WARM_CLONE",
    "Kernel",
    "KernelArtifact",
    "KernelBoundary",
    "KernelBuild",
    "KernelBuildError",
    "KernelInput",
    "KernelLoadError",
    "KernelRegistry",
    "KernelResult",
    "KernelSolve",
    "KernelTopology",
    "LoadedKernel",
    "RESIDUAL_NORMALIZATION_BALANCED",
    "RESIDUAL_NORMALIZATION_FAST",
    "RESIDUAL_NORMALIZATION_NONE",
    "RESIDUAL_NORMALIZATION_SAFE",
    "SOLVER_METHOD_LEVENBERG_MARQUARDT",
    "SOLVER_METHOD_NEWTON_KRYLOV",
    "SOLVER_METHOD_NEWTON_RAPHSON",
    "SOLVER_METHOD_POWELL",
    "SolverThreadError",
    "TopologyError",
    "VEQlibSolver",
    "build",
    "build_kernel",
    "default_kernel_cache_root",
    "initial_policy_code",
    "residual_normalization_code",
    "solver_method_code",
]
