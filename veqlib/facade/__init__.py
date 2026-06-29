"""Python-facing VEQlib facade APIs.

This package owns typed runtime arguments, artifact lifecycle, and Python handle
helpers for the C++/CMake core under :mod:`veqlib.core`.
"""

from __future__ import annotations

from .affinity import current_cpu_affinity, pinned_cpu
from .builder import KernelArtifact, KernelBuildError, build_kernel, default_kernel_cache_root
from .kernel import Kernel, build, solve
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
from .scan import PayloadSequenceStep, payload_json_with_initial_policy, solve_payload_sequence
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
    "PayloadSequenceStep",
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
    "current_cpu_affinity",
    "default_kernel_cache_root",
    "initial_policy_code",
    "payload_json_with_initial_policy",
    "pinned_cpu",
    "residual_normalization_code",
    "solve",
    "solve_payload_sequence",
    "solver_method_code",
]
