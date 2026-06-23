from __future__ import annotations

from .benchmark import (
    LifecycleBenchmarkConfig,
    benchmark_kernel_lifecycle,
    benchmark_kernel_lifecycle_json,
)
from .kernel_builder import (
    KernelArtifact,
    KernelBuildError,
    build_kernel,
    default_kernel_cache_root,
)
from .kernel_registry import (
    KernelLoadError,
    KernelRegistry,
    LoadedKernel,
    SolverThreadError,
    ThreadOwnedKernelSolver,
    load_kernel,
)
from .legacy_compare import LegacyCompareConfig, benchmark_legacy_veqpy_comparison
from .options import (
    INITIAL_POLICY_COLD,
    INITIAL_POLICY_COLD_GEOMETRIC,
    INITIAL_POLICY_COLD_ZEROS,
    INITIAL_POLICY_WARM_CLONE,
    RESIDUAL_NORMALIZATION_BLOCK_RMS,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_POWELL,
    initial_policy_code,
    residual_normalization_code,
    solver_method_code,
)
from .solver import VEQlibSolver

__all__ = [
    "INITIAL_POLICY_COLD",
    "INITIAL_POLICY_COLD_GEOMETRIC",
    "INITIAL_POLICY_COLD_ZEROS",
    "INITIAL_POLICY_WARM_CLONE",
    "KernelArtifact",
    "LegacyCompareConfig",
    "LifecycleBenchmarkConfig",
    "KernelBuildError",
    "KernelLoadError",
    "KernelRegistry",
    "LoadedKernel",
    "RESIDUAL_NORMALIZATION_BLOCK_RMS",
    "SOLVER_METHOD_LEVENBERG_MARQUARDT",
    "SOLVER_METHOD_POWELL",
    "SolverThreadError",
    "ThreadOwnedKernelSolver",
    "VEQlibSolver",
    "benchmark_kernel_lifecycle",
    "benchmark_legacy_veqpy_comparison",
    "benchmark_kernel_lifecycle_json",
    "build_kernel",
    "default_kernel_cache_root",
    "initial_policy_code",
    "load_kernel",
    "residual_normalization_code",
    "solver_method_code",
]
