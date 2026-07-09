"""
Package: veqpy.kernels

Role:
- Define the public Kernel runtime handle and typed Kernel contract.
- Dispatch private Cxx and Numba backend implementations from KernelRecipe.backend.

Public API:
- Kernel.
- KernelRecipe, KernelTopology, KernelBoundary, KernelSource, and KernelConfig.
- KernelPrepareResult, SolveResult, ParetoResult, ParetoSample, TopologyError,
  and config_with_overrides.

Dependencies:
- veqpy.kernels.abi for shared route, option, identity, and source-lowering rules.
- veqpy.model and veqpy.numerics are consumed by backend implementation modules.

Downstream:
- veqpy root and veqpy.api re-export the public Kernel surface.
- Tests and benchmarks may import concrete backend modules directly for focused coverage.

Design notes:
- Backend classes are implementation details under cxx_kernel and numba_kernel.
- Package-root imports are the stable user-facing Kernel type surface.
"""

from __future__ import annotations

from .errors import TopologyError
from .kernel import Kernel
from .pareto import KernelParetoSignature, ParetoResult, ParetoSample
from .types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    config_with_overrides,
)

__all__ = [
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelPrepareResult",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "KernelParetoSignature",
    "ParetoResult",
    "ParetoSample",
    "SolveResult",
    "TopologyError",
    "config_with_overrides",
]
