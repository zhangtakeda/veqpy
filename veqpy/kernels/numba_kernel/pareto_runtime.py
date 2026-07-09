"""
Module: veqpy.kernels.numba_kernel.pareto_runtime

Role:
- Provide lightweight Numba-runtime geometry snapshots for Pareto searches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.kernels.types import KernelBoundary, KernelSource

    from .runtime import NumbaRuntime


def sample_r_surface(
    runtime: NumbaRuntime,
    x: np.ndarray,
    boundary: KernelBoundary,
    source: KernelSource,
) -> np.ndarray:
    """Return major-radius samples on the current solve grid without Equilibrium."""

    runtime.set_case(boundary, source)
    x_eval = runtime.coerce_x(x)
    out = np.empty(runtime.plan.x_size, dtype=np.float64)
    runtime.layout.run_fused_residual_into(x_eval, out)
    return runtime.geometry_workspace.R_surface.copy()
