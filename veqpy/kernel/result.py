"""SolveResult adapters for the temporary legacy NumbaKernel path."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from veqlib.facade import KernelConfig, SolveResult
from veqpy.operator import Operator
from veqpy.solver import SolverResult


def solve_result_from_legacy(
    solver_result: SolverResult,
    operator: Operator,
    config: KernelConfig,
) -> SolveResult:
    """Map a legacy SolverResult plus final runtime state to Kernel SolveResult."""

    x_final = operator.coerce_x(solver_result.x).copy()
    raw = operator.residual_var(x_final)
    alpha = operator.source_workspace.alpha_state.copy()
    scaled = _scaled_residual_snapshot(raw, operator, config)
    return SolveResult(
        elapsed_ms=float(solver_result.elapsed) / 1000.0,
        success=solver_result.success,
        info=1 if solver_result.success else 0,
        nfev=solver_result.function_evaluations,
        njev=solver_result.jacobian_evaluations,
        callbacks=0,
        jacobian_component_evaluations=0,
        jvp_evaluations=0,
        linear_iterations=solver_result.iterations,
        raw_norm=float(norm(raw)),
        scaled_norm=float(norm(scaled)),
        x=x_final,
        raw=raw,
        scaled=scaled,
        alpha=alpha,
    )


def _scaled_residual_snapshot(
    raw: np.ndarray,
    operator: Operator,
    config: KernelConfig,
) -> np.ndarray:
    mode = config.norm
    if mode == "none":
        return raw.copy()
    block_lengths = operator.residual_block_lengths()
    if block_lengths is None:
        scale = max(float(norm(raw)) / np.sqrt(max(raw.size, 1)), 1.0)
        return raw / scale
    scale = _block_scale(raw, np.asarray(block_lengths, dtype=np.int64))
    return raw / scale


def _block_scale(raw: np.ndarray, block_lengths: np.ndarray) -> np.ndarray:
    scale = np.ones_like(raw, dtype=np.float64)
    offset = 0
    for length in block_lengths:
        block_size = int(length)
        block = raw[offset : offset + block_size]
        if block_size > 0:
            value = max(float(norm(block)) / np.sqrt(block_size), 1.0)
            scale[offset : offset + block_size] = value
        offset += block_size
    return scale
