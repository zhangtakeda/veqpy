"""SolveResult adapters for the temporary legacy NumbaKernel path."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from veqlib.facade import KernelConfig, SolveResult
from veqpy.operator import Operator
from veqpy.solver import SolverResult

from ..solver.residual_scale import make_residual_scale


def solve_result_from_legacy(
    solver_result: SolverResult,
    operator: Operator,
    config: KernelConfig,
) -> SolveResult:
    """Map a legacy SolverResult plus final runtime state to Kernel SolveResult."""

    x_final = operator.coerce_x(solver_result.x).copy()
    raw = operator.residual_var(x_final)
    alpha = operator.source_workspace.alpha_state.copy()
    x_reference = operator.coerce_x(solver_result.x0).copy()
    scaled = _scaled_residual_snapshot(raw, x_reference, operator, config)
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
    x_reference: np.ndarray,
    operator: Operator,
    config: KernelConfig,
) -> np.ndarray:
    mode = config.norm
    if mode == "none":
        return raw.copy()
    reference_raw = operator.residual_var(x_reference)
    block_lengths = operator.residual_block_lengths()
    scale = _reference_residual_scale(reference_raw, x_reference, operator, config, block_lengths)
    if scale is None:
        return raw.copy()
    return raw / scale


def _reference_residual_scale(
    reference_raw: np.ndarray,
    x_reference: np.ndarray,
    operator: Operator,
    config: KernelConfig,
    block_lengths: np.ndarray | None,
) -> np.ndarray | None:
    params: dict[str, object] = {}
    if config.norm in {"balanced", "safe"}:
        params.update(
            floor=config.residual_normalization_floor,
            max_ratio=config.residual_normalization_max_ratio,
            huber_tau=config.residual_normalization_huber_tau,
        )
    if config.norm == "safe":
        params.update(
            residual_fun=lambda x: operator.residual_var(operator.coerce_x(x), check=False),
            x_guess=x_reference,
            x_scale=_x_scale_for_reference(operator, x_reference),
            probe_count=config.residual_normalization_probe_count,
            probe_step=config.residual_normalization_probe_step,
            sensitivity_lambda=config.residual_normalization_sensitivity_lambda,
        )
    scale = make_residual_scale(
        config.norm,
        reference_raw,
        None if block_lengths is None else np.asarray(block_lengths, dtype=np.int64),
        **params,
    )
    if scale is None:
        return None
    return np.asarray(scale, dtype=np.float64)


def _x_scale_for_reference(operator: Operator, x_reference: np.ndarray) -> np.ndarray | None:
    from ..solver.solver import _build_x_block_scale_vector

    return _build_x_block_scale_vector(operator, x_reference)
