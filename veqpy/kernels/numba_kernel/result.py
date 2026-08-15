"""
Module: veqpy.kernels.numba_kernel.result

Role:
- Convert Numba runtime outcomes into public ``_SolveSnapshot`` values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm

from veqpy.kernels.types import _BackendConfig, _SolveSnapshot

from .residual_scale import make_residual_scale

if TYPE_CHECKING:
    from .runtime import NumbaRuntime


def solve_result_from_runtime(
    *,
    x0: np.ndarray,
    x: np.ndarray,
    raw: np.ndarray,
    alpha: np.ndarray,
    success: bool,
    nfev: int,
    njev: int,
    iterations: int,
    elapsed_ms: float,
    runtime: NumbaRuntime,
    config: _BackendConfig,
    preprocess_ms: float = 0.0,
    solver_ms: float | None = None,
    postprocess_ms: float = 0.0,
) -> _SolveSnapshot:
    """Build a ``_SolveSnapshot`` from a direct residual runtime."""

    x_final = runtime.coerce_x(x).copy()
    raw_final = np.asarray(raw, dtype=np.float64).copy()
    x_reference = runtime.coerce_x(x0).copy()
    scaled = _scaled_residual_snapshot(raw_final, x_reference, runtime, config)
    return _SolveSnapshot(
        elapsed_ms=float(elapsed_ms),
        success=bool(success),
        info=1 if success else 0,
        nfev=int(nfev),
        njev=int(njev),
        callbacks=0,
        jacobian_component_evaluations=0,
        jvp_evaluations=0,
        linear_iterations=int(iterations),
        raw_norm=float(norm(raw_final)),
        scaled_norm=float(norm(scaled)),
        x=x_final,
        raw=raw_final,
        scaled=scaled,
        alpha=np.asarray(alpha, dtype=np.float64).copy(),
        preprocess_ms=float(preprocess_ms),
        solver_ms=float(elapsed_ms if solver_ms is None else solver_ms),
        postprocess_ms=float(postprocess_ms),
    )


def _scaled_residual_snapshot(
    raw: np.ndarray,
    x_reference: np.ndarray,
    runtime: NumbaRuntime,
    config: _BackendConfig,
) -> np.ndarray:
    mode = config.norm
    if mode == "none":
        return raw.copy()
    reference_raw = _runtime_residual(runtime, x_reference)
    block_lengths = runtime.residual_block_lengths()
    scale = _reference_residual_scale(reference_raw, x_reference, runtime, config, block_lengths)
    if scale is None:
        return raw.copy()
    return raw / scale


def _reference_residual_scale(
    reference_raw: np.ndarray,
    x_reference: np.ndarray,
    runtime: NumbaRuntime,
    config: _BackendConfig,
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
            residual_fun=lambda x: _runtime_residual(runtime, runtime.coerce_x(x)),
            x_guess=x_reference,
            x_scale=_x_scale_for_reference(runtime, x_reference),
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


def _runtime_residual(runtime: NumbaRuntime, x: np.ndarray) -> np.ndarray:
    return runtime.residual_for_current_case(x)


def _x_scale_for_reference(runtime: NumbaRuntime, x_reference: np.ndarray) -> np.ndarray | None:
    x_eval = np.asarray(x_reference, dtype=np.float64)
    scale = np.ones_like(x_eval)
    for _, profile_name, coeff_indices, offset, profile_scale in runtime.active_profile_blocks():
        coeff_indices = np.asarray(coeff_indices, dtype=np.int64)
        length = int(coeff_indices.size)
        if length <= 0:
            continue
        if np.any(coeff_indices < 0) or np.any(coeff_indices >= x_eval.size):
            return None
        block_guess = x_eval[coeff_indices]
        guess_rms = float(np.linalg.norm(block_guess) / np.sqrt(length))
        offset_scale = 0.0 if profile_name in {"h", "v", "psin"} else abs(float(offset))
        profile_scale = abs(float(profile_scale))
        profile_prior = _x_scale_profile_prior(profile_name)
        if abs(profile_scale - 1.0) <= 1.0e-12:
            profile_scale = profile_prior
        block_scale = max(offset_scale, profile_scale, profile_prior, guess_rms, 1.0e-2)
        scale[coeff_indices] = block_scale
    return scale


def _x_scale_profile_prior(name: str) -> float:
    if name in {"h", "v", "psin"}:
        return 1.5e-1
    if name == "k":
        return 1.0
    if name.startswith(("c", "s")):
        return 5.0e-2
    if name == "F":
        return 2.5e-1
    return 5.0e-2
