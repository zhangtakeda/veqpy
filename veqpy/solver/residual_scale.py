"""
Module: solver.residual_scale

Role:
- Registry-backed residual normalization scale builders.
- Each strategy registers a ``(residual, block_lengths, **params) -> scale`` builder.

Public API:
- _RESIDUAL_SCALE_BUILDER
- make_residual_scale
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veqpy.base.registry import Registry

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_RESIDUAL_SCALE_BUILDER: Registry[str, Callable] = Registry(str, Callable)
DEFAULT_RESIDUAL_NORMALIZATION = "balance"

# -----------------------------------------------------------------------------
# Mode helpers
# -----------------------------------------------------------------------------


def _mode_is_block_rms(mode: str) -> bool:
    """Check if *mode* maps to the block-RMS (legacy) scale builder."""
    key = mode.lower()
    if key not in _RESIDUAL_SCALE_BUILDER:
        return False
    return _RESIDUAL_SCALE_BUILDER[key] is _build_block_rms_scale


def make_residual_scale(
    mode: str,
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    **params: object,
) -> np.ndarray:
    """Build the residual scale vector for a registered normalization strategy."""
    mode = mode.lower()
    if mode not in _RESIDUAL_SCALE_BUILDER:
        available = ", ".join(sorted(_RESIDUAL_SCALE_BUILDER.registry))
        raise ValueError(f"Unknown residual normalization: {mode!r}. Available: {available}")
    return _RESIDUAL_SCALE_BUILDER[mode](residual, block_lengths, **params)


# -----------------------------------------------------------------------------
# Scale builders (registered)
# -----------------------------------------------------------------------------


@_RESIDUAL_SCALE_BUILDER("block_rms", "fast")
def _build_block_rms_scale(residual: np.ndarray, block_lengths: np.ndarray) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    block_rms = _block_rms_values(residual_eval, lengths_eval)
    if block_rms is None:
        return None

    scale = np.empty_like(residual_eval)
    offset = 0
    for value, length in zip(block_rms, lengths_eval, strict=False):
        block_scale = max(float(value), 1.0)
        scale[offset : offset + int(length)] = block_scale
        offset += int(length)
    return scale


@_RESIDUAL_SCALE_BUILDER("block_huber", "balance", "balanced")
def _build_block_huber_scale(
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    floor: float,
    max_ratio: float,
    huber_tau: float,
    **params: object,
) -> np.ndarray:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim != 1 or residual_eval.size == 0:
        return np.ones_like(residual_eval, dtype=np.float64)

    floor_eval = max(float(floor), np.finfo(np.float64).tiny)
    block_values = _robust_balanced_block_rms_values(
        residual_eval,
        block_lengths,
        huber_tau=float(huber_tau),
    )
    if block_values is None:
        global_scale = _robust_rms(residual_eval, huber_tau=float(huber_tau))
        global_scale = _clip_scale_by_anchor(
            np.asarray([global_scale], dtype=np.float64),
            floor=floor_eval,
            max_ratio=max_ratio,
        )[0]
        return np.full_like(residual_eval, global_scale, dtype=np.float64)

    clipped_values = _clip_scale_by_anchor(block_values, floor=floor_eval, max_ratio=max_ratio)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    scale = np.empty_like(residual_eval, dtype=np.float64)
    offset = 0
    for value, length in zip(clipped_values, lengths_eval, strict=False):
        block_size = int(length)
        scale[offset : offset + block_size] = float(value)
        offset += block_size
    return scale


@_RESIDUAL_SCALE_BUILDER("block_sensitivity", "safe")
def _build_block_sensitivity_scale(
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    residual_fun: Callable[[np.ndarray], np.ndarray],
    x_guess: np.ndarray,
    x_scale: np.ndarray | None,
    floor: float,
    max_ratio: float,
    huber_tau: float,
    probe_count: int,
    probe_step: float,
    sensitivity_lambda: float,
    **params: object,
) -> np.ndarray:
    residual_eval = np.asarray(residual, dtype=np.float64)
    amplitude_values = _robust_balanced_block_rms_values(
        residual_eval,
        block_lengths,
        huber_tau=huber_tau,
    )
    if amplitude_values is None or block_lengths is None or int(probe_count) <= 0:
        return _build_block_huber_scale(
            residual_eval,
            block_lengths,
            floor=floor,
            max_ratio=max_ratio,
            huber_tau=huber_tau,
        )

    x_eval = np.asarray(x_guess, dtype=np.float64)
    if x_scale is None or np.asarray(x_scale).shape != x_eval.shape:
        x_scale_eval = np.maximum(np.abs(x_eval), 1.0)
    else:
        x_scale_eval = np.asarray(x_scale, dtype=np.float64)
    q = int(probe_count)
    step = float(probe_step)
    rng = np.random.default_rng(0)
    sensitivity_sq = np.zeros_like(amplitude_values, dtype=np.float64)
    for _ in range(q):
        signs = rng.integers(0, 2, size=x_eval.shape, dtype=np.int8).astype(np.float64)
        signs = signs * 2.0 - 1.0
        probe_x = x_eval + step * x_scale_eval * signs
        diff = (np.asarray(residual_fun(probe_x), dtype=np.float64) - residual_eval) / step
        probe_values = _robust_balanced_block_rms_values(diff, block_lengths, huber_tau=huber_tau)
        if probe_values is None:
            continue
        sensitivity_sq += probe_values * probe_values
    sensitivity_values = np.sqrt(sensitivity_sq / float(q))
    combined = np.sqrt(
        amplitude_values * amplitude_values + (float(sensitivity_lambda) * sensitivity_values) ** 2
    )
    clipped_values = _clip_scale_by_anchor(combined, floor=floor, max_ratio=max_ratio)

    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    scale = np.empty_like(residual_eval, dtype=np.float64)
    offset = 0
    for value, length in zip(clipped_values, lengths_eval, strict=False):
        block_size = int(length)
        scale[offset : offset + block_size] = float(value)
        offset += block_size
    return scale


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _residual_rms(residual: np.ndarray) -> float:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim != 1 or residual_eval.size == 0:
        return 1.0
    return float(np.linalg.norm(residual_eval) / np.sqrt(residual_eval.size))


def _block_rms_values(residual: np.ndarray, block_lengths: np.ndarray) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    if (
        residual_eval.ndim != 1
        or lengths_eval.ndim != 1
        or residual_eval.size == 0
        or lengths_eval.size == 0
    ):
        return None
    if int(np.sum(lengths_eval)) != int(residual_eval.size):
        return None

    values = np.empty_like(lengths_eval, dtype=np.float64)
    offset = 0
    for idx, length in enumerate(lengths_eval):
        block_size = int(length)
        if block_size <= 0:
            return None
        block = residual_eval[offset : offset + block_size]
        block_rms = float(np.linalg.norm(block) / np.sqrt(block_size))
        if not np.isfinite(block_rms):
            return None
        values[idx] = block_rms
        offset += block_size
    return values


def _robust_balanced_block_rms_values(
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    huber_tau: float,
) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if block_lengths is None:
        return None
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    if (
        residual_eval.ndim != 1
        or lengths_eval.ndim != 1
        or residual_eval.size == 0
        or lengths_eval.size == 0
        or int(np.sum(lengths_eval)) != int(residual_eval.size)
    ):
        return None

    values = np.empty_like(lengths_eval, dtype=np.float64)
    offset = 0
    for idx, length in enumerate(lengths_eval):
        block_size = int(length)
        if block_size <= 0:
            return None
        values[idx] = _robust_rms(
            residual_eval[offset : offset + block_size],
            huber_tau=huber_tau,
        )
        offset += block_size
    return values


def _robust_rms(values: np.ndarray, *, huber_tau: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    cutoff = center + max(float(huber_tau), 0.0) * 1.4826 * mad
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        cutoff = center
    clipped = np.minimum(finite, cutoff)
    return _stable_rms(clipped)


def _stable_rms(values: np.ndarray) -> float:
    """Overflow-resistant RMS of finite entries, with nonfinite values ignored."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    max_abs = float(np.max(finite))
    if max_abs == 0.0:
        return 0.0
    scaled = finite / max_abs
    return float(max_abs * np.sqrt(np.mean(scaled * scaled)))


def _balanced_residual_anchor(values: np.ndarray, *, floor: float) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return float(floor)
    anchor = float(np.median(finite_positive))
    if not np.isfinite(anchor) or anchor < floor:
        return float(floor)
    return anchor


def _clip_scale_by_anchor(values: np.ndarray, *, floor: float, max_ratio: float) -> np.ndarray:
    values_eval = np.asarray(values, dtype=np.float64)
    floor_eval = max(float(floor), np.finfo(np.float64).tiny)
    ratio_eval = max(float(max_ratio), 1.0)
    scale = np.maximum(values_eval, floor_eval)
    scale[~np.isfinite(scale)] = floor_eval
    anchor = _balanced_residual_anchor(scale, floor=floor_eval)
    lower = max(floor_eval, anchor / ratio_eval)
    upper = _balanced_residual_upper_cap(anchor, floor=floor_eval, max_ratio=ratio_eval)
    return np.clip(scale, lower, upper)


def _balanced_residual_upper_cap(anchor: float, *, floor: float, max_ratio: float) -> float:
    cap = float(anchor) * float(max_ratio)
    if not np.isfinite(cap) or cap < floor:
        return float(np.finfo(np.float64).max)
    return max(float(floor), cap)
