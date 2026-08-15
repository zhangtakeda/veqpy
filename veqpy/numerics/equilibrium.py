"""Private numerical helpers for materializing frozen base Equilibrium state.

This module contains only radial resampling needed by the Kernel output path.
GEQDSK parsing/export and visualization are intentionally outside VEQPy.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .interpolate import interpolation_matrix


def resample_equilibrium_root_fields(
    *,
    source_grid: Grid,
    target_grid: Grid,
    psin: np.ndarray,
    psin_r: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample solver-owned radial roots without constructing geometry."""

    source_r = source_grid.r
    target_r = target_grid.r
    if float(source_r[0]) > 1.0e-14 or float(source_r[-1]) < 1.0 - 1.0e-14:
        remap = interpolation_matrix(source_r, target_r)
        psin_out = remap @ psin
        psin_r_out = remap @ psin_r
        FFn_out = remap @ FFn_psin
        Pn_out = remap @ Pn_psin
        psin_rr_out = target_grid.differentiate(psin_r_out)
        _extend_psin_to_missing_axis(
            source_r,
            target_r,
            psin,
            psin_r,
            psin_out,
            psin_r_out,
            psin_rr_out,
        )
    else:
        psin_out = _resample_profile_linear(source_r, psin, target_r)
        psin_r_out = _resample_profile_linear(source_r, psin_r, target_r)
        FFn_out = _resample_profile_linear(source_r, FFn_psin, target_r)
        Pn_out = _resample_profile_linear(source_r, Pn_psin, target_r)
        psin_rr_out = target_grid.differentiate(psin_r_out)
    return psin_out, psin_r_out, psin_rr_out, FFn_out, Pn_out


def _extend_psin_to_missing_axis(
    source_r: np.ndarray,
    target_r: np.ndarray,
    source_psin: np.ndarray,
    source_psin_r: np.ndarray,
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
) -> None:
    """Restore the regular even flux-coordinate limit at a missing axis."""

    if source_r.size < 2 or float(source_r[0]) <= 1.0e-14:
        return
    r0 = float(source_r[0])
    r1 = float(source_r[1])
    axis = target_r < r1
    if not np.any(axis) or r1 <= r0:
        return
    psin1 = float(source_psin[1])
    psin_r0 = float(source_psin_r[0])
    psin_r1 = float(source_psin_r[1])
    if not np.isfinite(psin1) or not np.isfinite(psin_r0) or not np.isfinite(psin_r1):
        return
    r0_sq = r0 * r0
    r1_sq = r1 * r1
    ratio0 = psin_r0 / r0
    ratio1 = psin_r1 / r1
    ratio_gradient = (ratio1 - ratio0) / (r1_sq - r0_sq)
    ratio_axis = ratio0 - ratio_gradient * r0_sq
    r_axis = target_r[axis]
    r_axis_sq = r_axis * r_axis
    out_psin[axis] = r_axis_sq * (0.5 * ratio_axis + 0.25 * ratio_gradient * r_axis_sq)
    out_psin_r[axis] = r_axis * (ratio_axis + ratio_gradient * r_axis_sq)
    out_psin_rr[axis] = ratio_axis + 3.0 * ratio_gradient * r_axis_sq
    psin_at_r1 = r1_sq * (0.5 * ratio_axis + 0.25 * ratio_gradient * r1_sq)
    out_psin[~axis] += psin_at_r1 - psin1


def _resample_profile_linear(
    r_src: np.ndarray,
    values: np.ndarray,
    r_eval: np.ndarray,
) -> np.ndarray:
    r_src = np.asarray(r_src, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    r_eval = np.asarray(r_eval, dtype=np.float64)
    if r_src.ndim != 1 or values.ndim != 1 or r_eval.ndim != 1 or r_src.shape != values.shape:
        raise ValueError("expected 1D source/evaluation arrays with matching source shape")
    return np.interp(r_eval, r_src, values, left=float(values[0]), right=float(values[-1]))


__all__ = ["resample_equilibrium_root_fields"]
