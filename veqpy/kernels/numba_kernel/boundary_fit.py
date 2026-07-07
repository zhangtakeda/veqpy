"""
Module: veqpy.kernels.numba_kernel.boundary_fit

Role:
- Provide the Numba backend implementation of boundary phase-QR fitting.

Notes:
- The algorithm mirrors ``veqpy.kernels.boundary_fit`` but keeps the hot loop in
  compiled code for scatter-boundary materialization experiments.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from numba import njit

from veqpy.kernels.boundary_fit import MAX_BOUNDARY_FOURIER_ORDER


def fit_boundary_params_numba(
    R_boundary: Any,
    Z_boundary: Any,
    *,
    c_order: int,
    s_order: int,
    maxtol: float = 1.0e-2,
) -> dict[str, float | np.ndarray]:
    """Fit RZ boundary samples with the Numba phase-QR implementation."""

    R, Z = _coerce_boundary_points(R_boundary, Z_boundary)
    c_order = int(c_order)
    s_order = int(s_order)
    maxtol = float(maxtol)
    if maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")
    if c_order < 0 or s_order < 0:
        raise ValueError(f"c_order and s_order must be non-negative, got {c_order!r}/{s_order!r}")
    if c_order > MAX_BOUNDARY_FOURIER_ORDER or s_order > MAX_BOUNDARY_FOURIER_ORDER:
        raise ValueError(
            f"c_order and s_order must be <= {MAX_BOUNDARY_FOURIER_ORDER}, "
            f"got {c_order!r}/{s_order!r}"
        )
    fit_variables = c_order + 1 + s_order
    if R.size < fit_variables:
        raise ValueError(
            "R_boundary/Z_boundary do not contain enough points for QR boundary fitting: "
            f"need at least {fit_variables}, got {R.size}"
        )

    R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error = _fit_boundary_params_qr_numba(
        R,
        Z,
        c_order,
        s_order,
    )
    if rms >= maxtol:
        warnings.warn(
            (
                f"Boundary fit RMS {float(rms):.6e} exceeds maxtol "
                f"{maxtol:.6e} for c/s orders={c_order}/{s_order}"
            ),
            stacklevel=2,
        )
    return {
        "R0": float(R0),
        "Z0": float(Z0),
        "a": float(a),
        "ka": float(ka),
        "c_offsets": np.asarray(c_offsets, dtype=np.float64),
        "s_offsets": np.asarray(s_offsets, dtype=np.float64),
        "rms": float(rms),
        "max_curve_error": float(max_curve_error),
        "c_order": int(c_order),
        "s_order": int(s_order),
    }


def _coerce_boundary_points(R_boundary: Any, Z_boundary: Any) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R_boundary, dtype=np.float64)
    Z = np.asarray(Z_boundary, dtype=np.float64)
    if R.ndim != 1:
        raise ValueError(f"R_boundary must be 1D, got {R.shape}")
    if Z.ndim != 1:
        raise ValueError(f"Z_boundary must be 1D, got {Z.shape}")
    if R.shape != Z.shape:
        raise ValueError(
            f"R_boundary and Z_boundary must have the same shape, got {R.shape} and {Z.shape}"
        )
    if R.size < 4:
        raise ValueError("R_boundary and Z_boundary must contain at least four points")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError("R_boundary and Z_boundary must contain only finite values")
    return R.astype(np.float64, copy=True), Z.astype(np.float64, copy=True)


@njit(cache=True)
def _fit_boundary_params_qr_numba(
    R: np.ndarray,
    Z: np.ndarray,
    c_order: int,
    s_order: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, float, float]:
    r_min = R[0]
    r_max = R[0]
    z_min = Z[0]
    z_max = Z[0]
    for value in R:
        if value < r_min:
            r_min = value
        if value > r_max:
            r_max = value
    for value in Z:
        if value < z_min:
            z_min = value
        if value > z_max:
            z_max = value

    R0 = 0.5 * (r_max + r_min)
    Z0 = 0.5 * (z_max + z_min)
    a = 0.5 * (r_max - r_min)
    if a <= 0.0:
        raise ValueError("Boundary width must be positive")
    ka = max(0.5 * (z_max - z_min) / a, 1.0e-6)

    r_points, z_points = _ordered_boundary_variant_numba(R, Z)
    theta = _infer_theta_numba(z_points, Z0, a, ka)
    theta_bar_target = _infer_theta_bar_target_numba(r_points, theta, R0, a)
    delta = theta_bar_target - theta
    coefficients = _solve_phase_projection_qr_numba(theta, delta, c_order, s_order)
    c_offsets, s_offsets = _coefficients_to_offsets_numba(coefficients, c_order, s_order)
    fitted_boundary = _build_boundary_numba(R0, Z0, a, ka, c_offsets, s_offsets, theta)
    rms = _rms_r_error_numba(r_points, fitted_boundary)
    max_curve_error = _max_bidirectional_distance_numba(r_points, z_points, fitted_boundary)
    return R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error


@njit(cache=True)
def _ordered_boundary_variant_numba(
    R: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = R.size
    start = 0
    min_z = Z[0]
    for index in range(1, n):
        if Z[index] < min_z:
            min_z = Z[index]
            start = index

    r_ordered = np.empty(n, dtype=np.float64)
    z_ordered = np.empty(n, dtype=np.float64)
    for offset in range(n):
        source = (start + offset) % n
        r_ordered[offset] = R[source]
        z_ordered[offset] = Z[source]

    direction_probe_count = min(5, n - 1)
    if direction_probe_count > 0:
        total = 0.0
        for index in range(1, direction_probe_count + 1):
            total += r_ordered[index] - r_ordered[0]
        if total / direction_probe_count > 0.0:
            reversed_r = np.empty(n, dtype=np.float64)
            reversed_z = np.empty(n, dtype=np.float64)
            reversed_r[0] = r_ordered[0]
            reversed_z[0] = z_ordered[0]
            for index in range(1, n):
                source = n - index
                reversed_r[index] = r_ordered[source]
                reversed_z[index] = z_ordered[source]
            r_ordered = reversed_r
            z_ordered = reversed_z
    return r_ordered, z_ordered


@njit(cache=True)
def _infer_theta_numba(
    z_points: np.ndarray,
    z0: float,
    a_value: float,
    ka: float,
) -> np.ndarray:
    n = z_points.size
    theta = np.empty(n, dtype=np.float64)
    theta[0] = 0.5 * math.pi
    previous = theta[0]
    step = 2.0 * math.pi / max(n, 1)
    scale = a_value * max(ka, 1.0e-6)

    for index in range(1, n):
        sin_theta = -(z_points[index] - z0) / scale
        if sin_theta < -1.0:
            sin_theta = -1.0
        elif sin_theta > 1.0:
            sin_theta = 1.0
        alpha = math.asin(sin_theta)
        candidate0 = _next_phase_candidate_numba(alpha, previous)
        candidate1 = _next_phase_candidate_numba(math.pi - alpha, previous)
        candidate2 = candidate0 + 2.0 * math.pi
        candidate3 = candidate1 + 2.0 * math.pi
        target = previous + step
        best = candidate0
        best_distance = abs(candidate0 - target)
        distance = abs(candidate1 - target)
        if distance < best_distance:
            best = candidate1
            best_distance = distance
        distance = abs(candidate2 - target)
        if distance < best_distance:
            best = candidate2
            best_distance = distance
        distance = abs(candidate3 - target)
        if distance < best_distance:
            best = candidate3
        theta[index] = best
        previous = best
    return theta


@njit(cache=True)
def _next_phase_candidate_numba(candidate: float, previous: float) -> float:
    while candidate < previous - 1.0e-12:
        candidate += 2.0 * math.pi
    return candidate


@njit(cache=True)
def _infer_theta_bar_target_numba(
    r_points: np.ndarray,
    theta: np.ndarray,
    R0: float,
    a: float,
) -> np.ndarray:
    if a <= 0.0:
        raise ValueError("Boundary width must be positive")
    n = r_points.size
    target = np.empty(n, dtype=np.float64)
    max_excess = 0.0
    for index in range(n):
        cosine_target = (r_points[index] - R0) / a
        excess = abs(cosine_target) - 1.0
        if excess > max_excess:
            max_excess = excess
        if cosine_target < -1.0:
            cosine_target = -1.0
        elif cosine_target > 1.0:
            cosine_target = 1.0
        principal = math.acos(cosine_target)
        positive = principal + 2.0 * math.pi * round((theta[index] - principal) / (2.0 * math.pi))
        negative = -principal + 2.0 * math.pi * round(
            (theta[index] + principal) / (2.0 * math.pi)
        )
        if abs(positive - theta[index]) <= abs(negative - theta[index]):
            target[index] = positive
        else:
            target[index] = negative
    if max_excess > 1.0e-8:
        raise ValueError("R_boundary values are inconsistent with R0/a for phase QR fitting")

    unwrapped = np.empty(n, dtype=np.float64)
    unwrapped[0] = target[0]
    for index in range(1, n):
        current = target[index]
        previous = unwrapped[index - 1]
        while current - previous > math.pi:
            current -= 2.0 * math.pi
        while current - previous < -math.pi:
            current += 2.0 * math.pi
        unwrapped[index] = current

    mean_delta = 0.0
    for index in range(n):
        mean_delta += theta[index] - unwrapped[index]
    mean_delta /= max(n, 1)
    shift = 2.0 * math.pi * round(mean_delta / (2.0 * math.pi))
    for index in range(n):
        unwrapped[index] += shift
    return unwrapped


@njit(cache=True)
def _solve_phase_projection_qr_numba(
    theta: np.ndarray,
    delta: np.ndarray,
    c_order: int,
    s_order: int,
) -> np.ndarray:
    matrix = _phase_design_matrix_numba(theta, c_order, s_order)
    n_rows, n_cols = matrix.shape
    q = np.zeros((n_rows, n_cols), dtype=np.float64)
    r = np.zeros((n_cols, n_cols), dtype=np.float64)
    diagonal_max = 0.0

    for col in range(n_cols):
        work = np.empty(n_rows, dtype=np.float64)
        for row in range(n_rows):
            work[row] = matrix[row, col]
        for previous_col in range(col):
            projection = 0.0
            for row in range(n_rows):
                projection += q[row, previous_col] * work[row]
            r[previous_col, col] = projection
            for row in range(n_rows):
                work[row] -= projection * q[row, previous_col]
        norm = 0.0
        for row in range(n_rows):
            norm += work[row] * work[row]
        norm = math.sqrt(norm)
        r[col, col] = norm
        if norm > diagonal_max:
            diagonal_max = norm
        if norm > 0.0:
            for row in range(n_rows):
                q[row, col] = work[row] / norm

    tolerance = np.finfo(np.float64).eps * max(n_rows, n_cols) * max(diagonal_max, 1.0)
    for col in range(n_cols):
        if abs(r[col, col]) <= tolerance:
            raise ValueError("R_boundary/Z_boundary do not provide full-rank phase QR fitting data")

    y = np.empty(n_cols, dtype=np.float64)
    for col in range(n_cols):
        value = 0.0
        for row in range(n_rows):
            value += q[row, col] * delta[row]
        y[col] = value

    x = np.empty(n_cols, dtype=np.float64)
    for row in range(n_cols - 1, -1, -1):
        value = y[row]
        for col in range(row + 1, n_cols):
            value -= r[row, col] * x[col]
        x[row] = value / r[row, row]
    return x


@njit(cache=True)
def _phase_design_matrix_numba(
    theta: np.ndarray,
    c_order: int,
    s_order: int,
) -> np.ndarray:
    n_rows = theta.size
    n_cols = c_order + 1 + s_order
    matrix = np.empty((n_rows, n_cols), dtype=np.float64)
    for row in range(n_rows):
        matrix[row, 0] = 1.0
    col = 1
    for order in range(1, c_order + 1):
        for row in range(n_rows):
            matrix[row, col] = math.cos(order * theta[row])
        col += 1
    for order in range(1, s_order + 1):
        for row in range(n_rows):
            matrix[row, col] = math.sin(order * theta[row])
        col += 1
    return matrix


@njit(cache=True)
def _coefficients_to_offsets_numba(
    coefficients: np.ndarray,
    c_order: int,
    s_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    c_offsets = np.empty(c_order + 1, dtype=np.float64)
    for index in range(c_order + 1):
        c_offsets[index] = coefficients[index]
    s_offsets = np.zeros(s_order + 1, dtype=np.float64)
    for index in range(1, s_order + 1):
        s_offsets[index] = coefficients[c_order + index]

    c_offsets[0] = (c_offsets[0] + math.pi) % (2.0 * math.pi) - math.pi
    if s_offsets.size > 0:
        s_offsets[0] = 0.0
    return c_offsets, s_offsets


@njit(cache=True)
def _build_boundary_numba(
    R0: float,
    Z0: float,
    a: float,
    ka: float,
    c_offsets: np.ndarray,
    s_offsets: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    n = theta.size
    boundary = np.empty((n, 2), dtype=np.float64)
    for row in range(n):
        theta_bar = theta[row] + c_offsets[0]
        for order in range(1, c_offsets.size):
            theta_bar += c_offsets[order] * math.cos(order * theta[row])
        for order in range(1, s_offsets.size):
            theta_bar += s_offsets[order] * math.sin(order * theta[row])
        boundary[row, 0] = R0 + a * math.cos(theta_bar)
        boundary[row, 1] = Z0 - a * ka * math.sin(theta[row])
    return boundary


@njit(cache=True)
def _rms_r_error_numba(r_points: np.ndarray, fitted_boundary: np.ndarray) -> float:
    total = 0.0
    for row in range(r_points.size):
        diff = r_points[row] - fitted_boundary[row, 0]
        total += diff * diff
    return math.sqrt(total / max(r_points.size, 1))


@njit(cache=True)
def _max_bidirectional_distance_numba(
    r_points: np.ndarray,
    z_points: np.ndarray,
    fitted_boundary: np.ndarray,
) -> float:
    n = r_points.size
    max_distance = 0.0
    for row in range(n):
        best = math.inf
        r0 = r_points[row]
        z0 = z_points[row]
        for col in range(n):
            dr = r0 - fitted_boundary[col, 0]
            dz = z0 - fitted_boundary[col, 1]
            distance2 = dr * dr + dz * dz
            if distance2 < best:
                best = distance2
        distance = math.sqrt(best)
        if distance > max_distance:
            max_distance = distance

    for row in range(n):
        best = math.inf
        r0 = fitted_boundary[row, 0]
        z0 = fitted_boundary[row, 1]
        for col in range(n):
            dr = r0 - r_points[col]
            dz = z0 - z_points[col]
            distance2 = dr * dr + dz * dz
            if distance2 < best:
                best = distance2
        distance = math.sqrt(best)
        if distance > max_distance:
            max_distance = distance
    return max_distance
