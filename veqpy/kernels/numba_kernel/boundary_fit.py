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

from veqpy.kernels.boundary_fit import (
    MAX_BOUNDARY_FOURIER_ORDER,
    normalize_boundary_fit_method,
)

_WEIGHTED_QR_WEIGHT_FLOOR = 1.0e-2


def fit_boundary_params_numba(
    R_boundary: Any,
    Z_boundary: Any,
    *,
    c_order: int,
    s_order: int,
    maxtol: float = 1.0e-2,
    method: str | None = "gnqr",
) -> dict[str, float | np.ndarray]:
    """Fit RZ boundary samples with the Numba boundary fitter."""

    R, Z = _coerce_boundary_points(R_boundary, Z_boundary)
    method = normalize_boundary_fit_method(method)
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

    if method == "qr":
        R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error = (
            _fit_boundary_params_weighted_qr_numba(R, Z, c_order, s_order, 0)
        )
    elif method == "gnqr":
        R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error = (
            _fit_boundary_params_weighted_qr_numba(R, Z, c_order, s_order, 2)
        )
    elif method == "least-square":
        R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error = (
            _fit_boundary_params_least_square_numba(R, Z, c_order, s_order)
        )
    else:
        raise AssertionError(f"unhandled boundary fit method {method!r}")
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
        "method": method,
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
def _fit_boundary_params_weighted_qr_numba(
    R: np.ndarray,
    Z: np.ndarray,
    c_order: int,
    s_order: int,
    gn_steps: int,
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
    matrix = _phase_design_matrix_numba(theta, c_order, s_order)
    coefficients = _solve_weighted_phase_projection_qr_numba(theta, theta_bar_target, matrix, a)
    for _ in range(gn_steps):
        candidate, accepted = _apply_gnqr_step_numba(coefficients, r_points, theta, matrix, R0, a)
        coefficients = candidate
        if not accepted:
            break
    c_offsets, s_offsets = _coefficients_to_offsets_numba(coefficients, c_order, s_order)
    fitted_boundary = _build_boundary_numba(R0, Z0, a, ka, c_offsets, s_offsets, theta)
    rms = _rms_r_error_numba(r_points, fitted_boundary)
    max_curve_error = _max_bidirectional_distance_numba(r_points, z_points, fitted_boundary)
    return R0, Z0, a, ka, c_offsets, s_offsets, rms, max_curve_error


@njit(cache=True)
def _fit_boundary_params_least_square_numba(
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
    lower, upper = _least_square_bounds_numba(
        r_points,
        z_points,
        a,
        c_order,
        s_order,
    )
    vector = np.zeros(4 + c_order + 1 + s_order, dtype=np.float64)
    vector[0] = R0
    vector[1] = Z0
    vector[2] = a
    vector[3] = ka
    vector = _clip_vector_numba(vector, lower, upper)
    vector = _bounded_least_square_lm_numba(
        vector,
        lower,
        upper,
        r_points,
        z_points,
        c_order,
        s_order,
        40,
    )
    R0, Z0, a, ka, c_offsets, s_offsets = _unpack_least_square_vector_numba(
        vector,
        c_order,
        s_order,
    )
    c_offsets, s_offsets = _normalize_offsets_numba(c_offsets, s_offsets)
    theta = _infer_theta_numba(z_points, Z0, a, ka)
    fitted_boundary = _build_boundary_numba(R0, Z0, a, ka, c_offsets, s_offsets, theta)
    residual = _least_square_residual_numba(vector, r_points, z_points, c_order, s_order)
    rms = _residual_rms_numba(residual)
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
    return _solve_matrix_qr_numba(matrix, delta)


@njit(cache=True)
def _solve_weighted_phase_projection_qr_numba(
    theta: np.ndarray,
    theta_bar_target: np.ndarray,
    matrix: np.ndarray,
    a: float,
) -> np.ndarray:
    n_rows, n_cols = matrix.shape
    weighted_matrix = np.empty((n_rows, n_cols), dtype=np.float64)
    weighted_delta = np.empty(n_rows, dtype=np.float64)
    weight_floor = max(a * _WEIGHTED_QR_WEIGHT_FLOOR, 0.0)
    for row in range(n_rows):
        weight = abs(a * math.sin(theta_bar_target[row]))
        if weight < weight_floor:
            weight = weight_floor
        weighted_delta[row] = (theta_bar_target[row] - theta[row]) * weight
        for col in range(n_cols):
            weighted_matrix[row, col] = matrix[row, col] * weight
    return _solve_matrix_qr_numba(weighted_matrix, weighted_delta)


@njit(cache=True)
def _solve_matrix_qr_numba(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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
            value += q[row, col] * rhs[row]
        y[col] = value

    x = np.empty(n_cols, dtype=np.float64)
    for row in range(n_cols - 1, -1, -1):
        value = y[row]
        for col in range(row + 1, n_cols):
            value -= r[row, col] * x[col]
        x[row] = value / r[row, row]
    return x


@njit(cache=True)
def _apply_gnqr_step_numba(
    coefficients: np.ndarray,
    r_points: np.ndarray,
    theta: np.ndarray,
    matrix: np.ndarray,
    R0: float,
    a: float,
) -> tuple[np.ndarray, bool]:
    n_rows, n_cols = matrix.shape
    jacobian = np.empty((n_rows, n_cols), dtype=np.float64)
    rhs = np.empty(n_rows, dtype=np.float64)
    residual_total = 0.0
    for row in range(n_rows):
        theta_bar = theta[row]
        for col in range(n_cols):
            theta_bar += matrix[row, col] * coefficients[col]
        residual = r_points[row] - (R0 + a * math.cos(theta_bar))
        scale = a * math.sin(theta_bar)
        rhs[row] = -residual
        residual_total += residual * residual
        for col in range(n_cols):
            jacobian[row, col] = scale * matrix[row, col]
    step = _solve_matrix_qr_numba(jacobian, rhs)
    current_objective = residual_total / max(n_rows, 1)
    for damping_index in range(5):
        damping = 1.0 / (2.0**damping_index)
        candidate = np.empty(n_cols, dtype=np.float64)
        for col in range(n_cols):
            candidate[col] = coefficients[col] + damping * step[col]
        candidate_objective = _r_objective_numba(r_points, theta, matrix, candidate, R0, a)
        if candidate_objective < current_objective:
            return candidate, True
    return coefficients, False


@njit(cache=True)
def _r_objective_numba(
    r_points: np.ndarray,
    theta: np.ndarray,
    matrix: np.ndarray,
    coefficients: np.ndarray,
    R0: float,
    a: float,
) -> float:
    n_rows, n_cols = matrix.shape
    total = 0.0
    for row in range(n_rows):
        theta_bar = theta[row]
        for col in range(n_cols):
            theta_bar += matrix[row, col] * coefficients[col]
        residual = r_points[row] - (R0 + a * math.cos(theta_bar))
        total += residual * residual
    return total / max(n_rows, 1)


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
def _least_square_bounds_numba(
    r_points: np.ndarray,
    z_points: np.ndarray,
    initial_a: float,
    c_order: int,
    s_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    r_min = r_points[0]
    r_max = r_points[0]
    z_min = z_points[0]
    z_max = z_points[0]
    for value in r_points:
        if value < r_min:
            r_min = value
        if value > r_max:
            r_max = value
    for value in z_points:
        if value < z_min:
            z_min = value
        if value > z_max:
            z_max = value
    span_r = r_max - r_min
    span_z = z_max - z_min
    variable_count = 4 + c_order + 1 + s_order
    lower = np.empty(variable_count, dtype=np.float64)
    upper = np.empty(variable_count, dtype=np.float64)
    lower[0] = r_min - 0.25 * span_r
    upper[0] = r_max + 0.25 * span_r
    lower[1] = z_min - 0.25 * span_z
    upper[1] = z_max + 0.25 * span_z
    lower[2] = max(1.0e-6, 0.25 * initial_a)
    upper[2] = max(max(4.0 * initial_a, span_z), 1.0)
    lower[3] = 1.0e-6
    upper[3] = 10.0
    for index in range(4, variable_count):
        lower[index] = -10.0
        upper[index] = 10.0
    return lower, upper


@njit(cache=True)
def _clip_vector_numba(vector: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    out = np.empty(vector.size, dtype=np.float64)
    for index in range(vector.size):
        value = vector[index]
        if value < lower[index]:
            value = lower[index]
        elif value > upper[index]:
            value = upper[index]
        out[index] = value
    return out


@njit(cache=True)
def _unpack_least_square_vector_numba(
    vector: np.ndarray,
    c_order: int,
    s_order: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    R0 = vector[0]
    Z0 = vector[1]
    a = vector[2]
    ka = vector[3]
    c_offsets = np.empty(c_order + 1, dtype=np.float64)
    for index in range(c_order + 1):
        c_offsets[index] = vector[4 + index]
    s_offsets = np.zeros(s_order + 1, dtype=np.float64)
    for index in range(1, s_order + 1):
        s_offsets[index] = vector[4 + c_order + index]
    return R0, Z0, a, ka, c_offsets, s_offsets


@njit(cache=True)
def _normalize_offsets_numba(
    c_offsets: np.ndarray,
    s_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c_out = c_offsets.copy()
    s_out = s_offsets.copy()
    c_out[0] = (c_out[0] + math.pi) % (2.0 * math.pi) - math.pi
    if s_out.size > 0:
        s_out[0] = 0.0
    return c_out, s_out


@njit(cache=True)
def _least_square_residual_numba(
    vector: np.ndarray,
    r_points: np.ndarray,
    z_points: np.ndarray,
    c_order: int,
    s_order: int,
) -> np.ndarray:
    R0, Z0, a, ka, c_offsets, s_offsets = _unpack_least_square_vector_numba(
        vector,
        c_order,
        s_order,
    )
    theta = _infer_theta_numba(z_points, Z0, a, ka)
    residual = np.empty(r_points.size * 2, dtype=np.float64)
    for row in range(r_points.size):
        theta_bar = theta[row] + c_offsets[0]
        for order in range(1, c_offsets.size):
            theta_bar += c_offsets[order] * math.cos(order * theta[row])
        for order in range(1, s_offsets.size):
            theta_bar += s_offsets[order] * math.sin(order * theta[row])
        fitted_r = R0 + a * math.cos(theta_bar)
        fitted_z = Z0 - a * ka * math.sin(theta[row])
        residual[row] = r_points[row] - fitted_r
        residual[r_points.size + row] = z_points[row] - fitted_z
    return residual


@njit(cache=True)
def _residual_rms_numba(residual: np.ndarray) -> float:
    total = 0.0
    for value in residual:
        total += value * value
    return math.sqrt(total / max(residual.size, 1))


@njit(cache=True)
def _least_square_objective_numba(residual: np.ndarray) -> float:
    total = 0.0
    for value in residual:
        total += value * value
    return total / max(residual.size, 1)


@njit(cache=True)
def _bounded_least_square_lm_numba(
    vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    r_points: np.ndarray,
    z_points: np.ndarray,
    c_order: int,
    s_order: int,
    max_iterations: int,
) -> np.ndarray:
    x = _clip_vector_numba(vector, lower, upper)
    variable_count = x.size
    lambda_value = 1.0e-4
    for _ in range(max_iterations):
        residual = _least_square_residual_numba(x, r_points, z_points, c_order, s_order)
        current_objective = _least_square_objective_numba(residual)
        row_count = residual.size
        jacobian = np.empty((row_count, variable_count), dtype=np.float64)
        for col in range(variable_count):
            step_size = 1.0e-6 * max(abs(x[col]), 1.0)
            trial = x.copy()
            trial[col] = min(upper[col], x[col] + step_size)
            actual_step = trial[col] - x[col]
            if abs(actual_step) < 1.0e-14:
                trial[col] = max(lower[col], x[col] - step_size)
                actual_step = trial[col] - x[col]
            if abs(actual_step) < 1.0e-14:
                for row in range(row_count):
                    jacobian[row, col] = 0.0
                continue
            trial_residual = _least_square_residual_numba(
                trial,
                r_points,
                z_points,
                c_order,
                s_order,
            )
            for row in range(row_count):
                jacobian[row, col] = (trial_residual[row] - residual[row]) / actual_step

        accepted = False
        for _attempt in range(8):
            augmented_rows = row_count + variable_count
            augmented = np.zeros((augmented_rows, variable_count), dtype=np.float64)
            rhs = np.zeros(augmented_rows, dtype=np.float64)
            for row in range(row_count):
                rhs[row] = -residual[row]
                for col in range(variable_count):
                    augmented[row, col] = jacobian[row, col]
            damping = math.sqrt(lambda_value)
            for col in range(variable_count):
                augmented[row_count + col, col] = damping
            step = _solve_matrix_qr_numba(augmented, rhs)
            candidate = np.empty(variable_count, dtype=np.float64)
            step_norm = 0.0
            x_norm = 0.0
            for col in range(variable_count):
                candidate[col] = x[col] + step[col]
                if candidate[col] < lower[col]:
                    candidate[col] = lower[col]
                elif candidate[col] > upper[col]:
                    candidate[col] = upper[col]
                actual = candidate[col] - x[col]
                step_norm += actual * actual
                x_norm += x[col] * x[col]
            candidate_residual = _least_square_residual_numba(
                candidate,
                r_points,
                z_points,
                c_order,
                s_order,
            )
            candidate_objective = _least_square_objective_numba(candidate_residual)
            if candidate_objective < current_objective:
                x = candidate
                lambda_value = max(lambda_value * 0.3, 1.0e-12)
                accepted = True
                if math.sqrt(step_norm) <= 1.0e-9 * (math.sqrt(x_norm) + 1.0):
                    return x
                if current_objective - candidate_objective <= 1.0e-14 * max(current_objective, 1.0):
                    return x
                break
            lambda_value = min(lambda_value * 10.0, 1.0e12)
        if not accepted:
            return x
    return x


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
