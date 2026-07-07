"""
Module: veqpy.kernels.boundary_fit

Role:
- Fit discrete RZ boundary points into KernelBoundary-compatible parameters.

Notes:
- This module is an internal Kernel helper, not a model-layer object.
- The baseline fitter is a deterministic phase projection solved by QR.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

MAX_BOUNDARY_FOURIER_ORDER = 20


def fit_boundary_params(
    R_boundary: Any,
    Z_boundary: Any,
    *,
    c_order: int | None,
    s_order: int | None,
    maxtol: float = 1.0e-2,
    R0: float | None = None,
    Z0: float | None = None,
    a: float | None = None,
    ka: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Fit RZ boundary samples to Kernel boundary parameters.

    ``c_order`` and ``s_order`` are the highest cosine and sine Fourier orders.
    When both are omitted, the fitter searches for the smallest order satisfying
    ``maxtol``.
    """
    R, Z = _coerce_boundary_points(R_boundary, Z_boundary)
    maxtol = float(maxtol)
    if maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")

    if (c_order is None) != (s_order is None):
        raise ValueError("c_order and s_order must be provided together or both omitted")
    if c_order is None and s_order is None:
        return _fit_minimal_order_boundary(R, Z, maxtol=maxtol, R0=R0, Z0=Z0, a=a, ka=ka)

    assert c_order is not None and s_order is not None
    params = _fit_boundary_for_orders(
        R,
        Z,
        c_order=int(c_order),
        s_order=int(s_order),
        R0=R0,
        Z0=Z0,
        a=a,
        ka=ka,
    )
    if params["rms"] >= maxtol:
        warnings.warn(
            (
                f"Boundary fit RMS {float(params['rms']):.6e} exceeds maxtol "
                f"{maxtol:.6e} for c/s orders={c_order}/{s_order}"
            ),
            stacklevel=2,
        )
    return params


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


def _fit_minimal_order_boundary(
    R: np.ndarray,
    Z: np.ndarray,
    *,
    maxtol: float,
    R0: float | None,
    Z0: float | None,
    a: float | None,
    ka: float | None,
) -> dict[str, float | np.ndarray]:
    best = None
    curve_tol = max(maxtol * 0.25, 1.0e-6)

    for step in range(MAX_BOUNDARY_FOURIER_ORDER):
        c_order = step
        s_order = step + 1
        params = _fit_boundary_for_orders(
            R,
            Z,
            c_order=c_order,
            s_order=s_order,
            R0=R0,
            Z0=Z0,
            a=a,
            ka=ka,
        )
        if best is None or params["rms"] < best["rms"]:
            best = params
        if params["rms"] < maxtol and params["max_curve_error"] < curve_tol:
            return params

    if best is None:
        raise RuntimeError("Boundary fitting failed to produce any candidate.")
    return best


def _fit_boundary_for_orders(
    R: np.ndarray,
    Z: np.ndarray,
    *,
    c_order: int,
    s_order: int,
    R0: float | None,
    Z0: float | None,
    a: float | None,
    ka: float | None,
) -> dict[str, float | np.ndarray]:
    _validate_orders(c_order, s_order)

    r_min = float(np.nanmin(R))
    r_max = float(np.nanmax(R))
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    r_mid = 0.5 * (r_max + r_min)
    z_mid = 0.5 * (z_max + z_min)
    span_r = r_max - r_min
    span_z = z_max - z_min

    initial_R0 = float(R0) if R0 is not None else r_mid
    initial_Z0 = float(Z0) if Z0 is not None else z_mid
    initial_a = float(a) if a is not None else 0.5 * span_r
    if initial_a <= 0.0:
        raise ValueError("Boundary width must be positive")
    ka0 = max(float(ka) if ka is not None else float(0.5 * span_z / initial_a), 1.0e-6)
    fit_variables = c_order + 1 + s_order
    if R.size < fit_variables:
        raise ValueError(
            "R_boundary/Z_boundary do not contain enough points for QR boundary fitting: "
            f"need at least {fit_variables}, got {R.size}"
        )

    r_points, z_points = _ordered_boundary_variant(R, Z)
    start = {
        "R0": initial_R0,
        "Z0": initial_Z0,
        "a": initial_a,
        "ka": ka0,
        "c_offsets": np.zeros(c_order + 1, dtype=np.float64),
        "s_offsets": np.zeros(s_order + 1, dtype=np.float64),
    }
    fit = _fit_boundary_variant(
        r_points,
        z_points,
        start=start,
        c_order=c_order,
        s_order=s_order,
    )
    fitted = fit["params"]
    fitted_boundary = _evaluate_boundary_fit(r_points, z_points, fitted)
    c_offsets, s_offsets = _normalize_fitted_offsets(fitted["c_offsets"], fitted["s_offsets"])

    return {
        "R0": float(fitted["R0"]),
        "Z0": float(fitted["Z0"]),
        "a": float(fitted["a"]),
        "ka": float(fitted["ka"]),
        "c_offsets": c_offsets,
        "s_offsets": s_offsets,
        "rms": float(fit["rms"]),
        "max_curve_error": _max_bidirectional_distance(
            np.column_stack((r_points, z_points)), fitted_boundary
        ),
        "c_order": int(c_order),
        "s_order": int(s_order),
    }


def _validate_orders(c_order: int, s_order: int) -> None:
    if int(c_order) < 0 or int(s_order) < 0:
        raise ValueError(f"c_order and s_order must be non-negative, got {c_order!r}/{s_order!r}")
    if int(c_order) > MAX_BOUNDARY_FOURIER_ORDER or int(s_order) > MAX_BOUNDARY_FOURIER_ORDER:
        raise ValueError(
            f"c_order and s_order must be <= {MAX_BOUNDARY_FOURIER_ORDER}, "
            f"got {c_order!r}/{s_order!r}"
        )


def _ordered_boundary_variant(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = int(np.argmin(Z))
    r_ordered = np.roll(R, -start)
    z_ordered = np.roll(Z, -start)
    direction_probe_count = min(5, r_ordered.size - 1)
    if direction_probe_count > 0:
        mean_dr = float(np.mean(r_ordered[1 : direction_probe_count + 1] - r_ordered[0]))
        if mean_dr > 0.0:
            r_ordered = np.concatenate(([r_ordered[0]], r_ordered[:0:-1]))
            z_ordered = np.concatenate(([z_ordered[0]], z_ordered[:0:-1]))
    return r_ordered, z_ordered


def _evaluate_boundary_fit(
    r_points: np.ndarray,
    z_points: np.ndarray,
    params: dict[str, float | np.ndarray],
) -> np.ndarray:
    theta = _infer_theta(z_points, float(params["Z0"]), float(params["a"]), float(params["ka"]))
    return _build_boundary(
        R0=float(params["R0"]),
        Z0=float(params["Z0"]),
        a=float(params["a"]),
        ka=float(params["ka"]),
        c_offsets=np.asarray(params["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(params["s_offsets"], dtype=np.float64),
        theta=theta,
    )


def _fit_boundary_variant(
    r_points: np.ndarray,
    z_points: np.ndarray,
    *,
    start: dict[str, float | np.ndarray],
    c_order: int,
    s_order: int,
) -> dict[str, float | dict[str, float | np.ndarray]]:
    start = dict(start)
    theta = _infer_theta(
        z_points,
        float(start["Z0"]),
        float(start["a"]),
        float(start["ka"]),
    )
    theta_bar_target = _infer_theta_bar_target(
        r_points,
        theta,
        R0=float(start["R0"]),
        a=float(start["a"]),
    )
    coefficients = _solve_phase_projection_qr(
        theta,
        theta_bar_target - theta,
        c_order=c_order,
        s_order=s_order,
    )
    c_offsets, s_offsets = _coefficients_to_offsets(
        coefficients,
        c_order=c_order,
        s_order=s_order,
    )
    params = {
        "R0": float(start["R0"]),
        "Z0": float(start["Z0"]),
        "a": float(start["a"]),
        "ka": float(start["ka"]),
        "c_offsets": c_offsets,
        "s_offsets": s_offsets,
    }
    fitted_boundary = _build_boundary(
        R0=float(params["R0"]),
        Z0=float(params["Z0"]),
        a=float(params["a"]),
        ka=float(params["ka"]),
        c_offsets=c_offsets,
        s_offsets=s_offsets,
        theta=theta,
    )
    return {
        "rms": float(np.sqrt(np.mean((r_points - fitted_boundary[:, 0]) ** 2))),
        "params": params,
    }


def _infer_theta_bar_target(
    r_points: np.ndarray,
    theta: np.ndarray,
    *,
    R0: float,
    a: float,
) -> np.ndarray:
    if a <= 0.0:
        raise ValueError("Boundary width must be positive")
    cosine_target = (np.asarray(r_points, dtype=np.float64) - float(R0)) / float(a)
    excess = np.maximum(np.abs(cosine_target) - 1.0, 0.0)
    max_excess = float(np.max(excess)) if excess.size else 0.0
    if max_excess > 1.0e-8:
        raise ValueError(
            "R_boundary values are inconsistent with R0/a for phase QR fitting: "
            f"max normalized excess is {max_excess:.6e}"
        )

    principal = np.arccos(np.clip(cosine_target, -1.0, 1.0))
    positive = principal + 2.0 * np.pi * np.round((theta - principal) / (2.0 * np.pi))
    negative = -principal + 2.0 * np.pi * np.round((theta + principal) / (2.0 * np.pi))
    target = np.where(np.abs(positive - theta) <= np.abs(negative - theta), positive, negative)
    target = np.unwrap(target)
    target += 2.0 * np.pi * np.round(float(np.mean(theta - target)) / (2.0 * np.pi))
    return target


def _solve_phase_projection_qr(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    c_order: int,
    s_order: int,
) -> np.ndarray:
    matrix = _phase_design_matrix(theta, c_order=c_order, s_order=s_order)
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.abs(np.diag(r))
    scale = float(np.max(diagonal)) if diagonal.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * max(scale, 1.0)
    rank = int(np.count_nonzero(diagonal > tolerance))
    expected_rank = c_order + 1 + s_order
    if rank < expected_rank:
        raise ValueError(
            "R_boundary/Z_boundary do not provide full-rank phase QR fitting data: "
            f"rank {rank}, need {expected_rank}"
        )
    return np.linalg.solve(r, q.T @ np.asarray(delta, dtype=np.float64))


def _phase_design_matrix(theta: np.ndarray, *, c_order: int, s_order: int) -> np.ndarray:
    columns = [np.ones_like(theta, dtype=np.float64)]
    columns.extend(np.cos(order * theta) for order in range(1, c_order + 1))
    columns.extend(np.sin(order * theta) for order in range(1, s_order + 1))
    return np.column_stack(columns)


def _coefficients_to_offsets(
    coefficients: np.ndarray,
    *,
    c_order: int,
    s_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    c_offsets = np.asarray(coefficients[: c_order + 1], dtype=np.float64).copy()
    s_offsets = np.zeros(s_order + 1, dtype=np.float64)
    if s_order > 0:
        s_offsets[1:] = np.asarray(
            coefficients[c_order + 1 : c_order + 1 + s_order],
            dtype=np.float64,
        )
    return _normalize_fitted_offsets(c_offsets, s_offsets)


def _normalize_fitted_offsets(
    c_offsets: np.ndarray | list[float],
    s_offsets: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    c_out = np.asarray(c_offsets, dtype=np.float64).copy()
    s_out = np.asarray(s_offsets, dtype=np.float64).copy()
    c_out[0] = float((c_out[0] + np.pi) % (2.0 * np.pi) - np.pi)
    if s_out.size > 0:
        s_out[0] = 0.0
    return c_out, s_out


def _infer_theta(z_points: np.ndarray, z0: float, a_value: float, ka: float) -> np.ndarray:
    sin_theta = np.clip(-(z_points - z0) / (a_value * max(float(ka), 1.0e-6)), -1.0, 1.0)
    theta = np.empty_like(sin_theta)
    theta[0] = 0.5 * np.pi
    previous = theta[0]
    step = 2.0 * np.pi / max(len(z_points), 1)

    for index in range(1, len(z_points)):
        alpha = np.arcsin(sin_theta[index])
        candidates = []
        for candidate in (alpha, np.pi - alpha):
            while candidate < previous - 1.0e-12:
                candidate += 2.0 * np.pi
            candidates.extend((candidate, candidate + 2.0 * np.pi))
        target = previous + step
        theta[index] = min(candidates, key=lambda value: abs(value - target))
        previous = theta[index]

    return theta


def _max_bidirectional_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _build_boundary(
    *,
    R0: float,
    Z0: float,
    a: float,
    ka: float,
    c_offsets: np.ndarray,
    s_offsets: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    theta_bar = theta + c_offsets[0]
    for order in range(1, c_offsets.shape[0]):
        theta_bar += c_offsets[order] * np.cos(order * theta)
    for order in range(1, s_offsets.shape[0]):
        theta_bar += s_offsets[order] * np.sin(order * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))
