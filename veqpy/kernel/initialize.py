"""
Module: kernel.initialize

Role:
- Build kernel-owned packed initial states.
- Keep geometric initializer formulas out of the direct runtime facade.

Public API:
- build_boundary_slope_initial_state
- estimate_axis_shift_h0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from veqpy.kernel.runtime import KernelRuntimePlan
    from veqpy.workspace import ProfileWorkspace

_SOURCE_CURRENT_ROUGHNESS_WEIGHT = 0.5
_SOURCE_STRUCTURE_REL_TOL = 1.0e-6
_AXIS_SHIFT_COEFF_TOL = 1.0e-6
_PSIN_INITIAL_COEFF_COUNT = 1
_PSIN_INITIAL_COEFF_DAMPING = 0.5
_PSIN_INITIAL_COEFF_ABS_LIMIT = 1.0 - 1.0e-6
_PSIN_PROJECTION_RCOND = 1.0e-12
_PSIN_MONOTONIC_TOL = 1.0e-10
_PSIN_RADIAL_DERIVATIVE_TOL = 1.0e-10
_PSIN_VALUE_MARGIN = 5.0e-2
_TINY = 1.0e-16


def build_boundary_slope_initial_state(
    *,
    case: object,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    source_psin_target: Callable[[np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray:
    """Build the geometric packed initial state for a kernel layout."""

    x = np.zeros(int(plan.x_size), dtype=np.float64)
    _seed_axis_and_boundary_shape_terms(
        x,
        h0_est=estimate_axis_shift_h0(case),
        plan=plan,
        profile_workspace=profile_workspace,
    )
    _seed_active_psin_coefficients(
        x,
        plan=plan,
        profile_workspace=profile_workspace,
        source_psin_target=source_psin_target,
    )
    return x


def _seed_axis_and_boundary_shape_terms(
    x: np.ndarray,
    *,
    h0_est: float,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
) -> None:
    for slot, profile_id in enumerate(profile_workspace.active_profile_ids):
        length = int(profile_workspace.active_lengths[slot])
        if length <= 0:
            continue

        profile_id_int = int(profile_id)
        name = profile_workspace.profile_names[profile_id_int]
        idx0 = int(profile_workspace.active_coeff_index_rows[slot, 0])

        if name == "h":
            x[idx0] = h0_est
        elif name.startswith(("c", "s")):
            offset = float(plan.profile_offsets[profile_id_int])
            power = int(plan.profile_powers[profile_id_int])
            x[idx0] = -offset / float(2 * power + 1)


def estimate_axis_shift_h0(case: object) -> float:
    """Estimate the geometric axis radial-shift coefficient from source moments.

    The estimate keeps the large-aspect-ratio Shafranov scaling ``a / R0`` but
    replaces the old hard source-class switch with a continuous roughness drive.
    Small source structure and small axis shifts are snapped to zero so
    numerically uniform sources do not perturb finite-difference solvers.
    """

    boundary = case.boundary
    epsilon = float(boundary.a) / float(boundary.R0)
    kappa = abs(float(boundary.ka))
    elongation_factor = 2.0 * kappa / (1.0 + kappa * kappa)

    pressure_drive = _relative_abs_rms(case.heat_input)
    current_drive = _relative_abs_rms(case.current_input)
    source_drive = np.hypot(
        pressure_drive,
        _SOURCE_CURRENT_ROUGHNESS_WEIGHT * current_drive,
    )

    h0 = float(epsilon * elongation_factor * np.tanh(source_drive))
    return 0.0 if abs(h0) <= _AXIS_SHIFT_COEFF_TOL else h0


def _relative_abs_rms(values: np.ndarray) -> float:
    source = np.abs(np.asarray(values, dtype=np.float64))
    mean = float(np.mean(source))
    centered = source - mean
    rms = float(np.sqrt(np.mean(centered * centered)))
    relative = rms / (mean + _TINY)
    return 0.0 if relative <= _SOURCE_STRUCTURE_REL_TOL else relative


def _seed_active_psin_coefficients(
    x: np.ndarray,
    *,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    source_psin_target: Callable[[np.ndarray], np.ndarray | None] | None,
) -> None:
    if source_psin_target is None:
        return
    if not bool(getattr(plan.source_execution, "requires_optimized_psin_profile", False)):
        return

    psin_profile_id = profile_workspace.profile_index.get("psin")
    if psin_profile_id is None:
        return
    psin_slot = profile_workspace.active_slot_for_profile_id(int(psin_profile_id))
    if psin_slot < 0:
        return
    coeff_count = int(profile_workspace.active_lengths[psin_slot])
    if coeff_count <= 0:
        return

    fit_count = min(coeff_count, _PSIN_INITIAL_COEFF_COUNT)
    raw_target = source_psin_target(x)
    if fit_count == 1:
        coeff = _project_psin0_source_target_coefficient(
            raw_target,
            expected_shape=plan.grid_workspace.rho.shape,
            plan=plan,
            profile_workspace=profile_workspace,
            psin_profile_id=int(psin_profile_id),
        )
    else:
        target = _normalized_source_psin_target(
            raw_target,
            expected_shape=plan.grid_workspace.rho.shape,
        )
        if target is None:
            return
        coeff = _project_psin_target_coefficients(
            target,
            plan=plan,
            profile_workspace=profile_workspace,
            psin_profile_id=int(psin_profile_id),
            coeff_count=fit_count,
        )
    if coeff is None:
        return

    coeff_indices = profile_workspace.active_coeff_index_rows[psin_slot, :fit_count]
    x[coeff_indices] = coeff


def _normalized_source_psin_target(
    target: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if target is None:
        return None
    psin = np.asarray(target, dtype=np.float64)
    if psin.shape != expected_shape or not np.all(np.isfinite(psin)):
        return None

    offset = float(psin[0])
    scale = float(psin[-1] - offset)
    if abs(scale) <= _TINY:
        return None

    normalized = (psin - offset) / scale
    normalized[0] = 0.0
    normalized[-1] = 1.0
    if not np.all(np.isfinite(normalized)):
        return None
    return normalized


def _project_psin_target_coefficients(
    target: np.ndarray,
    *,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    psin_profile_id: int,
    coeff_count: int,
) -> np.ndarray | None:
    if coeff_count == 1:
        return _project_psin0_target_coefficient(
            target,
            plan=plan,
            profile_workspace=profile_workspace,
            psin_profile_id=psin_profile_id,
        )

    base, basis, base_r, basis_r = _psin_projection_basis(
        plan=plan,
        profile_workspace=profile_workspace,
        psin_profile_id=psin_profile_id,
        coeff_count=coeff_count,
    )
    if basis.shape[1] <= 0:
        return None

    weights = np.asarray(plan.grid_workspace.weights, dtype=np.float64)
    row_weights = np.sqrt(np.maximum(weights, 0.0))
    lhs = basis * row_weights[:, None]
    rhs = (target - base) * row_weights
    try:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=_PSIN_PROJECTION_RCOND)[0]
    except np.linalg.LinAlgError:
        return None
    if coeff.shape != (coeff_count,) or not np.all(np.isfinite(coeff)):
        return None

    base_error = _weighted_rms(target - base, weights)
    best_coeff: np.ndarray | None = None
    best_error = base_error
    for damping in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        trial_coeff = _clip_psin_initial_coefficients(
            damping * _PSIN_INITIAL_COEFF_DAMPING * coeff,
            base_r=base_r,
            basis_r=basis_r,
        )
        if trial_coeff is None:
            continue
        trial = base + basis @ trial_coeff
        trial_r = base_r + basis_r @ trial_coeff
        if not _valid_psin_profile(trial, trial_r):
            continue
        trial_error = _weighted_rms(target - trial, weights)
        if np.isfinite(trial_error) and trial_error < best_error:
            best_error = trial_error
            best_coeff = trial_coeff

    return best_coeff


def _project_psin0_target_coefficient(
    target: np.ndarray,
    *,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    psin_profile_id: int,
) -> np.ndarray | None:
    weights = np.asarray(plan.grid_workspace.weights, dtype=np.float64)
    value_scale = float(plan.profile_scales[psin_profile_id])
    offset = float(plan.profile_offsets[psin_profile_id])
    radial_power = profile_workspace.profile_rp_fields[psin_profile_id, 0]
    radial_power_r = profile_workspace.profile_rp_fields[psin_profile_id, 1]
    envelope = profile_workspace.profile_env_fields[psin_profile_id, 0]
    envelope_r = profile_workspace.profile_env_fields[psin_profile_id, 1]

    base = value_scale * radial_power * offset
    basis = value_scale * radial_power * envelope
    denominator = float(np.dot(weights, basis * basis))
    if denominator <= _TINY:
        return None
    numerator = float(np.dot(weights, basis * (target - base)))
    raw_coeff = numerator / denominator
    if not np.isfinite(raw_coeff):
        return None

    base_r = value_scale * radial_power_r * offset
    basis_r = value_scale * (radial_power_r * envelope + radial_power * envelope_r)
    coeff = _clip_scalar_psin_initial_coefficient(
        _PSIN_INITIAL_COEFF_DAMPING * raw_coeff,
        base_r=base_r,
        basis_r=basis_r,
    )
    if coeff is None:
        return None
    if 2.0 * coeff * numerator - coeff * coeff * denominator <= 0.0:
        return None

    values = base + coeff * basis
    radial_derivative = base_r + coeff * basis_r
    if not _valid_psin_profile(values, radial_derivative):
        return None
    return np.array([coeff], dtype=np.float64)


def _project_psin0_source_target_coefficient(
    target: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...],
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    psin_profile_id: int,
) -> np.ndarray | None:
    if target is None:
        return None
    psin = np.asarray(target, dtype=np.float64)
    if psin.shape != expected_shape:
        return None

    coeff = _project_psin0_source_target_coefficient_impl(
        psin,
        np.asarray(plan.grid_workspace.weights, dtype=np.float64),
        profile_workspace.profile_rp_fields[psin_profile_id, 0],
        profile_workspace.profile_rp_fields[psin_profile_id, 1],
        profile_workspace.profile_env_fields[psin_profile_id, 0],
        profile_workspace.profile_env_fields[psin_profile_id, 1],
        float(plan.profile_scales[psin_profile_id]),
        float(plan.profile_offsets[psin_profile_id]),
    )
    if not np.isfinite(coeff):
        return None
    return np.array([float(coeff)], dtype=np.float64)


@njit(cache=True, nogil=True)
def _project_psin0_source_target_coefficient_impl(
    target: np.ndarray,
    weights: np.ndarray,
    radial_power: np.ndarray,
    radial_power_r: np.ndarray,
    envelope: np.ndarray,
    envelope_r: np.ndarray,
    value_scale: float,
    offset: float,
) -> float:
    size = target.shape[0]
    if size < 2:
        return np.nan
    target_offset = target[0]
    target_scale = target[size - 1] - target_offset
    if not np.isfinite(target_offset) or not np.isfinite(target_scale):
        return np.nan
    if abs(target_scale) <= _TINY:
        return np.nan

    denominator = 0.0
    numerator = 0.0
    lower = -_PSIN_INITIAL_COEFF_ABS_LIMIT
    upper = _PSIN_INITIAL_COEFF_ABS_LIMIT

    for i in range(size):
        raw_value = target[i]
        if not np.isfinite(raw_value):
            return np.nan
        normalized = (raw_value - target_offset) / target_scale
        if i == 0:
            normalized = 0.0
        elif i == size - 1:
            normalized = 1.0

        base = value_scale * radial_power[i] * offset
        basis = value_scale * radial_power[i] * envelope[i]
        weight = weights[i]
        denominator += weight * basis * basis
        numerator += weight * basis * (normalized - base)

        base_r = value_scale * radial_power_r[i] * offset
        basis_r = value_scale * (radial_power_r[i] * envelope[i] + radial_power[i] * envelope_r[i])
        rhs = _PSIN_RADIAL_DERIVATIVE_TOL - base_r
        if basis_r > _TINY:
            candidate = rhs / basis_r
            if candidate > lower:
                lower = candidate
        elif basis_r < -_TINY:
            candidate = rhs / basis_r
            if candidate < upper:
                upper = candidate
        elif base_r <= _PSIN_RADIAL_DERIVATIVE_TOL:
            return np.nan

    if denominator <= _TINY:
        return np.nan
    if lower > upper:
        return np.nan

    coeff = _PSIN_INITIAL_COEFF_DAMPING * numerator / denominator
    if coeff < lower:
        coeff = lower
    elif coeff > upper:
        coeff = upper
    if not np.isfinite(coeff):
        return np.nan
    if 2.0 * coeff * numerator - coeff * coeff * denominator <= 0.0:
        return np.nan

    previous = 0.0
    for i in range(size):
        base = value_scale * radial_power[i] * offset
        basis = value_scale * radial_power[i] * envelope[i]
        value = base + coeff * basis
        base_r = value_scale * radial_power_r[i] * offset
        basis_r = value_scale * (radial_power_r[i] * envelope[i] + radial_power[i] * envelope_r[i])
        radial_derivative = base_r + coeff * basis_r
        if not np.isfinite(value) or not np.isfinite(radial_derivative):
            return np.nan
        if radial_derivative <= _PSIN_RADIAL_DERIVATIVE_TOL:
            return np.nan
        if value < -_PSIN_VALUE_MARGIN or value > 1.0 + _PSIN_VALUE_MARGIN:
            return np.nan
        if i > 0 and value - previous < -_PSIN_MONOTONIC_TOL:
            return np.nan
        previous = value

    return coeff


def _psin_projection_basis(
    *,
    plan: KernelRuntimePlan,
    profile_workspace: ProfileWorkspace,
    psin_profile_id: int,
    coeff_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value_scale = float(plan.profile_scales[psin_profile_id])
    offset = float(plan.profile_offsets[psin_profile_id])
    radial_power = profile_workspace.profile_rp_fields[psin_profile_id, 0]
    radial_power_r = profile_workspace.profile_rp_fields[psin_profile_id, 1]
    envelope = profile_workspace.profile_env_fields[psin_profile_id, 0]
    envelope_r = profile_workspace.profile_env_fields[psin_profile_id, 1]
    chebyshev = plan.grid_workspace.T[:coeff_count]
    chebyshev_r = plan.grid_workspace.T_r[:coeff_count]

    base = value_scale * radial_power * offset
    basis = value_scale * (radial_power * envelope)[:, None] * chebyshev.T
    base_r = value_scale * radial_power_r * offset
    basis_r = value_scale * (
        (radial_power_r * envelope)[:, None] * chebyshev.T
        + radial_power[:, None] * (envelope_r[:, None] * chebyshev.T)
        + radial_power[:, None] * (envelope[:, None] * chebyshev_r.T)
    )
    return base, basis, base_r, basis_r


def _clip_psin_initial_coefficients(
    coeff: np.ndarray,
    *,
    base_r: np.ndarray,
    basis_r: np.ndarray,
) -> np.ndarray | None:
    if coeff.shape != (1,) or basis_r.shape[1] != 1:
        return coeff

    clipped = _clip_scalar_psin_initial_coefficient(
        float(coeff[0]),
        base_r=base_r,
        basis_r=basis_r[:, 0],
    )
    if clipped is None:
        return None
    return np.array([clipped], dtype=np.float64)


def _clip_scalar_psin_initial_coefficient(
    coeff: float,
    *,
    base_r: np.ndarray,
    basis_r: np.ndarray,
) -> float | None:
    lower = -_PSIN_INITIAL_COEFF_ABS_LIMIT
    upper = _PSIN_INITIAL_COEFF_ABS_LIMIT
    slope = basis_r
    rhs = _PSIN_RADIAL_DERIVATIVE_TOL - base_r
    positive = slope > _TINY
    negative = slope < -_TINY

    if np.any(positive):
        lower = max(lower, float(np.max(rhs[positive] / slope[positive])))
    if np.any(negative):
        upper = min(upper, float(np.min(rhs[negative] / slope[negative])))
    flat = ~(positive | negative)
    if np.any(base_r[flat] <= _PSIN_RADIAL_DERIVATIVE_TOL):
        return None
    if lower > upper:
        return None

    return min(max(float(coeff), lower), upper)


def _valid_psin_profile(values: np.ndarray, radial_derivative: np.ndarray) -> bool:
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return False
    if (
        radial_derivative.shape != values.shape
        or not np.all(np.isfinite(radial_derivative))
        or float(np.min(radial_derivative)) <= _PSIN_RADIAL_DERIVATIVE_TOL
    ):
        return False
    if float(np.min(values)) < -_PSIN_VALUE_MARGIN:
        return False
    if float(np.max(values)) > 1.0 + _PSIN_VALUE_MARGIN:
        return False
    return bool(np.all(np.diff(values) >= -_PSIN_MONOTONIC_TOL))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    weighted_norm = float(np.dot(weights, values * values))
    weight_sum = float(np.sum(weights))
    return float(np.sqrt(weighted_norm / max(weight_sum, _TINY)))
