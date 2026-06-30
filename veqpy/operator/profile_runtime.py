"""
Module: operator.profile_runtime

Role:
- Consolidate shared Python rules for profile/problem setup and ProfileWorkspace refresh.
- Keep profile parameter parsing, Stage-A binding, and Fourier-family details out of operator.py.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veqpy.engine import validate_route
from veqpy.model import Problem
from veqpy.operator.packed_layout import build_profile_layout
from veqpy.workspace import GridWorkspace


def build_profile_parameter_arrays(
    *,
    problem: Problem,
    grid_workspace: GridWorkspace,
    profile_names: tuple[str, ...],
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]],
    profile_offset_specs: dict[str, float | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flat profile metadata arrays used by workspace and Stage A."""
    count = len(profile_names)
    profile_offsets = np.empty(count, dtype=np.float64)
    profile_scales = np.empty(count, dtype=np.float64)
    profile_powers = np.empty(count, dtype=np.int64)
    profile_envelope_powers = np.empty(count, dtype=np.int64)
    profile_amplitude_powers = np.empty(count, dtype=np.float64)
    refresh_profile_parameter_arrays(
        problem=problem,
        grid_workspace=grid_workspace,
        profile_names=profile_names,
        profile_offsets=profile_offsets,
        profile_scales=profile_scales,
        profile_powers=profile_powers,
        profile_envelope_powers=profile_envelope_powers,
        profile_amplitude_powers=profile_amplitude_powers,
        profile_static_kwargs_by_name=profile_static_kwargs_by_name,
        profile_offset_specs=profile_offset_specs,
    )
    return (
        profile_offsets,
        profile_scales,
        profile_powers,
        profile_envelope_powers,
        profile_amplitude_powers,
    )


def refresh_profile_parameter_arrays(
    *,
    problem: Problem,
    grid_workspace: GridWorkspace,
    profile_names: tuple[str, ...],
    profile_offsets: np.ndarray,
    profile_scales: np.ndarray,
    profile_powers: np.ndarray,
    profile_envelope_powers: np.ndarray,
    profile_amplitude_powers: np.ndarray,
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]],
    profile_offset_specs: dict[str, float | str],
) -> None:
    """Refresh flat profile metadata arrays from setup semantics."""
    del grid_workspace
    for p, name in enumerate(profile_names):
        static_kwargs = profile_static_kwargs_by_name.get(name, {})
        profile_powers[p] = int(static_kwargs.get("power", 0))
        profile_envelope_powers[p] = int(static_kwargs.get("envelope_power", 1))
        profile_amplitude_powers[p] = float(static_kwargs.get("amplitude_power", 1.0))
        profile_offsets[p] = _profile_offset(problem, name, profile_offset_specs)
        profile_scales[p] = _profile_scale(problem, name)


def refresh_profile_runtime(
    *,
    problem: Problem,
    operator_grid: GridWorkspace,
    profile_names: tuple[str, ...],
    profile_workspace,
    profile_offsets: np.ndarray,
    profile_scales: np.ndarray,
    profile_powers: np.ndarray,
    profile_envelope_powers: np.ndarray,
    profile_amplitude_powers: np.ndarray,
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]],
    profile_offset_specs: dict[str, float | str],
) -> None:
    """Refresh workspace profile slots from flat plan metadata."""
    refresh_profile_parameter_arrays(
        problem=problem,
        grid_workspace=operator_grid,
        profile_names=profile_names,
        profile_offsets=profile_offsets,
        profile_scales=profile_scales,
        profile_powers=profile_powers,
        profile_envelope_powers=profile_envelope_powers,
        profile_amplitude_powers=profile_amplitude_powers,
        profile_static_kwargs_by_name=profile_static_kwargs_by_name,
        profile_offset_specs=profile_offset_specs,
    )
    for p, _name in enumerate(profile_names):
        profile_workspace.refresh_profile_slot(
            profile_id=p,
            grid_workspace=operator_grid,
            offset=float(profile_offsets[p]),
            scale=float(profile_scales[p]),
            power=int(profile_powers[p]),
            envelope_power=int(profile_envelope_powers[p]),
            amplitude_power=float(profile_amplitude_powers[p]),
            coeff=None,
        )
    refresh_fourier_family_base_fields(
        M_max=operator_grid.M_max,
        profile_index=profile_workspace.profile_index,
        profile_workspace=profile_workspace,
        c_family_base_fields=profile_workspace.c_family_base_fields,
        s_family_base_fields=profile_workspace.s_family_base_fields,
    )


def refresh_stage_a_runtime(
    *,
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    profile_offsets: np.ndarray,
    profile_scales: np.ndarray,
    profile_amplitude_powers: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_amplitude_powers: np.ndarray,
    active_lengths: np.ndarray,
    active_coeff_index_rows: np.ndarray,
) -> None:
    """Refresh Stage-A active profile metadata arrays in place."""
    if active_profile_ids.size == 0:
        return

    for slot, p in enumerate(active_profile_ids):
        p_int = int(p)
        L = int(profile_L[p_int])
        coeff_indices = coeff_index[p_int, : L + 1]

        # These compact arrays are the Stage-A ABI.  The hot kernel never looks
        # up profile objects by name during residual evaluation.
        active_offsets[slot] = profile_offsets[p_int]
        active_scales[slot] = profile_scales[p_int]
        active_amplitude_powers[slot] = profile_amplitude_powers[p_int]
        active_lengths[slot] = coeff_indices.size
        if active_coeff_index_rows.shape[1] > 0:
            active_coeff_index_rows[slot].fill(-1)
            active_coeff_index_rows[slot, : coeff_indices.size] = coeff_indices


def refresh_fourier_family_base_fields(
    *,
    M_max: int,
    profile_index: dict[str, int],
    profile_workspace,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
) -> None:
    """Refresh cached c/s Fourier family base fields from active profiles."""
    c_family_base_fields.fill(0.0)
    s_family_base_fields.fill(0.0)
    for order in range(int(M_max) + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            np.copyto(c_family_base_fields[order], profile_workspace.fields_for(c_name))
        if order == 0:
            # s0 is not a physical sine mode; keep the zero-filled base row.
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            np.copyto(s_family_base_fields[order], profile_workspace.fields_for(s_name))


def refresh_fourier_family_metadata(
    *,
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
    profile_L: np.ndarray,
    profile_index: dict[str, int],
    c_offsets: np.ndarray | None,
    s_offsets: np.ndarray | None,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
) -> tuple[int, int]:
    """Infer effective c/s family orders and zero inactive family tails."""
    c_effective_order = 0
    for name in c_profile_names:
        order = int(name[1:])
        if int(profile_L[profile_index[name]]) >= 0:
            c_effective_order = max(c_effective_order, order)
            continue
        if (
            c_offsets is not None
            and order < c_offsets.shape[0]
            and abs(float(c_offsets[order])) > 1e-14
        ):
            c_effective_order = max(c_effective_order, order)

    s_effective_order = 0
    for name in s_profile_names:
        order = int(name[1:])
        if int(profile_L[profile_index[name]]) >= 0:
            s_effective_order = max(s_effective_order, order)
            continue
        if (
            s_offsets is not None
            and order < s_offsets.shape[0]
            and abs(float(s_offsets[order])) > 1e-14
        ):
            s_effective_order = max(s_effective_order, order)

    if c_effective_order + 1 < c_family_fields.shape[0]:
        # Zero inactive tails immediately so later fused geometry calls cannot
        # see leftover higher-order data from an earlier problem.
        c_family_fields[c_effective_order + 1 :].fill(0.0)
    if s_effective_order + 1 < s_family_fields.shape[0]:
        # Same tail cleanup for sine modes; s0 may already be a structural zero.
        s_family_fields[s_effective_order + 1 :].fill(0.0)
    return c_effective_order, s_effective_order


def validate_problem_compatibility(
    problem: Problem,
    *,
    profile_names: tuple[str, ...],
    prefix_profile_names: tuple[str, ...],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    order_offsets: np.ndarray,
    validate_source_inputs: Callable[[Problem], None],
) -> None:
    """Validate that a replacement problem preserves the bound operator layout."""
    validate_route(problem.route, problem.coordinate, problem.nodes)
    next_profile_L, next_coeff_index, next_order_offsets = build_profile_layout(
        problem.active_profiles,
        profile_names=profile_names,
        prefix_profile_names=prefix_profile_names,
    )
    if not np.array_equal(next_profile_L, profile_L):
        raise ValueError("Replacement problem changes the active profile layout")
    if not np.array_equal(next_coeff_index, coeff_index):
        raise ValueError("Replacement problem changes the packed coefficient layout")
    if not np.array_equal(next_order_offsets, order_offsets):
        raise ValueError("Replacement problem changes the degree ordering layout")
    _validate_active_prefix_profile_ownership(
        problem=problem,
        profile_names=profile_names,
        profile_L=profile_L,
    )
    validate_source_inputs(problem)


def _validate_active_prefix_profile_ownership(
    *,
    problem: Problem,
    profile_names: tuple[str, ...],
    profile_L: np.ndarray,
) -> None:
    active_names = {
        profile_names[index] for index, length in enumerate(profile_L) if int(length) >= 0
    }
    requires_active_F = problem.route == "PJ2"
    requires_active_psin = (
        problem.route != "PJ2" and problem.coordinate == "psin" and problem.nodes == "uniform"
    )

    if "F" in active_names and not requires_active_F:
        raise ValueError(
            f"{problem.route} does not accept an active F profile; "
            "active F is only supported for PJ2"
        )
    if requires_active_F and "F" not in active_names:
        raise ValueError(f"{problem.route} requires an active F profile")
    if "F" in active_names and "psin" in active_names:
        raise ValueError("Active F and active psin profiles are mutually exclusive")
    if "psin" in active_names and not requires_active_psin:
        raise ValueError(
            f"{problem.route} {problem.coordinate}/{problem.nodes} does not accept "
            "an active psin profile"
        )
    if requires_active_psin and "psin" not in active_names:
        raise ValueError(f"{problem.route} requires an active psin profile")


def _profile_offset(
    problem: Problem,
    name: str,
    profile_offset_specs: dict[str, float | str],
) -> float:
    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        return 0.0 if order >= problem.c_offsets.shape[0] else float(problem.c_offsets[order])
    if name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        return 0.0 if order >= problem.s_offsets.shape[0] else float(problem.s_offsets[order])
    try:
        offset_spec = profile_offset_specs[name]
    except KeyError as exc:
        raise KeyError(f"Unknown profile name {name!r}") from exc
    if isinstance(offset_spec, str):
        return float(getattr(problem, offset_spec))
    return float(offset_spec)


def _profile_scale(problem: Problem, name: str) -> float:
    if name == "F":
        # F coefficients represent the normalized F**2 amplitude; the profile
        # evaluator applies amplitude_power=0.5 and this scale restores F units.
        return float(problem.R0 * problem.B0)
    return 1.0
