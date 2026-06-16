"""
Module: operator.profile_runtime

Role:
- Consolidate shared Python rules for profile/case setup and ProfileWorkspace refresh.
- Keep profile parameter parsing, Stage-A binding, and Fourier-family details out of operator.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine.numba_source import validate_route
from veqpy.model.problem import Problem
from veqpy.model.profile import Profile
from veqpy.operator.packed_layout import build_profile_layout, coeff_array_from_list
from veqpy.workspace import GridWorkspace

if TYPE_CHECKING:
    from veqpy.workspace.profile_workspace import ProfileWorkspace


def make_profile(
    *,
    case: Problem,
    operator_grid: GridWorkspace | None = None,
    name: str,
    profile_L: np.ndarray,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]],
    profile_offset_specs: dict[str, float | str],
) -> Profile:
    """Construct one profile object from case inputs and packed layout metadata."""
    kwargs: dict[str, float | int | np.ndarray | None] = {}
    static_kwargs = profile_static_kwargs_by_name.get(name)
    if static_kwargs is None and name.startswith(("c", "s")) and name[1:].isdigit():
        order = int(name[1:])
        # Fourier modes get their regularity power from the grid order table.
        # c0 is the radial shift baseline and intentionally has no axis power.
        static_kwargs = {} if order == 0 else {"power": int(operator_grid.K_values[order])}
    if static_kwargs is not None:
        kwargs.update(static_kwargs)

    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        kwargs["offset"] = 0.0 if order >= case.c_offsets.shape[0] else float(case.c_offsets[order])
    elif name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        kwargs["offset"] = 0.0 if order >= case.s_offsets.shape[0] else float(case.s_offsets[order])
    else:
        try:
            offset_spec = profile_offset_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile name {name!r}") from exc
        kwargs["offset"] = (
            float(getattr(case, offset_spec))
            if isinstance(offset_spec, str)
            else float(offset_spec)
        )

    kwargs["scale"] = _profile_scale(case, name)

    p = profile_index[name]
    L = int(profile_L[p])
    template_profile = case.profiles.get(name)
    coeff = None if template_profile is None else template_profile.coeff
    # L < 0 marks a passive profile: construct the Profile object but leave its
    # coefficient vector absent so Stage A will not expect packed coefficients.
    kwargs["coeff"] = (
        None if L < 0 or coeff is None else coeff_array_from_list(name, coeff)[: L + 1].copy()
    )
    return Profile(**kwargs)


def refresh_profile_runtime(
    *,
    case: Problem,
    operator_grid: GridWorkspace,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    profiles_by_name: dict[str, Profile],
    profile_workspace: ProfileWorkspace,
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]],
    profile_offset_specs: dict[str, float | str],
) -> None:
    """Refresh profile objects and workspace slots from a replacement case."""
    for name in profile_names:
        profile = profiles_by_name[name]
        static_kwargs = profile_static_kwargs_by_name.get(name)
        if static_kwargs is None and name.startswith(("c", "s")) and name[1:].isdigit():
            order = int(name[1:])
            # Repeat make_profile's static-field rule during case replacement so
            # reused Profile instances stay synchronized with grid topology.
            static_kwargs = {} if order == 0 else {"power": int(operator_grid.K_values[order])}
        elif static_kwargs is None:
            static_kwargs = {}
        profile.power = int(static_kwargs.get("power", 0))
        profile.envelope_power = int(static_kwargs.get("envelope_power", 1))
        profile.amplitude_power = float(static_kwargs.get("amplitude_power", 1.0))
        if name.startswith("c") and name[1:].isdigit():
            order = int(name[1:])
            profile.offset = (
                0.0 if order >= case.c_offsets.shape[0] else float(case.c_offsets[order])
            )
        elif name.startswith("s") and name[1:].isdigit():
            order = int(name[1:])
            profile.offset = (
                0.0 if order >= case.s_offsets.shape[0] else float(case.s_offsets[order])
            )
        else:
            offset_spec = profile_offset_specs[name]
            profile.offset = (
                float(getattr(case, offset_spec))
                if isinstance(offset_spec, str)
                else float(offset_spec)
            )
        profile.scale = _profile_scale(case, name)
        p = profile_index[name]
        L = int(profile_L[p])
        template_profile = case.profiles.get(name)
        coeff = None if template_profile is None else template_profile.coeff
        profile.coeff = (
            None if L < 0 or coeff is None else coeff_array_from_list(name, coeff)[: L + 1].copy()
        )
        profile_workspace.refresh_profile_slot(
            profile_id=p,
            profile=profile,
            grid_workspace=operator_grid,
        )
    refresh_fourier_family_base_fields(
        M_max=operator_grid.M_max,
        profile_index=profile_index,
        profile_workspace=profile_workspace,
        c_family_base_fields=profile_workspace.c_family_base_fields,
        s_family_base_fields=profile_workspace.s_family_base_fields,
    )


def _profile_scale(case: Problem, name: str) -> float:
    if name == "F":
        # F coefficients represent the normalized F**2 amplitude; the profile
        # evaluator applies amplitude_power=0.5 and this scale restores F units.
        return float(case.R0 * case.B0)
    return 1.0


def refresh_stage_a_runtime(
    *,
    active_profile_ids: np.ndarray,
    profile_names: tuple[str, ...],
    profiles_by_name: dict[str, Profile],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
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
        profile_name = profile_names[p_int]
        profile = profiles_by_name[profile_name]
        L = int(profile_L[p_int])
        coeff_indices = coeff_index[p_int, : L + 1]

        # These compact arrays are the Stage-A ABI.  The hot kernel never looks
        # up Profile objects by name during residual evaluation.
        active_offsets[slot] = profile.offset
        active_scales[slot] = profile.scale
        active_amplitude_powers[slot] = profile.amplitude_power
        active_lengths[slot] = coeff_indices.size
        if active_coeff_index_rows.shape[1] > 0:
            active_coeff_index_rows[slot].fill(-1)
            active_coeff_index_rows[slot, : coeff_indices.size] = coeff_indices


def refresh_fourier_family_base_fields(
    *,
    M_max: int,
    profile_index: dict[str, int],
    profile_workspace: ProfileWorkspace,
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
    profile_coeffs: dict[str, list[float] | np.ndarray | int | None],
    c_offsets: np.ndarray | None,
    s_offsets: np.ndarray | None,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
) -> tuple[int, int]:
    """Infer effective c/s family orders and zero inactive family tails."""
    c_effective_order = 0
    for name in c_profile_names:
        order = int(name[1:])
        if profile_coeffs.get(name) is not None:
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
        if profile_coeffs.get(name) is not None:
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
        # see leftover higher-order data from an earlier case.
        c_family_fields[c_effective_order + 1 :].fill(0.0)
    if s_effective_order + 1 < s_family_fields.shape[0]:
        # Same tail cleanup for sine modes; s0 may already be a structural zero.
        s_family_fields[s_effective_order + 1 :].fill(0.0)
    return c_effective_order, s_effective_order


def validate_case_compatibility(
    case: Problem,
    *,
    profile_names: tuple[str, ...],
    prefix_profile_names: tuple[str, ...],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    order_offsets: np.ndarray,
    validate_source_inputs: Callable[[Problem], None],
) -> None:
    """Validate that a replacement case preserves the bound operator layout."""
    validate_route(case.route, case.coordinate, case.nodes)
    next_profile_L, next_coeff_index, next_order_offsets = build_profile_layout(
        case.profile_coeffs,
        profile_names=profile_names,
        prefix_profile_names=prefix_profile_names,
    )
    if not np.array_equal(next_profile_L, profile_L):
        raise ValueError("Replacement case changes the active profile layout")
    if not np.array_equal(next_coeff_index, coeff_index):
        raise ValueError("Replacement case changes the packed coefficient layout")
    if not np.array_equal(next_order_offsets, order_offsets):
        raise ValueError("Replacement case changes the degree ordering layout")
    _validate_active_prefix_profile_ownership(
        case=case,
        profile_names=profile_names,
        profile_L=profile_L,
    )
    validate_source_inputs(case)


def _validate_active_prefix_profile_ownership(
    *,
    case: Problem,
    profile_names: tuple[str, ...],
    profile_L: np.ndarray,
) -> None:
    active_names = {
        profile_names[index] for index, length in enumerate(profile_L) if int(length) >= 0
    }
    requires_active_F = case.route == "PJ2"
    requires_active_psin = (
        case.route != "PJ2" and case.coordinate == "psin" and case.nodes == "uniform"
    )

    if "F" in active_names and not requires_active_F:
        raise ValueError(
            f"{case.route} does not accept an active F profile; active F is only supported for PJ2"
        )
    if requires_active_F and "F" not in active_names:
        raise ValueError(f"{case.route} requires an active F profile")
    if "F" in active_names and "psin" in active_names:
        raise ValueError("Active F and active psin profiles are mutually exclusive")
    if "psin" in active_names and not requires_active_psin:
        raise ValueError(
            f"{case.route} {case.coordinate}/{case.nodes} does not accept an active psin profile"
        )
    if requires_active_psin and "psin" not in active_names:
        raise ValueError(f"{case.route} requires an active psin profile")
