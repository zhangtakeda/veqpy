"""Private JAX profile-stage entrypoints for PF/rho/grid lowering."""

from __future__ import annotations

from typing import Any

from veqpy.engine.jax.state import JaxStaticSpec
from veqpy.workspace.field_rows import GRID_RADIAL_RHO_POWERS_START

_AMPLITUDE_POWER_FLOOR = 1.0e-10


def evaluate_profile_stage_pf_rho_grid(
    jax_module: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    x: Any,
) -> Any:
    """Return profile fields refreshed from packed ``x`` for PF/rho/grid."""

    jnp = jax_module.numpy
    profile_fields = leaves["profile_fields_template"]
    profile_rp_fields = leaves["profile_rp_fields"]
    profile_env_fields = leaves["profile_env_fields"]
    offsets = leaves["active_offsets"]
    scales = leaves["active_scales"]
    coeff_index_rows = leaves["active_coeff_index_rows"]
    grid_radial_fields = leaves["grid_radial_fields"]
    active_profile_ids = tuple(int(v) for v in spec.active_profile_ids)
    active_lengths = tuple(int(v) for v in spec.active_lengths)
    t, t_r, t_rr = _profile_basis_views(
        grid_radial_fields,
        grid_k_max=int(spec.k_max),
        grid_l_max=int(spec.l_max),
    )

    for slot, profile_id in enumerate(active_profile_ids):
        length = active_lengths[slot]
        indices = coeff_index_rows[slot, :length]
        coeff = x[indices]
        series = coeff @ t[:length]
        series_r = coeff @ t_r[:length]
        series_rr = coeff @ t_rr[:length]
        env = profile_env_fields[profile_id, 0]
        env_r = profile_env_fields[profile_id, 1]
        env_rr = profile_env_fields[profile_id, 2]
        base = env * series
        base_r = env_r * series + env * series_r
        base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
        offset = offsets[slot]
        amplitude_power = float(spec.active_amplitude_powers[slot])
        amp, amp_r, amp_rr = _apply_amplitude_power(
            jnp,
            offset + base,
            base_r,
            base_rr,
            amplitude_power,
        )
        rp = profile_rp_fields[profile_id, 0]
        rp_r = profile_rp_fields[profile_id, 1]
        rp_rr = profile_rp_fields[profile_id, 2]
        scale = scales[slot]
        value = scale * (rp * amp)
        value_r = scale * (rp_r * amp + rp * amp_r)
        value_rr = scale * (rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr)
        profile_fields = profile_fields.at[profile_id, 0].set(value)
        profile_fields = profile_fields.at[profile_id, 1].set(value_r)
        profile_fields = profile_fields.at[profile_id, 2].set(value_rr)
    return profile_fields


def _profile_basis_views(
    grid_radial_fields: Any,
    *,
    grid_k_max: int,
    grid_l_max: int,
) -> tuple[Any, Any, Any]:
    t_start = GRID_RADIAL_RHO_POWERS_START + grid_k_max + 2
    t_stop = t_start + grid_l_max + 1
    t_r_stop = t_stop + grid_l_max + 1
    return (
        grid_radial_fields[t_start:t_stop],
        grid_radial_fields[t_stop:t_r_stop],
        grid_radial_fields[t_r_stop : t_r_stop + grid_l_max + 1],
    )


def _apply_amplitude_power(
    jnp: Any,
    amp: Any,
    amp_r: Any,
    amp_rr: Any,
    amplitude_power: float,
) -> tuple[Any, Any, Any]:
    if amplitude_power == 1.0:
        return amp, amp_r, amp_rr

    amp_eval = jnp.maximum(amp, _AMPLITUDE_POWER_FLOOR)
    if amplitude_power == 0.5:
        value = jnp.sqrt(amp_eval)
        inv_value = 1.0 / value
        inv_value3 = inv_value / amp_eval
        return (
            value,
            0.5 * amp_r * inv_value,
            0.5 * amp_rr * inv_value - 0.25 * amp_r * amp_r * inv_value3,
        )

    value = amp_eval**amplitude_power
    value_r = amplitude_power * amp_eval ** (amplitude_power - 1.0) * amp_r
    value_rr = (
        amplitude_power * amp_eval ** (amplitude_power - 1.0) * amp_rr
        + amplitude_power
        * (amplitude_power - 1.0)
        * amp_eval ** (amplitude_power - 2.0)
        * amp_r
        * amp_r
    )
    return value, value_r, value_rr
