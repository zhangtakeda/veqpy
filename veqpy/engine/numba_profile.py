"""
Module: engine.numba_profile

Role:
- Compute profile fields.
- Inputs use basis tables and profile coefficients.

Public API:
- update_profile
- update_profiles_packed_bulk

Notes:
- update_profile is used for one explicit-coefficient profile.
- update_profiles_packed_bulk is used for packed runtime updates in Stage A.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_AMPLITUDE_POWER_FLOOR = 1.0e-10


@njit(cache=True, fastmath=True, nogil=True)
def _apply_amplitude_power(
    amp: float,
    amp_r: float,
    amp_rr: float,
    amplitude_power: float,
) -> tuple[float, float, float]:
    """Apply ``amp**amplitude_power`` and chain-rule derivative rows."""

    if amplitude_power == 1.0:
        return amp, amp_r, amp_rr

    if amp < _AMPLITUDE_POWER_FLOOR:
        amp = _AMPLITUDE_POWER_FLOOR

    if amplitude_power == 0.5:
        value = np.sqrt(amp)
        inv_value = 1.0 / value
        inv_value3 = inv_value / amp
        return (
            value,
            0.5 * amp_r * inv_value,
            0.5 * amp_rr * inv_value - 0.25 * amp_r * amp_r * inv_value3,
        )

    value = amp**amplitude_power
    value_r = amplitude_power * amp ** (amplitude_power - 1.0) * amp_r
    value_rr = (
        amplitude_power * amp ** (amplitude_power - 1.0) * amp_rr
        + amplitude_power
        * (amplitude_power - 1.0)
        * amp ** (amplitude_power - 2.0)
        * amp_r
        * amp_r
    )
    return value, value_r, value_rr


@njit(cache=True, fastmath=True, nogil=True)
def update_profile(
    out_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
    amplitude_power: float,
) -> None:
    """Update one profile field set in place."""
    nr = out_fields.shape[1]

    if coeff is None:
        # Passive profiles are pure offset * rho_power envelopes.  They still
        # need derivative rows so geometry can consume them like active profiles.
        if amplitude_power == 1.0:
            for i in range(nr):
                out_fields[0, i] = offset * rp_fields[0, i]
                out_fields[1, i] = offset * rp_fields[1, i]
                out_fields[2, i] = offset * rp_fields[2, i]
            return
        amp, amp_r, amp_rr = _apply_amplitude_power(offset, 0.0, 0.0, amplitude_power)
        for i in range(nr):
            out_fields[0, i] = amp * rp_fields[0, i]
            out_fields[1, i] = amp * rp_fields[1, i] + rp_fields[0, i] * amp_r
            out_fields[2, i] = (
                amp * rp_fields[2, i]
                + 2.0 * rp_fields[1, i] * amp_r
                + rp_fields[0, i] * amp_rr
            )
        return

    coeff_size = coeff.size
    if amplitude_power == 1.0:
        for i in range(nr):
            series = 0.0
            series_r = 0.0
            series_rr = 0.0
            # Build Chebyshev value/derivative series first, then apply the envelope
            # and radial power through product-rule derivatives.
            for k in range(coeff_size):
                c = coeff[k]
                series += c * T[k, i]
                series_r += c * T_r[k, i]
                series_rr += c * T_rr[k, i]

            env = env_fields[0, i]
            env_r = env_fields[1, i]
            env_rr = env_fields[2, i]
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp = offset + base

            rp = rp_fields[0, i]
            rp_r = rp_fields[1, i]
            rp_rr = rp_fields[2, i]
            out_fields[0, i] = rp * amp
            out_fields[1, i] = rp_r * amp + rp * base_r
            out_fields[2, i] = rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr
        return

    if amplitude_power == 0.5:
        for i in range(nr):
            series = 0.0
            series_r = 0.0
            series_rr = 0.0
            for k in range(coeff_size):
                c = coeff[k]
                series += c * T[k, i]
                series_r += c * T_r[k, i]
                series_rr += c * T_rr[k, i]

            env = env_fields[0, i]
            env_r = env_fields[1, i]
            env_rr = env_fields[2, i]
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp_raw = offset + base
            if amp_raw < _AMPLITUDE_POWER_FLOOR:
                amp_raw = _AMPLITUDE_POWER_FLOOR
            amp = np.sqrt(amp_raw)
            inv_amp = 1.0 / amp
            inv_amp3 = inv_amp / amp_raw
            amp_r = 0.5 * base_r * inv_amp
            amp_rr = 0.5 * base_rr * inv_amp - 0.25 * base_r * base_r * inv_amp3

            rp = rp_fields[0, i]
            rp_r = rp_fields[1, i]
            rp_rr = rp_fields[2, i]
            out_fields[0, i] = rp * amp
            out_fields[1, i] = rp_r * amp + rp * amp_r
            out_fields[2, i] = rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr
        return

    for i in range(nr):
        series = 0.0
        series_r = 0.0
        series_rr = 0.0
        # Build Chebyshev value/derivative series first, then apply the envelope
        # and radial power through product-rule derivatives.
        for k in range(coeff_size):
            c = coeff[k]
            series += c * T[k, i]
            series_r += c * T_r[k, i]
            series_rr += c * T_rr[k, i]

        env = env_fields[0, i]
        env_r = env_fields[1, i]
        env_rr = env_fields[2, i]
        base = env * series
        base_r = env_r * series + env * series_r
        base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
        amp, amp_r, amp_rr = _apply_amplitude_power(
            offset + base, base_r, base_rr, amplitude_power
        )

        rp = rp_fields[0, i]
        rp_r = rp_fields[1, i]
        rp_rr = rp_fields[2, i]
        out_fields[0, i] = rp * amp
        out_fields[1, i] = rp_r * amp + rp * amp_r
        out_fields[2, i] = rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr


@njit(cache=True, fastmath=True, nogil=True)
def update_profiles_packed_bulk(
    profile_fields: np.ndarray,
    profile_rp_fields: np.ndarray,
    profile_env_fields: np.ndarray,
    active_profile_ids: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    amplitude_powers: np.ndarray,
    x: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Refresh all active profile fields from packed x in bulk."""
    n_active = active_profile_ids.shape[0]
    nr = profile_fields.shape[2]

    for active_slot in range(n_active):
        profile_id = active_profile_ids[active_slot]
        coeff_size = lengths[active_slot]
        offset = offsets[active_slot]
        scale = scales[active_slot]
        amplitude_power = amplitude_powers[active_slot]

        if amplitude_power == 1.0:
            for i in range(nr):
                series = 0.0
                series_r = 0.0
                series_rr = 0.0

                for k in range(coeff_size):
                    # coeff_index_rows maps this active slot into packed x; using it
                    # here keeps the hot kernel independent of profile names.
                    c = x[coeff_index_rows[active_slot, k]]
                    series += c * T[k, i]
                    series_r += c * T_r[k, i]
                    series_rr += c * T_rr[k, i]

                env = profile_env_fields[profile_id, 0, i]
                env_r = profile_env_fields[profile_id, 1, i]
                env_rr = profile_env_fields[profile_id, 2, i]
                base = env * series
                base_r = env_r * series + env * series_r
                base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
                amp = offset + base

                rp = profile_rp_fields[profile_id, 0, i]
                rp_r = profile_rp_fields[profile_id, 1, i]
                rp_rr = profile_rp_fields[profile_id, 2, i]
                # Store three rows: value, rho derivative, and second rho
                # derivative.  Later stages never recompute profile derivatives.
                profile_fields[profile_id, 0, i] = scale * (rp * amp)
                profile_fields[profile_id, 1, i] = scale * (rp_r * amp + rp * base_r)
                profile_fields[profile_id, 2, i] = scale * (
                    rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr
                )
            continue

        if amplitude_power == 0.5:
            for i in range(nr):
                series = 0.0
                series_r = 0.0
                series_rr = 0.0

                for k in range(coeff_size):
                    c = x[coeff_index_rows[active_slot, k]]
                    series += c * T[k, i]
                    series_r += c * T_r[k, i]
                    series_rr += c * T_rr[k, i]

                env = profile_env_fields[profile_id, 0, i]
                env_r = profile_env_fields[profile_id, 1, i]
                env_rr = profile_env_fields[profile_id, 2, i]
                base = env * series
                base_r = env_r * series + env * series_r
                base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
                amp_raw = offset + base
                if amp_raw < _AMPLITUDE_POWER_FLOOR:
                    amp_raw = _AMPLITUDE_POWER_FLOOR
                amp = np.sqrt(amp_raw)
                inv_amp = 1.0 / amp
                inv_amp3 = inv_amp / amp_raw
                amp_r = 0.5 * base_r * inv_amp
                amp_rr = 0.5 * base_rr * inv_amp - 0.25 * base_r * base_r * inv_amp3

                rp = profile_rp_fields[profile_id, 0, i]
                rp_r = profile_rp_fields[profile_id, 1, i]
                rp_rr = profile_rp_fields[profile_id, 2, i]
                profile_fields[profile_id, 0, i] = scale * (rp * amp)
                profile_fields[profile_id, 1, i] = scale * (rp_r * amp + rp * amp_r)
                profile_fields[profile_id, 2, i] = scale * (
                    rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr
                )
            continue

        for i in range(nr):
            series = 0.0
            series_r = 0.0
            series_rr = 0.0

            for k in range(coeff_size):
                # coeff_index_rows maps this active slot into packed x; using it
                # here keeps the hot kernel independent of profile names.
                c = x[coeff_index_rows[active_slot, k]]
                series += c * T[k, i]
                series_r += c * T_r[k, i]
                series_rr += c * T_rr[k, i]

            env = profile_env_fields[profile_id, 0, i]
            env_r = profile_env_fields[profile_id, 1, i]
            env_rr = profile_env_fields[profile_id, 2, i]
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp, amp_r, amp_rr = _apply_amplitude_power(
                offset + base, base_r, base_rr, amplitude_power
            )

            rp = profile_rp_fields[profile_id, 0, i]
            rp_r = profile_rp_fields[profile_id, 1, i]
            rp_rr = profile_rp_fields[profile_id, 2, i]
            # Store three rows: value, rho derivative, and second rho
            # derivative.  Later stages never recompute profile derivatives.
            profile_fields[profile_id, 0, i] = scale * (rp * amp)
            profile_fields[profile_id, 1, i] = scale * (rp_r * amp + rp * amp_r)
            profile_fields[profile_id, 2, i] = scale * (
                rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr
            )
