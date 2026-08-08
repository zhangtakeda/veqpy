"""Self-contained Numba calculation kernels for ``Equilibrium``.

This module deliberately lives in :mod:`veqpy.model`: model-side field
materialization must not call solver kernels, adapters, or backend workspaces.
Each Reactive property owns the result array it materializes; these kernels only
fill that new result and any call-local scratch arrays. The coordinate/metric
stages remain grouped where their intermediate derivatives are genuinely shared.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MU0 = 4.0e-7 * np.pi
_AMPLITUDE_POWER_FLOOR = 1.0e-10

# Public packed geometry rows keep the historical ``surface_fields`` contract.
SIN_TB = 0
R = 1
R_T = 2
Z_T = 3
J = 4
JDIVR = 5
GRTDIVJR_T = 6
GTTDIVJR = 7
GTTDIVJR_R = 8

# Private coordinate intermediates share the same allocation.
R_R = 9
R_RR = 10
R_RT = 11
R_TT = 12
Z = 13
Z_R = 14
Z_RR = 15
Z_RT = 16
Z_TT = 17

S_R = 0
V_R = 1
KN = 2
KN_R = 3
LN_R = 4

# IMAS equilibrium ``profiles_1d`` geometric coefficients.
GM1 = 0
GM2 = 1
GM3 = 2
GM4 = 3
GM5 = 4
GM6 = 5
GM7 = 6
GM8 = 7
GM9 = 8

RHO_TOR = 0
RHO_TOR_NORM = 1
RHO_TOR_R = 2
RHO_TOR_NORM_R = 3

@njit(cache=True, nogil=True, inline="always")
def _power_terms_at(rho: float, power: int) -> tuple[float, float, float]:
    if power == 0:
        return 1.0, 0.0, 0.0
    value = rho**power
    first = power * rho ** (power - 1)
    second = 0.0 if power == 1 else power * (power - 1) * rho ** (power - 2)
    return value, first, second


@njit(cache=True, nogil=True, inline="always")
def _envelope_terms_at(rho: float, envelope_power: int) -> tuple[float, float, float]:
    if envelope_power == 0:
        return 1.0, 0.0, 0.0
    y = 1.0 - rho * rho
    if envelope_power == 1:
        return y, -2.0 * rho, -2.0
    value = y**envelope_power
    first = -2.0 * envelope_power * rho * y ** (envelope_power - 1)
    second = -2.0 * envelope_power * y ** (envelope_power - 1)
    second += (
        4.0
        * envelope_power
        * (envelope_power - 1)
        * rho
        * rho
        * y ** (envelope_power - 2)
    )
    return value, first, second


@njit(cache=True, nogil=True, inline="always")
def _amplitude_terms(
    value: float,
    first: float,
    second: float,
    amplitude_power: float,
) -> tuple[float, float, float]:
    if amplitude_power == 1.0:
        return value, first, second
    safe = max(value, _AMPLITUDE_POWER_FLOOR)
    if amplitude_power == 0.5:
        powered = np.sqrt(safe)
        inv = 1.0 / powered
        return (
            powered,
            0.5 * first * inv,
            0.5 * second * inv - 0.25 * first * first * inv / safe,
        )
    powered = safe**amplitude_power
    powered_r = amplitude_power * safe ** (amplitude_power - 1.0) * first
    powered_rr = amplitude_power * safe ** (amplitude_power - 1.0) * second
    powered_rr += (
        amplitude_power
        * (amplitude_power - 1.0)
        * safe ** (amplitude_power - 2.0)
        * first
        * first
    )
    return powered, powered_r, powered_rr


@njit(cache=True, nogil=True)
def update_profile_fields(
    out: np.ndarray,
    rho: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    scale: float,
    power: int,
    envelope_power: int,
    amplitude_power: float,
    offset: float,
    coeff: np.ndarray,
    coeff_count: int,
) -> None:
    """Evaluate one profile and two radial derivatives without temporaries."""

    for i in range(rho.shape[0]):
        rp, rp_r, rp_rr = _power_terms_at(rho[i], power)
        if coeff_count == 0:
            amp, amp_r, amp_rr = _amplitude_terms(
                offset, 0.0, 0.0, amplitude_power
            )
        else:
            series = 0.0
            series_r = 0.0
            series_rr = 0.0
            for k in range(coeff_count):
                coefficient = coeff[k]
                series += coefficient * T[k, i]
                series_r += coefficient * T_r[k, i]
                series_rr += coefficient * T_rr[k, i]
            env, env_r, env_rr = _envelope_terms_at(rho[i], envelope_power)
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp, amp_r, amp_rr = _amplitude_terms(
                offset + base, base_r, base_rr, amplitude_power
            )

        out[0, i] = scale * rp * amp
        out[1, i] = scale * (rp_r * amp + rp * amp_r)
        out[2, i] = scale * (rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr)


@njit(cache=True, nogil=True)
def update_r_coordinates(
    surface: np.ndarray,
    a: float,
    R0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
) -> None:
    c_limit = min(c.shape[0], cos_mtheta.shape[0])
    s_limit = min(s.shape[0], sin_mtheta.shape[0])
    for i in range(rho.shape[0]):
        rho_i = rho[i]
        for j in range(theta.shape[0]):
            tb = theta[j] + c[0, 0, i]
            tb_r = c[0, 1, i]
            tb_t = 1.0
            tb_rr = c[0, 2, i]
            tb_rt = 0.0
            tb_tt = 0.0
            for order in range(1, c_limit):
                ci = c[order, 0, i]
                cir = c[order, 1, i]
                cirr = c[order, 2, i]
                tb += ci * cos_mtheta[order, j]
                tb_r += cir * cos_mtheta[order, j]
                tb_t -= ci * m_sin_mtheta[order, j]
                tb_rr += cirr * cos_mtheta[order, j]
                tb_rt -= cir * m_sin_mtheta[order, j]
                tb_tt -= ci * m2_cos_mtheta[order, j]
            for order in range(1, s_limit):
                si = s[order, 0, i]
                sir = s[order, 1, i]
                sirr = s[order, 2, i]
                tb += si * sin_mtheta[order, j]
                tb_r += sir * sin_mtheta[order, j]
                tb_t += si * m_cos_mtheta[order, j]
                tb_rr += sirr * sin_mtheta[order, j]
                tb_rt += sir * m_cos_mtheta[order, j]
                tb_tt -= si * m2_sin_mtheta[order, j]

            cos_tb = np.cos(tb)
            sin_tb = np.sin(tb)
            radius = R0 + a * (h[0, i] + rho_i * cos_tb)
            if radius < 1.0e-6:
                radius = 1.0e-6
            surface[SIN_TB, i, j] = sin_tb
            surface[R, i, j] = radius
            surface[R_R, i, j] = a * (
                h[1, i] + cos_tb - rho_i * sin_tb * tb_r
            )
            surface[R_T, i, j] = -a * rho_i * sin_tb * tb_t
            surface[R_RR, i, j] = a * (
                h[2, i]
                - 2.0 * sin_tb * tb_r
                - rho_i * (cos_tb * tb_r * tb_r + sin_tb * tb_rr)
            )
            surface[R_RT, i, j] = -a * (
                sin_tb * tb_t
                + rho_i * (cos_tb * tb_r * tb_t + sin_tb * tb_rt)
            )
            surface[R_TT, i, j] = -a * rho_i * (
                cos_tb * tb_t * tb_t + sin_tb * tb_tt
            )


@njit(cache=True, nogil=True)
def update_z_coordinates(
    surface: np.ndarray,
    a: float,
    Z0: float,
    rho: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
) -> None:
    for i in range(rho.shape[0]):
        rho_i = rho[i]
        for j in range(sin_theta.shape[0]):
            sin_t = sin_theta[j]
            cos_t = cos_theta[j]
            surface[Z, i, j] = Z0 + a * (v[0, i] - rho_i * kappa[0, i] * sin_t)
            surface[Z_R, i, j] = a * (
                v[1, i] - (kappa[0, i] + rho_i * kappa[1, i]) * sin_t
            )
            surface[Z_T, i, j] = -a * rho_i * kappa[0, i] * cos_t
            surface[Z_RR, i, j] = a * (
                v[2, i] - (2.0 * kappa[1, i] + rho_i * kappa[2, i]) * sin_t
            )
            surface[Z_RT, i, j] = -a * (
                kappa[0, i] + rho_i * kappa[1, i]
            ) * cos_t
            surface[Z_TT, i, j] = a * rho_i * kappa[0, i] * sin_t


@njit(cache=True, nogil=True, inline="always")
def _axis_even_rho2_limit(value_1: float, value_2: float, rho: np.ndarray) -> float:
    """Extrapolate an even finite scalar from the first two off-axis rows."""

    x0 = rho[0] * rho[0]
    x1 = rho[1] * rho[1]
    x2 = rho[2] * rho[2]
    denominator = x2 - x1
    if denominator == 0.0 or not np.isfinite(value_1) or not np.isfinite(value_2):
        return np.nan
    return value_1 + (value_2 - value_1) / denominator * (x0 - x1)


@njit(cache=True, nogil=True, inline="always")
def _axis_linear_rho_limit(value_1: float, value_2: float, rho: np.ndarray) -> float:
    """Extrapolate a local surface quantity linearly in ``rho``."""

    denominator = rho[2] - rho[1]
    if denominator == 0.0 or not np.isfinite(value_1) or not np.isfinite(value_2):
        return np.nan
    return value_1 + (value_2 - value_1) / denominator * (rho[0] - rho[1])


@njit(cache=True, nogil=True, inline="always")
def _axis_leading_rho_coefficient(
    value_1: float, value_2: float, rho: np.ndarray
) -> float:
    """Recover the leading coefficient of ``value = rho*(A + O(rho))``."""

    if rho[1] == 0.0 or rho[2] == 0.0:
        return np.nan
    return _axis_linear_rho_limit(value_1 / rho[1], value_2 / rho[2], rho)


@njit(cache=True, nogil=True)
def update_metric_geometry(
    surface: np.ndarray,
    radial: np.ndarray,
    rho: np.ndarray,
) -> int:
    nr = surface.shape[1]
    nt = surface.shape[2]
    has_axis = nr >= 3 and abs(rho[0]) < 1.0e-10
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    for i in range(nr):
        sum_j = 0.0
        sum_jr = 0.0
        sum_gtt = 0.0
        sum_gtt_r = 0.0
        sum_jdivr = 0.0
        for j in range(nt):
            radius = surface[R, i, j]
            rr = surface[R_R, i, j]
            rt = surface[R_T, i, j]
            rrr = surface[R_RR, i, j]
            rrt = surface[R_RT, i, j]
            rtt = surface[R_TT, i, j]
            zr = surface[Z_R, i, j]
            zt = surface[Z_T, i, j]
            zrr = surface[Z_RR, i, j]
            zrt = surface[Z_RT, i, j]
            ztt = surface[Z_TT, i, j]

            jac = rt * zr - rr * zt
            jac_r = -(rrr * zt - rrt * zr + rr * zrt - rt * zrr)
            jac_t = -(rrt * zt - rtt * zr + rr * ztt - rt * zrt)
            jr = jac * radius
            jr_r = jac_r * radius + jac * rr
            jr_t = jac_t * radius + jac * rt
            grt = rr * rt + zr * zt
            grt_t = rrt * rt + rr * rtt + zrt * zt + zr * ztt
            gtt = rt * rt + zt * zt
            gtt_r = 2.0 * (rt * rrt + zt * zrt)
            jdivr = jac / radius
            if has_axis and i == 0:
                # The public Jacobian remains the raw coordinate Jacobian.  The
                # three metric ratios have removable axis singularities and are
                # reconstructed after all off-axis rows are available.
                grtdivjr_t = np.nan
                gttdivjr = np.nan
                gttdivjr_r = np.nan
            else:
                if jr == 0.0 or not np.isfinite(jr):
                    return i + 1
                inv_jr = 1.0 / jr
                grtdivjr_t = (grt_t - grt * jr_t * inv_jr) * inv_jr
                gttdivjr = gtt * inv_jr
                gttdivjr_r = gtt_r * inv_jr - gtt * jr_r * inv_jr * inv_jr

            surface[J, i, j] = jac
            surface[JDIVR, i, j] = jdivr
            surface[GRTDIVJR_T, i, j] = grtdivjr_t
            surface[GTTDIVJR, i, j] = gttdivjr
            surface[GTTDIVJR_R, i, j] = gttdivjr_r
            sum_j += jac
            sum_jr += jr
            if not (has_axis and i == 0):
                sum_gtt += gttdivjr
                sum_gtt_r += gttdivjr_r
            sum_jdivr += jdivr

        radial[S_R, i] = sum_j * theta_scale
        radial[V_R, i] = sum_jr * theta_scale * 2.0 * np.pi
        radial[KN, i] = sum_gtt * mean_scale
        radial[KN_R, i] = sum_gtt_r * mean_scale
        radial[LN_R, i] = sum_jdivr * mean_scale

    if has_axis:
        for j in range(nt):
            leading_gtt = _axis_leading_rho_coefficient(
                surface[GTTDIVJR, 1, j],
                surface[GTTDIVJR, 2, j],
                rho,
            )
            surface[GTTDIVJR, 0, j] = rho[0] * leading_gtt
            surface[GTTDIVJR_R, 0, j] = leading_gtt
            surface[GRTDIVJR_T, 0, j] = _axis_linear_rho_limit(
                surface[GRTDIVJR_T, 1, j],
                surface[GRTDIVJR_T, 2, j],
                rho,
            )
        sum_gtt = 0.0
        sum_gtt_r = 0.0
        for j in range(nt):
            sum_gtt += surface[GTTDIVJR, 0, j]
            sum_gtt_r += surface[GTTDIVJR_R, 0, j]
        radial[KN, 0] = sum_gtt * mean_scale
        radial[KN_R, 0] = sum_gtt_r * mean_scale

    for i in range(nr):
        for j in range(nt):
            if not np.isfinite(surface[J, i, j]):
                return i + 1
            if not np.isfinite(surface[JDIVR, i, j]):
                return i + 1
            if not np.isfinite(surface[GRTDIVJR_T, i, j]):
                return i + 1
            if not np.isfinite(surface[GTTDIVJR, i, j]):
                return i + 1
            if not np.isfinite(surface[GTTDIVJR_R, i, j]):
                return i + 1
        for field in range(radial.shape[0]):
            if not np.isfinite(radial[field, i]):
                return i + 1
    return 0


@njit(cache=True, nogil=True)
def materialize_metric_geometry(
    r_surface: np.ndarray,
    z_surface: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine independent coordinate stages and materialize metric outputs."""

    surface = np.empty_like(r_surface)
    radial = np.empty((5, rho.shape[0]), dtype=np.float64)
    r_rows = (SIN_TB, R, R_R, R_T, R_RR, R_RT, R_TT)
    z_rows = (Z, Z_R, Z_T, Z_RR, Z_RT, Z_TT)
    for row in r_rows:
        for i in range(surface.shape[1]):
            for j in range(surface.shape[2]):
                surface[row, i, j] = r_surface[row, i, j]
    for row in z_rows:
        for i in range(surface.shape[1]):
            for j in range(surface.shape[2]):
                surface[row, i, j] = z_surface[row, i, j]
    invalid = update_metric_geometry(surface, radial, rho)
    return surface[:9], radial, invalid


@njit(cache=True, nogil=True)
def update_rc(out: np.ndarray, R0: float, a: float, h: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = R0 + a * h[i]


@njit(cache=True, nogil=True)
def update_epsilon(out: np.ndarray, a: float, rho: np.ndarray, rc: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = a * rho[i] / rc[i]


@njit(cache=True, nogil=True)
def update_surface_area(out: np.ndarray, radius: np.ndarray, z_t: np.ndarray) -> None:
    scale = -2.0 * np.pi / radius.shape[1]
    for i in range(radius.shape[0]):
        total = 0.0
        for j in range(radius.shape[1]):
            total += radius[i, j] * z_t[i, j]
        out[i] = scale * total


@njit(cache=True, nogil=True)
def update_volume(out: np.ndarray, radius: np.ndarray, z_t: np.ndarray) -> None:
    scale = -2.0 * np.pi * np.pi / radius.shape[1]
    for i in range(radius.shape[0]):
        total = 0.0
        for j in range(radius.shape[1]):
            total += radius[i, j] * radius[i, j] * z_t[i, j]
        out[i] = scale * total


@njit(cache=True, nogil=True)
def update_scaled_product(
    out: np.ndarray, left: np.ndarray, right: np.ndarray, scale: float
) -> None:
    for i in range(out.shape[0]):
        out[i] = scale * left[i] * right[i]


@njit(cache=True, nogil=True)
def update_scaled_copy(out: np.ndarray, source: np.ndarray, scale: float) -> None:
    for i in range(out.shape[0]):
        out[i] = scale * source[i]


@njit(cache=True, nogil=True)
def update_f2(
    out: np.ndarray,
    ff_r: np.ndarray,
    accumulator: np.ndarray,
    edge_f2: float,
) -> None:
    edge = 0.0
    for k in range(out.shape[0]):
        edge += accumulator[-1, k] * ff_r[k]
    for i in range(out.shape[0]):
        total = 0.0
        for k in range(out.shape[0]):
            total += accumulator[i, k] * ff_r[k]
        out[i] = edge_f2 + 2.0 * (total - edge)


@njit(cache=True, nogil=True)
def update_f(out: np.ndarray, f2: np.ndarray, edge_f: float) -> int:
    sign = -1.0 if edge_f < 0.0 else 1.0
    for i in range(out.shape[0]):
        if f2[i] < 1.0e-6:
            return i + 1
        out[i] = sign * np.sqrt(f2[i])
    return 0


@njit(cache=True, nogil=True)
def update_pressure(
    out: np.ndarray,
    p_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    p0: float,
) -> None:
    edge_integral = 0.0
    for k in range(out.shape[0]):
        edge_integral += weights[k] * p_r[k]
    for i in range(out.shape[0]):
        prefix = 0.0
        for k in range(out.shape[0]):
            prefix += accumulator[i, k] * p_r[k]
        out[i] = p0 + prefix - edge_integral


@njit(cache=True, nogil=True)
def update_beta_t(
    pressure: np.ndarray,
    volume_r: np.ndarray,
    weights: np.ndarray,
    B0: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for i in range(pressure.shape[0]):
        weighted_volume = weights[i] * volume_r[i]
        numerator += weighted_volume * pressure[i]
        denominator += weighted_volume
    return 2.0 * MU0 * numerator / (denominator * B0 * B0)


@njit(cache=True, nogil=True)
def update_gn1(
    out: np.ndarray,
    radius: np.ndarray,
    jdivr: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            r = radius[i, j]
            out[i, j] = jdivr[i, j] * (ffn_psin[i] + r * r * pn_psin[i])


@njit(cache=True, nogil=True)
def update_gn2(
    out: np.ndarray,
    gttdivjr: np.ndarray,
    gttdivjr_r: np.ndarray,
    grtdivjr_t: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = gttdivjr[i, j] * psin_rr[i]
            out[i, j] += (gttdivjr_r[i, j] - grtdivjr_t[i, j]) * psin_r[i]


@njit(cache=True, nogil=True)
def update_linear_combination_2d(
    out: np.ndarray, left: np.ndarray, right: np.ndarray, a: float, b: float
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = a * left[i, j] + b * right[i, j]


@njit(cache=True, nogil=True)
def update_ip(gn1: np.ndarray, weights: np.ndarray, alpha1: float) -> float:
    total = 0.0
    for i in range(gn1.shape[0]):
        row_sum = 0.0
        for j in range(gn1.shape[1]):
            row_sum += gn1[i, j]
        total += weights[i] * row_sum
    return -alpha1 * (2.0 * np.pi / gn1.shape[1]) * total / MU0


@njit(cache=True, nogil=True, inline="always")
def _regularize_axis_rho2_1d(values: np.ndarray, rho: np.ndarray) -> None:
    if values.shape[0] < 3 or abs(rho[0]) >= 1.0e-10:
        return
    values[0] = _axis_even_rho2_limit(values[1], values[2], rho)


@njit(cache=True, nogil=True, inline="always")
def _regularize_axis_sqrt_rho_1d(values: np.ndarray, rho: np.ndarray) -> None:
    """Recover a quantity with leading ``sqrt(rho)`` axis behavior."""

    if values.shape[0] < 3 or abs(rho[0]) >= 1.0e-10:
        return
    if rho[1] <= 0.0 or rho[2] <= 0.0:
        values[0] = np.nan
        return
    coefficient = _axis_linear_rho_limit(
        values[1] / np.sqrt(rho[1]),
        values[2] / np.sqrt(rho[2]),
        rho,
    )
    values[0] = np.sqrt(rho[0]) * coefficient


@njit(cache=True, nogil=True, inline="always")
def _first_nonfinite_1d(values: np.ndarray) -> int:
    for i in range(values.shape[0]):
        if not np.isfinite(values[i]):
            return i + 1
    return 0


@njit(cache=True, nogil=True)
def update_q(
    out: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    alpha2: float,
    psin_r: np.ndarray,
    rho: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        denominator = alpha2 * psin_r[i]
        out[i] = np.nan if denominator == 0.0 else f[i] * ln_r[i] / denominator
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_shear(
    out: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    differentiator: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        derivative = 0.0
        for k in range(out.shape[0]):
            derivative += differentiator[i, k] * q[k]
        out[i] = rho[i] * derivative / q[i]


@njit(cache=True, nogil=True)
def update_itor(
    out: np.ndarray, kn: np.ndarray, alpha2: float, psin_r: np.ndarray
) -> None:
    scale = 2.0 * np.pi * alpha2 / MU0
    for i in range(out.shape[0]):
        out[i] = scale * kn[i] * psin_r[i]


@njit(cache=True, nogil=True)
def update_jtor(
    out: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
    ln_r: np.ndarray,
    s_r: np.ndarray,
    v_r: np.ndarray,
    alpha1: float,
    rho: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        if s_r[i] == 0.0:
            out[i] = np.nan
        else:
            out[i] = -alpha1 / (MU0 * s_r[i]) * (
                2.0 * np.pi * ffn_psin[i] * ln_r[i]
                + v_r[i] * pn_psin[i] / (2.0 * np.pi)
            )
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jpara(
    out: np.ndarray,
    f: np.ndarray,
    kn: np.ndarray,
    kn_r: np.ndarray,
    ln_r: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
    differentiator: np.ndarray,
) -> int:
    for i in range(out.shape[0]):
        if f[i] == 0.0 or ln_r[i] == 0.0:
            out[i] = np.nan
        else:
            derivative = 0.0
            for k in range(out.shape[0]):
                derivative += differentiator[i, k] * f[k]
            term = kn_r[i] * psin_r[i] / f[i]
            term += kn[i] * psin_rr[i] / f[i]
            term -= kn[i] * psin_r[i] * derivative / (f[i] * f[i])
            out[i] = alpha2 / MU0 * f[i] / ln_r[i] * term
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jtotal(
    out: np.ndarray,
    jpara: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    volume_r: np.ndarray,
    B0: float,
    rho: np.ndarray,
) -> int:
    """Write the IMAS convention ``<J·B>/B0`` from the PJ2 current."""

    gm1_scale = (2.0 * np.pi) ** 2
    for i in range(out.shape[0]):
        if volume_r[i] == 0.0 or B0 == 0.0:
            out[i] = np.nan
        else:
            gm1 = gm1_scale * ln_r[i] / volume_r[i]
            out[i] = jpara[i] * f[i] * gm1 / B0
    _regularize_axis_rho2_1d(out, rho)
    return _first_nonfinite_1d(out)


@njit(cache=True, nogil=True)
def update_jphi(
    out: np.ndarray,
    radius: np.ndarray,
    ffn_psin: np.ndarray,
    pn_psin: np.ndarray,
    alpha1: float,
) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            r = radius[i, j]
            out[i, j] = -alpha1 / (MU0 * r) * (
                ffn_psin[i] + r * r * pn_psin[i]
            )


@njit(cache=True, nogil=True)
def update_phi(
    out: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    accumulator: np.ndarray,
) -> None:
    for i in range(out.shape[0]):
        total = 0.0
        for k in range(out.shape[0]):
            total += accumulator[i, k] * f[k] * ln_r[k]
        out[i] = 2.0 * np.pi * total


@njit(cache=True, nogil=True)
def update_toroidal_flux_coordinates(
    out: np.ndarray,
    phi: np.ndarray,
    f: np.ndarray,
    ln_r: np.ndarray,
    B0: float,
    rho: np.ndarray,
) -> int:
    """Materialize IMAS ``rho_tor`` coordinates and radial derivatives."""

    if B0 == 0.0 or not np.isfinite(B0):
        return 1

    for i in range(phi.shape[0]):
        rho_tor_squared = phi[i] / (np.pi * B0)
        if rho_tor_squared < 0.0 or not np.isfinite(rho_tor_squared):
            return i + 1
        rho_tor = np.sqrt(rho_tor_squared)
        out[RHO_TOR, i] = rho_tor
        if rho_tor == 0.0:
            out[RHO_TOR_R, i] = np.nan
        else:
            out[RHO_TOR_R, i] = f[i] * ln_r[i] / (B0 * rho_tor)

    _regularize_axis_rho2_1d(out[RHO_TOR_R], rho)
    axis_value = out[RHO_TOR, 0]
    span = out[RHO_TOR, -1] - axis_value
    if span == 0.0 or not np.isfinite(span):
        return phi.shape[0]
    for i in range(phi.shape[0]):
        out[RHO_TOR_NORM, i] = (out[RHO_TOR, i] - axis_value) / span
        out[RHO_TOR_NORM_R, i] = out[RHO_TOR_R, i] / span

    for row in range(out.shape[0]):
        invalid = _first_nonfinite_1d(out[row])
        if invalid:
            return invalid
    return 0


@njit(cache=True, nogil=True)
def update_gm(
    out: np.ndarray,
    radius: np.ndarray,
    jacobian: np.ndarray,
    gttdivjr: np.ndarray,
    f: np.ndarray,
    psin_r: np.ndarray,
    ln_r: np.ndarray,
    surface_area_r: np.ndarray,
    volume_r: np.ndarray,
    rho_tor_r: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
) -> int:
    """Materialize the IMAS gm1--gm9 flux-surface geometry coefficients."""

    has_axis = out.shape[1] >= 3 and abs(rho[0]) < 1.0e-10
    gm1_scale = (2.0 * np.pi) ** 2
    gm9_scale = 2.0 * np.pi
    for i in range(out.shape[1]):
        if has_axis and i == 0:
            for field in range(out.shape[0]):
                out[field, i] = np.nan
            continue
        if volume_r[i] == 0.0 or not np.isfinite(volume_r[i]):
            return i + 1

        weight_sum = 0.0
        gm2_sum = 0.0
        gm3_sum = 0.0
        gm4_sum = 0.0
        gm5_sum = 0.0
        gm6_sum = 0.0
        gm7_sum = 0.0
        gm8_sum = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            jr = jac * r
            if r == 0.0 or jac == 0.0 or not np.isfinite(jr):
                return i + 1

            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / jr
            grad_rho_tor2 = (
                rho_tor_r[i]
                * rho_tor_r[i]
                * gttdivjr[i, j]
                * r
                / jac
            )
            b2 = bphi2 + bp2
            if (
                b2 <= 0.0
                or grad_rho_tor2 < 0.0
                or not np.isfinite(b2)
                or not np.isfinite(grad_rho_tor2)
            ):
                return i + 1

            weight_sum += jr
            gm2_sum += jr * grad_rho_tor2 / (r * r)
            gm3_sum += jr * grad_rho_tor2
            gm4_sum += jr / b2
            gm5_sum += jr * b2
            gm6_sum += jr * grad_rho_tor2 / b2
            gm7_sum += jr * np.sqrt(grad_rho_tor2)
            gm8_sum += jr * r

        if weight_sum == 0.0 or not np.isfinite(weight_sum):
            return i + 1
        inv_weight = 1.0 / weight_sum
        out[GM1, i] = gm1_scale * ln_r[i] / volume_r[i]
        out[GM2, i] = gm2_sum * inv_weight
        out[GM3, i] = gm3_sum * inv_weight
        out[GM4, i] = gm4_sum * inv_weight
        out[GM5, i] = gm5_sum * inv_weight
        out[GM6, i] = gm6_sum * inv_weight
        out[GM7, i] = gm7_sum * inv_weight
        out[GM8, i] = gm8_sum * inv_weight
        out[GM9, i] = gm9_scale * surface_area_r[i] / volume_r[i]

    if has_axis:
        for field in range(out.shape[0]):
            _regularize_axis_rho2_1d(out[field], rho)

    for field in range(out.shape[0]):
        invalid = _first_nonfinite_1d(out[field])
        if invalid:
            return invalid
    return 0


@njit(cache=True, nogil=True)
def update_ftrap(
    out: np.ndarray,
    radius: np.ndarray,
    jacobian: np.ndarray,
    gttdivjr: np.ndarray,
    f: np.ndarray,
    psin_r: np.ndarray,
    alpha2: float,
    rho: np.ndarray,
) -> int:
    has_axis = out.shape[0] >= 3 and abs(rho[0]) < 1.0e-10
    for i in range(out.shape[0]):
        if has_axis and i == 0:
            out[i] = np.nan
            continue
        weight_sum = 0.0
        b_sum = 0.0
        b2_sum = 0.0
        bmax = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / (jac * r)
            magnetic = np.sqrt(bphi2 + bp2)
            weight = jac * r
            weight_sum += weight
            b_sum += magnetic * weight
            b2_sum += magnetic * magnetic * weight
            bmax = max(bmax, magnetic)
        if weight_sum == 0.0 or bmax == 0.0:
            out[i] = np.nan
            continue
        h = b_sum / (weight_sum * bmax)
        h2 = b2_sum / (weight_sum * bmax * bmax)
        hf_sum = 0.0
        for j in range(radius.shape[1]):
            r = radius[i, j]
            jac = jacobian[i, j]
            bphi2 = (f[i] / r) ** 2
            bp2 = (alpha2 * psin_r[i]) ** 2 * gttdivjr[i, j] / (jac * r)
            x = np.sqrt(bphi2 + bp2) / bmax
            one_minus_x = 1.0 - x
            if one_minus_x < 0.0 and one_minus_x > -1.0e-14:
                one_minus_x = 0.0
            if one_minus_x < 0.0 or x == 0.0:
                hf_sum = np.nan
                break
            integrand = (1.0 - np.sqrt(one_minus_x) * (1.0 + 0.5 * x)) / (x * x)
            hf_sum += integrand * jac * r
        hf = hf_sum / weight_sum
        ftu = 1.0 - h2 / (h * h) * (
            1.0 - np.sqrt(1.0 - h) * (1.0 + 0.5 * h)
        )
        ftl = 1.0 - h2 * hf
        out[i] = 0.75 * ftu + 0.25 * ftl
    _regularize_axis_sqrt_rho_1d(out, rho)
    return _first_nonfinite_1d(out)
