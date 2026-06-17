"""Private JAX geometry-stage entrypoints for PF/rho/grid lowering."""

from __future__ import annotations

from typing import Any

from veqpy.engine.jax.state import JaxStaticSpec
from veqpy.workspace.field_rows import (
    GRID_POLOIDAL_COS_MTHETA_START,
    GRID_POLOIDAL_THETA,
    GRID_RADIAL_RHO,
)


def evaluate_geometry_stage_pf_rho_grid(
    jax_module: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    profile_fields: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return compact geometry surface/radial fields for PF/rho/grid."""

    jnp = jax_module.numpy
    c_fields, s_fields = _build_fourier_family_fields(jnp, leaves, spec, profile_fields)
    surface_fields, radial_fields = _evaluate_geometry(
        jnp,
        leaves,
        spec,
        profile_fields,
        c_fields,
        s_fields,
    )
    return surface_fields, radial_fields, c_fields, s_fields


def _build_fourier_family_fields(
    jnp: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    profile_fields: Any,
) -> tuple[Any, Any]:
    base_c = leaves["c_family_base_fields"]
    base_s = leaves["s_family_base_fields"]
    c_ids = leaves["c_family_source_profile_ids"]
    s_ids = leaves["s_family_source_profile_ids"]
    c_fields = jnp.zeros_like(base_c)
    s_fields = jnp.zeros_like(base_s)
    for order in range(base_c.shape[0]):
        profile_id = c_ids[order]
        safe_id = jnp.maximum(profile_id, 0)
        source = profile_fields[safe_id]
        value = jnp.where(profile_id >= 0, source, base_c[order])
        value = jnp.where(order <= int(spec.c_effective_order), value, jnp.zeros_like(value))
        c_fields = c_fields.at[order].set(value)
    s_fields = s_fields.at[0].set(base_s[0])
    for order in range(1, base_s.shape[0]):
        profile_id = s_ids[order]
        safe_id = jnp.maximum(profile_id, 0)
        source = profile_fields[safe_id]
        value = jnp.where(profile_id >= 0, source, base_s[order])
        value = jnp.where(order <= int(spec.s_effective_order), value, jnp.zeros_like(value))
        s_fields = s_fields.at[order].set(value)
    return c_fields, s_fields


def _evaluate_geometry(
    jnp: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    profile_fields: Any,
    c_fields: Any,
    s_fields: Any,
) -> tuple[Any, Any]:
    grid_radial_fields = leaves["grid_radial_fields"]
    grid_poloidal_fields = leaves["grid_poloidal_fields"]
    rho = grid_radial_fields[GRID_RADIAL_RHO]
    (
        theta,
        cos_mtheta,
        sin_mtheta,
        m_cos_mtheta,
        m_sin_mtheta,
        m2_cos_mtheta,
        m2_sin_mtheta,
    ) = _poloidal_views(grid_poloidal_fields, m_max=int(spec.m_max))

    rho_i = rho[:, None]
    theta_j = theta[None, :]
    sin_t = sin_mtheta[1][None, :]
    cos_t = cos_mtheta[1][None, :]
    h = profile_fields[0]
    v = profile_fields[1]
    k = profile_fields[2]
    h_i, h_r_i, h_rr_i = h[0, :, None], h[1, :, None], h[2, :, None]
    v_r_i, v_rr_i = v[1, :, None], v[2, :, None]
    k_i, k_r_i, k_rr_i = k[0, :, None], k[1, :, None], k[2, :, None]
    c0_i = c_fields[0, 0, :, None]
    c0_r_i = c_fields[0, 1, :, None]
    c0_rr_i = c_fields[0, 2, :, None]

    tb = theta_j + c0_i
    tb_r = c0_r_i + jnp.zeros_like(theta_j)
    tb_t = jnp.ones_like(tb)
    tb_rr = c0_rr_i + jnp.zeros_like(theta_j)
    tb_rt = jnp.zeros_like(tb)
    tb_tt = jnp.zeros_like(tb)

    c_limit = min(int(spec.c_effective_order) + 1, c_fields.shape[0], cos_mtheta.shape[0])
    s_limit = min(int(spec.s_effective_order) + 1, s_fields.shape[0], sin_mtheta.shape[0])
    for order in range(1, c_limit):
        cos_kt = cos_mtheta[order][None, :]
        k_sin_kt = m_sin_mtheta[order][None, :]
        k2_cos_kt = m2_cos_mtheta[order][None, :]
        c_i = c_fields[order, 0, :, None]
        c_r_i = c_fields[order, 1, :, None]
        c_rr_i = c_fields[order, 2, :, None]
        tb = tb + c_i * cos_kt
        tb_r = tb_r + c_r_i * cos_kt
        tb_t = tb_t - c_i * k_sin_kt
        tb_rr = tb_rr + c_rr_i * cos_kt
        tb_rt = tb_rt - c_r_i * k_sin_kt
        tb_tt = tb_tt - c_i * k2_cos_kt
    for order in range(1, s_limit):
        sin_kt = sin_mtheta[order][None, :]
        k_cos_kt = m_cos_mtheta[order][None, :]
        k2_sin_kt = m2_sin_mtheta[order][None, :]
        s_i = s_fields[order, 0, :, None]
        s_r_i = s_fields[order, 1, :, None]
        s_rr_i = s_fields[order, 2, :, None]
        tb = tb + s_i * sin_kt
        tb_r = tb_r + s_r_i * sin_kt
        tb_t = tb_t + s_i * k_cos_kt
        tb_rr = tb_rr + s_rr_i * sin_kt
        tb_rt = tb_rt + s_r_i * k_cos_kt
        tb_tt = tb_tt - s_i * k2_sin_kt

    sin_tb = jnp.sin(tb)
    cos_tb = jnp.cos(tb)
    a = leaves["a"]
    r0 = leaves["R0"]
    R = jnp.maximum(r0 + a * (h_i + rho_i * cos_tb), 1.0e-6)
    R_r = a * (h_r_i + cos_tb - rho_i * sin_tb * tb_r)
    R_t = -a * rho_i * sin_tb * tb_t
    R_rr = a * (
        h_rr_i - 2.0 * sin_tb * tb_r - rho_i * (cos_tb * tb_r * tb_r + sin_tb * tb_rr)
    )
    R_rt = -a * (sin_tb * tb_t + rho_i * (cos_tb * tb_r * tb_t + sin_tb * tb_rt))
    R_tt = -a * rho_i * (cos_tb * tb_t * tb_t + sin_tb * tb_tt)
    Z_r = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t)
    Z_t = -a * rho_i * k_i * cos_t
    Z_rr = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t)
    Z_rt = -a * (k_i + rho_i * k_r_i) * cos_t
    Z_tt = a * rho_i * k_i * sin_t

    J = jnp.maximum(R_t * Z_r - R_r * Z_t, 1.0e-6)
    J_r = -(R_rr * Z_t - R_rt * Z_r + R_r * Z_rt - R_t * Z_rr)
    J_t = -(R_rt * Z_t - R_tt * Z_r + R_r * Z_tt - R_t * Z_rt)
    JR = J * R
    JR_r = J_r * R + J * R_r
    JR_t = J_t * R + J * R_t
    JdivR = J / R

    grt = R_r * R_t + Z_r * Z_t
    grt_t = R_rt * R_t + R_r * R_tt + Z_rt * Z_t + Z_r * Z_tt
    gtt = R_t * R_t + Z_t * Z_t
    gtt_r = 2.0 * (R_t * R_rt + Z_t * Z_rt)
    inv_JR = 1.0 / JR
    grtdivJR_t = (grt_t - grt * JR_t * inv_JR) * inv_JR
    gttdivJR = gtt * inv_JR
    gttdivJR_r = gtt_r * inv_JR - gtt * JR_r * inv_JR * inv_JR

    theta_scale = 2.0 * jnp.pi / int(spec.nt)
    mean_scale = 1.0 / int(spec.nt)
    S_r = jnp.sum(J, axis=1) * theta_scale
    V_r = jnp.sum(JR, axis=1) * theta_scale * (2.0 * jnp.pi)
    Kn = jnp.sum(gttdivJR, axis=1) * mean_scale
    Kn_r = jnp.sum(gttdivJR_r, axis=1) * mean_scale
    Ln_r = jnp.sum(JdivR, axis=1) * mean_scale
    surface_fields = jnp.stack(
        (sin_tb, R, R_t, Z_t, J, JdivR, grtdivJR_t, gttdivJR, gttdivJR_r)
    )
    radial_fields = jnp.stack((S_r, V_r, Kn, Kn_r, Ln_r))
    return surface_fields, radial_fields


def _poloidal_views(
    grid_poloidal_fields: Any,
    *,
    m_max: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    block = m_max + 1
    start = GRID_POLOIDAL_COS_MTHETA_START
    theta = grid_poloidal_fields[GRID_POLOIDAL_THETA]
    cos_mtheta = grid_poloidal_fields[start : start + block]
    start += block
    sin_mtheta = grid_poloidal_fields[start : start + block]
    start += block
    m_cos_mtheta = grid_poloidal_fields[start : start + block]
    start += block
    m_sin_mtheta = grid_poloidal_fields[start : start + block]
    start += block
    m2_cos_mtheta = grid_poloidal_fields[start : start + block]
    start += block
    m2_sin_mtheta = grid_poloidal_fields[start : start + block]
    return theta, cos_mtheta, sin_mtheta, m_cos_mtheta, m_sin_mtheta, m2_cos_mtheta, m2_sin_mtheta
