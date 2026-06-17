"""Private JAX residual graph entrypoints for PF/rho/grid parity."""

from __future__ import annotations

from typing import Any

from veqpy.engine.jax.geometry import evaluate_geometry_stage_pf_rho_grid
from veqpy.engine.jax.profile import evaluate_profile_stage_pf_rho_grid
from veqpy.engine.jax.source import evaluate_source_stage_pf_rho_grid
from veqpy.engine.jax.state import JaxStaticSpec
from veqpy.workspace.field_rows import (
    GRID_POLOIDAL_COS_MTHETA_START,
    GRID_RADIAL_RHO_POWERS_START,
    GRID_RADIAL_Y,
)


def fused_residual_pf_rho_grid(
    jax_module: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    x: Any,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate the PF/rho/grid packed residual and return a stage snapshot."""

    jnp = jax_module.numpy
    profile_fields = evaluate_profile_stage_pf_rho_grid(jax_module, leaves, spec, x)
    geometry_surface_fields, geometry_radial_fields, c_fields, s_fields = (
        evaluate_geometry_stage_pf_rho_grid(jax_module, leaves, spec, profile_fields)
    )
    root_fields, alpha_state, _ = evaluate_source_stage_pf_rho_grid(
        jax_module,
        leaves,
        spec,
        geometry_radial_fields,
        geometry_surface_fields,
    )
    residual_surface_fields = _evaluate_residual_surface(
        jnp,
        alpha_state,
        root_fields,
        geometry_surface_fields,
    )
    packed = _pack_residual(jnp, leaves, spec, residual_surface_fields)
    snapshot = {
        "profile_fields": profile_fields,
        "c_family_fields": c_fields,
        "s_family_fields": s_fields,
        "geometry_surface_fields": geometry_surface_fields,
        "geometry_radial_fields": geometry_radial_fields,
        "root_fields": root_fields,
        "alpha_state": alpha_state,
        "residual_surface_fields": residual_surface_fields,
        "packed_residual": packed,
    }
    return packed, snapshot


def _evaluate_residual_surface(
    jnp: Any,
    alpha_state: Any,
    root_fields: Any,
    geometry_surface_fields: Any,
) -> Any:
    sin_tb = geometry_surface_fields[0]
    R = geometry_surface_fields[1]
    R_t = geometry_surface_fields[2]
    Z_t = geometry_surface_fields[3]
    J = geometry_surface_fields[4]
    JdivR = geometry_surface_fields[5]
    grtdivJR_t = geometry_surface_fields[6]
    gttdivJR = geometry_surface_fields[7]
    gttdivJR_r = geometry_surface_fields[8]
    psin_r = root_fields[1][:, None]
    psin_rr = root_fields[2][:, None]
    FFn_psin = root_fields[3][:, None]
    Pn_psin = root_fields[4][:, None]
    inv_J = 1.0 / J
    psin_R = -Z_t * inv_J * psin_r
    psin_Z = R_t * inv_J * psin_r
    G1n = JdivR * (FFn_psin + R * R * Pn_psin)
    G2n = gttdivJR * psin_rr + (gttdivJR_r - grtdivJR_t) * psin_r
    G = alpha_state[0] * G1n + alpha_state[1] * G2n
    Gpsin_R = G * psin_R
    Gpsin_Z = G * psin_Z
    Gpsin_R_sin_tb = Gpsin_R * sin_tb
    return jnp.stack((G, Gpsin_R, Gpsin_Z, Gpsin_R_sin_tb))


def _pack_residual(
    jnp: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    residual_surface_fields: Any,
) -> Any:
    grid_radial_fields = leaves["grid_radial_fields"]
    grid_poloidal_fields = leaves["grid_poloidal_fields"]
    weights = leaves["weights"]
    coeff_index_rows = leaves["active_coeff_index_rows"]
    T = _radial_T(grid_radial_fields, spec)
    y = grid_radial_fields[GRID_RADIAL_Y]
    rho_powers = _rho_powers(grid_radial_fields, spec)
    sin_mtheta, cos_mtheta = _poloidal_trig_views(grid_poloidal_fields, int(spec.m_max))
    sin_theta = sin_mtheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    G = residual_surface_fields[0]
    Gpsin_R = residual_surface_fields[1]
    Gpsin_Z = residual_surface_fields[2]
    Gpsin_R_sin_tb = residual_surface_fields[3]
    out = jnp.zeros((int(spec.x_size),), dtype=G.dtype)
    base_scale = 2.0 * jnp.pi / int(spec.nt)
    for slot, code in enumerate(spec.residual_block_codes):
        length = int(spec.active_lengths[slot])
        indices = coeff_index_rows[slot, :length]
        order = int(spec.residual_block_orders[slot])
        radial_power = int(spec.residual_block_radial_powers[slot])
        if int(code) == 0:
            collapsed = jnp.sum(Gpsin_R, axis=1)
            scaled = collapsed * y * weights * (base_scale * leaves["a"])
        elif int(code) == 1:
            collapsed = jnp.sum(Gpsin_Z, axis=1)
            scaled = collapsed * y * weights * (base_scale * leaves["a"])
        elif int(code) == 2:
            collapsed = jnp.sum(Gpsin_Z * sin_theta[None, :], axis=1)
            scaled = collapsed * rho * y * weights * (base_scale * (-leaves["a"]))
        elif int(code) == 3:
            collapsed = jnp.sum(Gpsin_R_sin_tb, axis=1)
            scaled = collapsed * rho * y * weights * (base_scale * (-leaves["a"]))
        elif int(code) == 4:
            collapsed = jnp.sum(Gpsin_R_sin_tb * cos_mtheta[order][None, :], axis=1)
            scaled = (
                collapsed
                * rho_powers[radial_power + 1]
                * y
                * weights
                * (base_scale * (-leaves["a"]))
            )
        elif int(code) == 5:
            collapsed = jnp.sum(Gpsin_R_sin_tb * sin_mtheta[order][None, :], axis=1)
            scaled = (
                collapsed
                * rho_powers[radial_power + 1]
                * y
                * weights
                * (base_scale * (-leaves["a"]))
            )
        elif int(code) == 6:
            collapsed = jnp.sum(G, axis=1)
            scaled = collapsed * rho2 * y * weights * base_scale
        elif int(code) == 7:
            collapsed = jnp.sum(G, axis=1)
            edge_scale = (leaves["R0"] * leaves["B0"]) * (leaves["R0"] * leaves["B0"])
            scaled = collapsed * y * y * weights * (base_scale * edge_scale)
        else:
            raise ValueError("Unknown residual block code")
        values = T[:length] @ scaled
        out = out.at[indices].set(values)
    return out


def _rho_powers(grid_radial_fields: Any, spec: JaxStaticSpec) -> Any:
    start = GRID_RADIAL_RHO_POWERS_START
    return grid_radial_fields[start : start + int(spec.k_max) + 2]


def _radial_T(grid_radial_fields: Any, spec: JaxStaticSpec) -> Any:
    start = GRID_RADIAL_RHO_POWERS_START + int(spec.k_max) + 2
    return grid_radial_fields[start : start + int(spec.l_max) + 1]


def _poloidal_trig_views(grid_poloidal_fields: Any, m_max: int) -> tuple[Any, Any]:
    block = m_max + 1
    cos_start = GRID_POLOIDAL_COS_MTHETA_START
    sin_start = cos_start + block
    cos_mtheta = grid_poloidal_fields[cos_start:sin_start]
    sin_mtheta = grid_poloidal_fields[sin_start : sin_start + block]
    return sin_mtheta, cos_mtheta
