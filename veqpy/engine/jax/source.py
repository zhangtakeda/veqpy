"""Private JAX source-stage entrypoints for PF/rho/grid lowering."""

from __future__ import annotations

from typing import Any

from veqpy.engine.jax.state import JaxStaticSpec
from veqpy.workspace.field_rows import GRID_RADIAL_RHO


def evaluate_source_stage_pf_rho_grid(
    jax_module: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    geometry_radial_fields: Any,
    geometry_surface_fields: Any,
) -> tuple[Any, Any, Any]:
    """Return root fields and source scale factors for PF/rho/grid."""

    del geometry_surface_fields
    jnp = jax_module.numpy
    grid_radial_fields = leaves["grid_radial_fields"]
    rho = grid_radial_fields[GRID_RADIAL_RHO]
    heat_input = leaves["materialized_heat_input"]
    current_input = leaves["materialized_current_input"]
    weights = leaves["weights"]
    differentiator = leaves["differentiator"]
    accumulator = leaves["accumulator"]
    V_r = geometry_radial_fields[1]
    Kn = geometry_radial_fields[2]
    Ln_r = geometry_radial_fields[4]

    integrand = Kn * (
        current_input * Ln_r + V_r * heat_input * (1.0 / (4.0 * jnp.pi * jnp.pi))
    )
    psin_r_raw = (accumulator @ integrand) * -2.0
    psi_square_sign = jnp.where(jnp.dot(psin_r_raw, weights) < 0.0, -1.0, 1.0)
    psin_r = jnp.where(psi_square_sign < 0.0, -psin_r_raw, psin_r_raw)
    psin_r = jnp.sqrt(jnp.maximum(psin_r, 1.0e-6)) / Kn
    psin_r = _regularize_psin_r(jnp, psin_r, rho, int(spec.n_axis_fix))
    integral_prof = jnp.dot(psin_r, weights)
    psin_r = psin_r / integral_prof
    psin_rr = differentiator @ psin_r
    psin = _normalize_psin(jnp, accumulator @ psin_r)

    has_ip = bool(spec.has_Ip)
    has_beta = bool(spec.has_beta)
    c2 = integral_prof * integral_prof
    if has_ip and not has_beta:
        g1n_integral = _g1n_rho_integral(
            jnp,
            current_input,
            heat_input,
            psin_r,
            Ln_r,
            V_r,
            weights,
            psi_square_sign,
        )
        alpha1 = -leaves["scaled_Ip"] / g1n_integral
    elif (not has_ip) and (not has_beta):
        alpha1 = -jnp.dot(heat_input, weights) / integral_prof
    else:
        raise ValueError("JAX PF/rho/grid supports Ip-only or unconstrained source scaling.")
    alpha2 = c2 * alpha1 if has_ip else psi_square_sign * integral_prof
    source_scale = psi_square_sign if has_ip else psi_square_sign / (alpha1 * alpha2)
    Pn_psin = source_scale * heat_input / psin_r
    FFn_psin = source_scale * current_input / psin_r
    FFn_psin = _regularize_axis_even(jnp, FFn_psin, rho, int(spec.n_axis_fix))
    root_fields = jnp.stack((psin, psin_r, psin_rr, FFn_psin, Pn_psin))
    alpha_state = jnp.stack((alpha1, alpha2))
    return root_fields, alpha_state, jnp.stack((FFn_psin, Pn_psin))


def _normalize_psin(jnp: Any, psin: Any) -> Any:
    offset = psin[0]
    scale = psin[-1] - offset
    out = (psin - offset) / scale
    out = out.at[0].set(0.0)
    out = out.at[-1].set(1.0)
    return out


def _regularize_axis_linear(jnp: Any, profile: Any, rho: Any, n_fix: int) -> Any:
    if n_fix <= 0:
        return profile
    anchor0 = n_fix
    anchor1 = n_fix + 1
    rho0 = rho[anchor0]
    rho1 = rho[anchor1]
    x0 = rho0 * rho0
    x1 = rho1 * rho1
    slope0 = profile[anchor0] / rho0
    slope1 = profile[anchor1] / rho1
    slope_gradient = (slope1 - slope0) / (x1 - x0)
    idx = jnp.arange(n_fix)
    x = rho[idx] * rho[idx]
    repaired = rho[idx] * (slope0 + slope_gradient * (x - x0))
    return profile.at[:n_fix].set(repaired)


def _regularize_psin_r(jnp: Any, psin_r: Any, rho: Any, n_fix: int) -> Any:
    psin_r = _regularize_axis_linear(jnp, psin_r, rho, n_fix)
    return jnp.maximum(psin_r, 1.0e-10)


def _regularize_axis_even(jnp: Any, profile: Any, rho: Any, n_fix: int) -> Any:
    if n_fix <= 0:
        return profile
    anchor0 = n_fix
    anchor1 = n_fix + 1
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    value0 = profile[anchor0]
    value1 = profile[anchor1]
    value_gradient = (value1 - value0) / (x1 - x0)
    idx = jnp.arange(n_fix)
    x = rho[idx] * rho[idx]
    repaired = value0 + value_gradient * (x - x0)
    return profile.at[:n_fix].set(repaired)


def _g1n_rho_integral(
    jnp: Any,
    FFn_r: Any,
    Pn_r: Any,
    psin_r: Any,
    Ln_r: Any,
    V_r: Any,
    weights: Any,
    source_scale: Any,
) -> Any:
    two_pi = 2.0 * jnp.pi
    inv_two_pi = 1.0 / two_pi
    terms = (
        weights
        * source_scale
        / psin_r
        * (two_pi * Ln_r * FFn_r + inv_two_pi * V_r * Pn_r)
    )
    return jnp.sum(terms)
