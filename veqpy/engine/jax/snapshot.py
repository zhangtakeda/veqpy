"""Private JAX snapshot graph for explicit host publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from veqpy.engine.jax.geometry import evaluate_geometry_stage_pf_rho_grid
from veqpy.engine.jax.profile import evaluate_profile_stage_pf_rho_grid
from veqpy.engine.jax.residual import _evaluate_residual_surface, _pack_residual
from veqpy.engine.jax.source import evaluate_source_stage_pf_rho_grid
from veqpy.engine.jax.state import JaxStaticSpec


@dataclass(frozen=True, slots=True)
class JaxSnapshotHost:
    """Private NumPy-backed snapshot published from a JAX runtime."""

    x: np.ndarray
    profile_fields: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    geometry_surface_fields: np.ndarray
    geometry_radial_fields: np.ndarray
    root_fields: np.ndarray
    alpha_state: np.ndarray
    residual_surface_fields: np.ndarray
    packed_residual: np.ndarray
    problem_generation: int
    static_signature: JaxStaticSpec
    publish_time: float


def fused_snapshot_pf_rho_grid(
    jax_module: Any,
    leaves: dict[str, Any],
    spec: JaxStaticSpec,
    x: Any,
) -> dict[str, Any]:
    """Evaluate the PF/rho/grid snapshot PyTree for explicit publication only."""

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
    return {
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
