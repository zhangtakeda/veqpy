"""Private host bridge for JAX-backed Operator residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.engine.jax.compile import compile_residual_pf_rho_grid
from veqpy.engine.jax.memory import copy_device_array_into, copy_device_array_to_numpy
from veqpy.engine.jax.state import JaxDeviceState, JaxRuntime, JaxStaticSpec

if TYPE_CHECKING:
    from veqpy.model.problem import Problem
    from veqpy.operator.build_plan import OperatorBuildPlan, ResidualBindingLayout
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace


def build_pf_rho_grid_runtime(
    *,
    jax_module: Any,
    plan: OperatorBuildPlan,
    problem: Problem,
    profile_workspace: ProfileWorkspace,
    grid_workspace: GridWorkspace,
    source_workspace: SourceWorkspace,
    residual_binding_layout: ResidualBindingLayout,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
) -> JaxRuntime:
    """Build private JAX runtime state from existing operator workspaces."""

    source_plan = plan.source_plan
    has_ip = bool(np.isfinite(float(source_plan.scaled_Ip)))
    has_beta = bool(np.isfinite(float(source_plan.beta)))
    n_axis_fix = int(np.searchsorted(grid_workspace.rho, float(fix_rho)))
    spec = JaxStaticSpec(
        route_key=tuple(source_plan.route_key),
        nr=int(grid_workspace.Nr),
        nt=int(grid_workspace.Nt),
        k_max=int(grid_workspace.K_max),
        l_max=int(grid_workspace.L_max),
        m_max=int(grid_workspace.M_max),
        x_size=int(plan.x_size),
        profile_names=tuple(plan.profile_names),
        active_profile_ids=tuple(int(v) for v in plan.active_profile_ids),
        active_lengths=tuple(int(v) for v in profile_workspace.active_lengths),
        residual_block_codes=tuple(
            int(v) for v in residual_binding_layout.active_residual_block_codes
        ),
        residual_block_orders=tuple(
            int(v) for v in residual_binding_layout.active_residual_block_orders
        ),
        residual_block_radial_powers=tuple(
            int(v) for v in residual_binding_layout.active_residual_block_radial_powers
        ),
        active_amplitude_powers=tuple(
            float(v) for v in profile_workspace.active_amplitude_powers
        ),
        c_effective_order=int(c_effective_order),
        s_effective_order=int(s_effective_order),
        n_axis_fix=n_axis_fix,
        has_Ip=has_ip,
        has_beta=has_beta,
    )
    leaves = {
        "profile_fields_template": _put(jax_module, profile_workspace.profile_fields),
        "profile_rp_fields": _put(jax_module, profile_workspace.profile_rp_fields),
        "profile_env_fields": _put(jax_module, profile_workspace.profile_env_fields),
        "active_offsets": _put(jax_module, profile_workspace.active_offsets),
        "active_scales": _put(jax_module, profile_workspace.active_scales),
        "active_coeff_index_rows": _put(jax_module, profile_workspace.active_coeff_index_rows),
        "c_family_base_fields": _put(jax_module, profile_workspace.c_family_base_fields),
        "s_family_base_fields": _put(jax_module, profile_workspace.s_family_base_fields),
        "c_family_source_profile_ids": _put(
            jax_module,
            profile_workspace.c_family_source_profile_ids,
        ),
        "s_family_source_profile_ids": _put(
            jax_module,
            profile_workspace.s_family_source_profile_ids,
        ),
        "grid_radial_fields": _put(jax_module, grid_workspace.radial_fields),
        "grid_poloidal_fields": _put(jax_module, grid_workspace.poloidal_fields),
        "weights": _put(jax_module, grid_workspace.weights),
        "differentiator": _put(jax_module, grid_workspace.differentiator),
        "accumulator": _put(jax_module, grid_workspace.accumulator),
        "materialized_heat_input": _put(jax_module, source_workspace.materialized_heat_input),
        "materialized_current_input": _put(
            jax_module,
            source_workspace.materialized_current_input,
        ),
        "scaled_Ip": _put_scalar(jax_module, source_plan.scaled_Ip),
        "beta": _put_scalar(jax_module, source_plan.beta),
        "a": _put_scalar(jax_module, problem.a),
        "R0": _put_scalar(jax_module, problem.R0),
        "Z0": _put_scalar(jax_module, problem.Z0),
        "B0": _put_scalar(jax_module, problem.B0),
    }
    runtime = JaxRuntime(static_spec=spec, device_state=JaxDeviceState(leaves=leaves))
    runtime.compiled_residual = compile_residual_pf_rho_grid(jax_module, spec)
    return runtime


def residual_var_numpy_bridge(
    *,
    jax_module: Any,
    runtime: JaxRuntime,
    x: np.ndarray,
    out: np.ndarray,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
) -> None:
    """Evaluate host NumPy ``x`` through JAX and copy residual/snapshot to host."""

    compiled = runtime.compiled_residual
    if compiled is None:
        compiled = compile_residual_pf_rho_grid(jax_module, runtime.static_spec)
        runtime.compiled_residual = compiled
    x_device = jax_module.device_put(np.asarray(x, dtype=np.float64))
    residual_device, snapshot = compiled(runtime.device_state.leaves, x_device)
    copy_device_array_into(residual_device, out)
    _publish_snapshot_to_host(
        snapshot,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
    )


def _publish_snapshot_to_host(
    snapshot: dict[str, Any],
    *,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
) -> None:
    np.copyto(profile_workspace.profile_fields, copy_device_array_to_numpy(snapshot["profile_fields"]))
    np.copyto(
        profile_workspace.c_family_fields,
        copy_device_array_to_numpy(snapshot["c_family_fields"]),
    )
    np.copyto(
        profile_workspace.s_family_fields,
        copy_device_array_to_numpy(snapshot["s_family_fields"]),
    )
    np.copyto(
        geometry_workspace.surface_fields,
        copy_device_array_to_numpy(snapshot["geometry_surface_fields"]),
    )
    np.copyto(
        geometry_workspace.radial_fields,
        copy_device_array_to_numpy(snapshot["geometry_radial_fields"]),
    )
    np.copyto(residual_workspace.root_fields, copy_device_array_to_numpy(snapshot["root_fields"]))
    np.copyto(source_workspace.alpha_state, copy_device_array_to_numpy(snapshot["alpha_state"]))
    np.copyto(
        residual_workspace.surface_fields,
        copy_device_array_to_numpy(snapshot["residual_surface_fields"]),
    )
    np.copyto(
        residual_workspace.packed_residual,
        copy_device_array_to_numpy(snapshot["packed_residual"]),
    )


def _put(jax_module: Any, value: np.ndarray) -> Any:
    return jax_module.device_put(np.asarray(value))


def _put_scalar(jax_module: Any, value: float) -> Any:
    return jax_module.device_put(np.asarray(float(value), dtype=np.float64))
