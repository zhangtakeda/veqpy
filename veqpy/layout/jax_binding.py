"""
Module: layout.jax_binding

Role:
- Build private JAX operator layout shells.
- Keep unsupported JAX behavior explicit until residual parity is implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine.backend import JaxBackendOptions, UnsupportedBackendFeature
from veqpy.engine.jax.config import require_jax
from veqpy.engine.jax.operator import build_pf_rho_grid_runtime, residual_var_numpy_bridge

if TYPE_CHECKING:
    from veqpy.model.problem import Problem
    from veqpy.operator.build_plan import OperatorBuildPlan, ResidualBindingLayout
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace

from .runtime import OperatorLayout

SUPPORTED_JAX_ROUTE_KEYS: frozenset[tuple[str, str, str]] = frozenset({("PF", "rho", "grid")})


def supported_jax_routes() -> list[str]:
    """Return human-readable supported route keys for error messages."""

    return ["/".join(route_key) for route_key in sorted(SUPPORTED_JAX_ROUTE_KEYS)]


def build_jax_operator_layout(
    *,
    plan: OperatorBuildPlan,
    problem: Problem,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
    grid_workspace: GridWorkspace,
    residual_binding_layout: ResidualBindingLayout,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
    backend_options: object | None = None,
) -> OperatorLayout:
    """Build a JAX layout shell after route capability validation."""

    del psin_profile_fields_available
    route_key = plan.source_plan.route_key
    if route_key not in SUPPORTED_JAX_ROUTE_KEYS:
        _raise_unsupported_route(route_key)

    options = backend_options if isinstance(backend_options, JaxBackendOptions) else None
    jax_module = require_jax(options)
    runtime = build_pf_rho_grid_runtime(
        jax_module=jax_module,
        plan=plan,
        problem=problem,
        profile_workspace=profile_workspace,
        grid_workspace=grid_workspace,
        source_workspace=source_workspace,
        residual_binding_layout=residual_binding_layout,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        fix_rho=fix_rho,
    )
    return _pf_rho_grid_jax_layout(
        jax_module=jax_module,
        runtime=runtime,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
    )


def _raise_unsupported_route(route_key: tuple[str, str, str]) -> None:
    route = "/".join(route_key)
    raise UnsupportedBackendFeature(
        f"backend='jax' does not support route='{route}'. "
        f"Supported JAX routes in this build: {supported_jax_routes()!r}."
    )


def _unsupported_jax_layout(route_key: tuple[str, str, str]) -> OperatorLayout:
    message = (
        f"backend='jax' route='{('/'.join(route_key))}' has a runtime shell, "
        "but JAX residual execution is not implemented until residual parity."
    )

    def raise_profile(_: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    def raise_stage() -> None:
        raise UnsupportedBackendFeature(message)

    def raise_source() -> tuple[float, float]:
        raise UnsupportedBackendFeature(message)

    def raise_residual(_: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    def raise_fused(_: np.ndarray, __: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    return OperatorLayout.from_callables(
        profile_stage_runner=raise_profile,
        geometry_stage_runner=raise_stage,
        source_stage_runner=raise_source,
        residual_full_stage_runner_into=raise_residual,
        fused_residual_runner_into=raise_fused,
        collocation_runner_into=raise_fused,
    )


def _pf_rho_grid_jax_layout(
    *,
    jax_module,
    runtime,
    profile_workspace: ProfileWorkspace,
    geometry_workspace: GeometryWorkspace,
    source_workspace: SourceWorkspace,
    residual_workspace: ResidualWorkspace,
) -> OperatorLayout:
    message = (
        "backend='jax' supports fused residual_var/residual_var_into for route='PF/rho/grid' "
        "only; staged and collocation public methods remain unsupported until explicit "
        "host publication contracts are added."
    )

    def raise_profile(_: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    def raise_stage() -> None:
        raise UnsupportedBackendFeature(message)

    def raise_source() -> tuple[float, float]:
        raise UnsupportedBackendFeature(message)

    def raise_residual(_: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    def fused(x_eval: np.ndarray, out: np.ndarray) -> None:
        residual_var_numpy_bridge(
            jax_module=jax_module,
            runtime=runtime,
            x=x_eval,
            out=out,
            profile_workspace=profile_workspace,
            geometry_workspace=geometry_workspace,
            source_workspace=source_workspace,
            residual_workspace=residual_workspace,
        )

    def collocation(_: np.ndarray, __: np.ndarray) -> None:
        raise UnsupportedBackendFeature(message)

    return OperatorLayout.from_callables(
        profile_stage_runner=raise_profile,
        geometry_stage_runner=raise_stage,
        source_stage_runner=raise_source,
        residual_full_stage_runner_into=raise_residual,
        fused_residual_runner_into=fused,
        collocation_runner_into=collocation,
    )
