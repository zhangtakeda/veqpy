"""
Module: numba_core.source_runtime

Role:
- Refresh source runtime memory from a bound ``SourcePlan`` and current psin state.
- Own source input materialization cache updates outside engine kernels.

Notes:
- This module mutates preallocated source runtime arrays in place.
- It does not choose routes, allocate runtime state, or bind source stage runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine import (
    build_source_remap_cache,
    resolve_source_inputs,
)
from veqpy.model.numerics import build_uniform_source_interpolation_coefficients

from .source_plan import SourcePlan

if TYPE_CHECKING:
    from veqpy.workspace import SourceWorkspace


def refresh_source_runtime(
    *,
    grid_rho: np.ndarray,
    source_plan: SourcePlan,
    source_execution: object,
    source_workspace: SourceWorkspace,
    psin: np.ndarray,
) -> None:
    """Refresh source interpolation caches and materialized source arrays."""
    case_key = (
        source_plan.coordinate,
        source_plan.nodes,
        source_plan.source_sample_count,
        source_plan.interpolation_kind if not source_plan.is_grid_nodes else "",
    )
    if source_workspace.cache_key != case_key:
        # Cache identity is tied to interpolation topology, not the numeric
        # source samples.  New heat/current values reuse the same matrices and
        # only rebuild spline coefficients below.
        if source_plan.is_grid_nodes:
            source_workspace.barycentric_weights = np.empty(0, dtype=np.float64)
            source_workspace.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_workspace.heat_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_workspace.current_spline_coeff = np.empty((0, 4), dtype=np.float64)
        else:
            (
                _,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
            ) = build_source_remap_cache(
                source_plan.coordinate,
                source_plan.source_sample_count,
                rho=grid_rho,
                interpolation_kind=source_plan.interpolation_kind,
            )
        source_workspace.cache_key = case_key
    if source_plan.is_grid_nodes:
        # Grid-node inputs already live on the operator rho grid; spline slots are
        # deliberately empty so later code cannot accidentally interpolate them.
        source_workspace.heat_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.current_spline_coeff = np.empty((0, 4), dtype=np.float64)
    else:
        # Coefficients depend on source values even when the remap cache does not.
        source_workspace.heat_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.scaled_heat,
            kind=source_plan.interpolation_kind,
        )
        source_workspace.current_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.scaled_current,
            kind=source_plan.interpolation_kind,
        )
    if source_plan.is_grid_nodes or not source_plan.is_psin_coordinate:
        if source_plan.is_grid_nodes:
            np.copyto(source_workspace.materialized_heat_input, source_plan.scaled_heat)
            np.copyto(source_workspace.materialized_current_input, source_plan.scaled_current)
        else:
            # Rho-coordinate uniform samples can be materialized during refresh
            # because the query is fixed by grid_rho.
            resolve_source_inputs(
                source_workspace.materialized_heat_input,
                source_workspace.materialized_current_input,
                source_plan.scaled_heat,
                source_plan.scaled_current,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
                source_workspace.heat_spline_coeff,
                source_workspace.current_spline_coeff,
                psin,
                source_plan.uses_barycentric_interpolation,
            )
    elif tuple(source_execution.route_key) == ("PJ2", "psin", "uniform"):
        # PJ2/psin/uniform updates psin by fixed point during the source stage;
        # the negative sentinel forces a fresh query seed on the next run.
        source_workspace.psin_query.fill(-1.0)
