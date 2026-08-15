"""
Module: veqpy.kernels.numba_kernel.source_runtime

Role:
- Refresh source runtime memory from a bound ``SourcePlan`` and current psin state.
- Own source input materialization cache updates outside residual kernels.

Notes:
- This module mutates preallocated source runtime arrays in place.
- It does not choose routes, allocate runtime state, or bind source stage runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.numerics import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    barycentric_log_weights,
    build_uniform_source_interpolation_coefficients,
)
from veqpy.numerics.interpolate import uniform_barycentric_weights

from .numba_source import (
    _explicit_local_barycentric_interpolate_pair_with_derivatives,
    build_source_remap_cache,
    resolve_source_inputs,
)
from .source_plan import SourcePlan

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.workspace import SourceWorkspace


def refresh_source_runtime(
    *,
    grid_r: np.ndarray,
    source_plan: SourcePlan,
    source_execution: object,
    source_workspace: SourceWorkspace,
    psin: np.ndarray,
) -> None:
    """Refresh source interpolation caches and materialized source arrays."""
    if source_plan.is_explicit_nodes:
        if source_plan.source_nodes is None:
            raise ValueError("explicit source plan is missing retained source_nodes")
        source_workspace.source_coordinate_nodes = np.array(
            source_plan.source_nodes,
            dtype=np.float64,
            copy=True,
        )
        stencil_size = min(
            int(source_workspace.source_coordinate_nodes.size),
            DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
        )
        local_weights = uniform_barycentric_weights(stencil_size)
        source_span = float(
            source_workspace.source_coordinate_nodes[-1]
            - source_workspace.source_coordinate_nodes[0]
        )
        uniform_tolerance = 1.0e-12 * max(
            1.0,
            abs(source_span),
        )
        expected_nodes = source_workspace.source_coordinate_nodes[0] + source_span * (
            np.arange(source_workspace.source_coordinate_nodes.size, dtype=np.float64)
            / float(source_workspace.source_coordinate_nodes.size - 1)
        )
        source_workspace.source_coordinate_weights = (
            local_weights.copy()
            if np.all(
                np.abs(source_workspace.source_coordinate_nodes - expected_nodes)
                <= uniform_tolerance
            )
            else np.empty(0, dtype=np.float64)
        )
        source_workspace.barycentric_weights = local_weights
        source_workspace.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
        source_workspace.pprime_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.driver_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.pressure_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.cache_key = (
            source_plan.coordinate,
            source_plan.nodes,
            source_plan.source_sample_count,
            "local-barycentric",
        )
        if source_plan.is_rho_coordinate:
            return
        if source_plan.coordinate == "r":
            values0 = (
                source_plan.scaled_pressure
                if source_plan.scaled_pressure is not None
                else source_plan.scaled_pprime
            )
            driver_derivative = (
                source_workspace.materialized_driver_derivative
                if source_plan.route in {"PP", "PI"}
                else source_workspace.materialized_driver_input
            )
            _explicit_local_barycentric_interpolate_pair_with_derivatives(
                source_workspace.materialized_pprime_input,
                source_workspace.materialized_driver_input,
                source_workspace.materialized_pprime_input,
                driver_derivative,
                values0,
                source_plan.scaled_driver,
                source_workspace.source_coordinate_nodes,
                source_workspace.source_coordinate_weights,
                grid_r,
                source_plan.scaled_pressure is not None,
                source_plan.route in {"PP", "PI"},
            )
            return
        if source_plan.route in {"PJ2", "PJ3"}:
            source_workspace.psin_query.fill(-1.0)
        return

    case_key = (
        source_plan.coordinate,
        source_plan.nodes,
        source_plan.source_sample_count,
        source_plan.interpolation_kind if not source_plan.is_grid_nodes else "",
    )
    if source_workspace.cache_key != case_key:
        # Cache identity is tied to interpolation topology, not the numeric
        # source samples. New pprime/driver values reuse the same matrices and
        # only rebuild spline coefficients below.
        if source_plan.is_grid_nodes and source_plan.is_rho_coordinate:
            # For a native rho route, ``grid`` means samples at the same
            # Gauss node values interpreted in sqrt(Phi_N), not values already
            # materialized on geometric r. Keep a stable global barycentric
            # representation for dynamic queries inside every source closure.
            source_workspace.source_coordinate_nodes = np.array(grid_r, copy=True)
            signs, log_weights = barycentric_log_weights(source_workspace.source_coordinate_nodes)
            source_workspace.source_coordinate_weights = signs * np.exp(
                log_weights - np.max(log_weights)
            )
            source_workspace.barycentric_weights = np.empty(0, dtype=np.float64)
            source_workspace.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_workspace.pprime_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_workspace.driver_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_workspace.pressure_spline_coeff = np.empty((0, 4), dtype=np.float64)
        elif source_plan.is_grid_nodes:
            source_workspace.barycentric_weights = np.empty(0, dtype=np.float64)
            source_workspace.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_workspace.pprime_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_workspace.driver_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_workspace.pressure_spline_coeff = np.empty((0, 4), dtype=np.float64)
        else:
            (
                _,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
            ) = build_source_remap_cache(
                source_plan.coordinate,
                source_plan.source_sample_count,
                r=grid_r,
                interpolation_kind=source_plan.interpolation_kind,
            )
        source_workspace.cache_key = case_key
    if source_plan.is_grid_nodes:
        # Grid-node inputs already live on the operator r grid; spline slots are
        # deliberately empty so later code cannot accidentally interpolate them.
        source_workspace.pprime_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.driver_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_workspace.pressure_spline_coeff = np.empty((0, 4), dtype=np.float64)
    else:
        # Coefficients depend on source values even when the remap cache does not.
        source_workspace.pprime_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.scaled_pprime,
            kind=source_plan.interpolation_kind,
        )
        source_workspace.driver_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.scaled_driver,
            kind=source_plan.interpolation_kind,
        )
    if source_plan.is_rho_coordinate:
        # Dynamic rho source inputs are materialized by the local closure
        # after geometry has been refreshed; there is no correct setup-time
        # query coordinate.
        pass
    elif source_plan.is_grid_nodes or not source_plan.is_psin_coordinate:
        if source_plan.is_grid_nodes:
            np.copyto(source_workspace.materialized_pprime_input, source_plan.scaled_pprime)
            np.copyto(source_workspace.materialized_driver_input, source_plan.scaled_driver)
        else:
            # r-coordinate uniform samples can be materialized during refresh
            # because the query is fixed by grid_r.
            resolve_source_inputs(
                source_workspace.materialized_pprime_input,
                source_workspace.materialized_driver_input,
                source_plan.scaled_pprime,
                source_plan.scaled_driver,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_workspace.barycentric_weights,
                source_workspace.fixed_remap_matrix,
                source_workspace.pprime_spline_coeff,
                source_workspace.driver_spline_coeff,
                psin,
                source_plan.uses_barycentric_interpolation,
            )
    elif (
        source_execution.route_key[0] in {"PJ2", "PJ3"}
        and source_execution.route_key[1] == "psin"
        and source_execution.route_key[2] in {"uniform", "explicit"}
    ):
        # F-coupled current routes update psin by fixed point during the source stage;
        # the negative sentinel forces a fresh query seed on the next run.
        source_workspace.psin_query.fill(-1.0)
