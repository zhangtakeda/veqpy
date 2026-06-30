"""
Module: workspace.source_workspace

Role:
- Own source-stage runtime memory, caches, and scratch arrays.

Public API:
- SourceWorkspace

Notes:
- Source planning and validation live in ``veqpy.operator.source_plan``.
- Source numerical kernels live in ``veqpy.engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.engine import SourceExecutionABI


@dataclass(init=False, slots=True)
class SourceWorkspace:
    """Source stage memory owner.

    The source stage owns route/interpolation cache arrays, materialization scratch,
    source-produced outputs, and source scale factors.  Profile inputs are supplied
    explicitly from ``ProfileWorkspace`` at binding time; this workspace does not
    borrow or cache profile views.

    ``array_scratch`` and ``matrix_scratch`` are reusable temporary work arrays
    allocated with the workspace, then reused by source kernels on the hot path.
    Scratch arrays do not carry persistent physical meaning across kernel calls;
    their slot meanings are owned by the consuming kernel and may be overwritten
    during one source-stage evaluation.
    """

    cache_key: tuple[str, str, int, str] | None

    barycentric_weights: np.ndarray
    fixed_remap_matrix: np.ndarray
    heat_spline_coeff: np.ndarray
    current_spline_coeff: np.ndarray

    psin_query: np.ndarray
    parameter_query: np.ndarray
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    array_scratch: np.ndarray
    matrix_scratch: np.ndarray

    target_root_fields: np.ndarray
    alpha_state: np.ndarray

    def __init__(self, *, nr: int, nt: int, source_execution: SourceExecutionABI) -> None:
        """Allocate source-stage runtime memory."""

        needs_psin_query = bool(source_execution.requires_psin_query_workspace)
        self.cache_key = None

        self.barycentric_weights = np.empty(0, dtype=np.float64)
        self.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
        self.heat_spline_coeff = np.empty((0, 4), dtype=np.float64)
        self.current_spline_coeff = np.empty((0, 4), dtype=np.float64)

        self.psin_query = (
            np.empty(nr, dtype=np.float64) if needs_psin_query else np.empty(0, dtype=np.float64)
        )
        self.parameter_query = (
            np.empty(nr, dtype=np.float64)
            if source_execution.requires_source_parameter_query
            else self.psin_query
        )
        self.materialized_heat_input = np.empty(nr, dtype=np.float64)
        self.materialized_current_input = np.empty(nr, dtype=np.float64)
        # The extra ``nr`` rows after the named scratch slots are reserved for
        # route-local dense systems such as strict PQ solves.
        self.array_scratch = np.empty((7 + nr, nr), dtype=np.float64)
        self.matrix_scratch = np.empty((1, nr, nt), dtype=np.float64)

        self.target_root_fields = (
            np.empty((3, nr), dtype=np.float64)
            if source_execution.requires_target_root_fields
            else np.empty((3, 0), dtype=np.float64)
        )
        self.alpha_state = np.zeros(2, dtype=np.float64)
