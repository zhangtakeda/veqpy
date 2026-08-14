"""
Module: veqpy.kernels.numba_kernel.workspace.source_workspace

Role:
- Own source-stage runtime memory, caches, and scratch arrays.

Public API:
- SourceWorkspace

Notes:
- Source planning and validation live in shared Kernel source semantics and
  backend-local runtime modules.
- Source numerical kernels live in the Numba backend package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.backend_abi import SourceExecutionABI


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
    pprime_spline_coeff: np.ndarray
    driver_spline_coeff: np.ndarray
    pressure_spline_coeff: np.ndarray
    source_coordinate_nodes: np.ndarray
    source_coordinate_weights: np.ndarray

    psin_query: np.ndarray
    parameter_query: np.ndarray
    materialized_pprime_input: np.ndarray
    pressure_derivative_work: np.ndarray
    driver_derivative_work: np.ndarray
    materialized_driver_input: np.ndarray
    materialized_driver_derivative: np.ndarray
    rho_query: np.ndarray
    rho_query_next: np.ndarray
    rho_derivative: np.ndarray
    rho_derivative_next: np.ndarray
    rho_pprime: np.ndarray
    rho_driver: np.ndarray
    rho_f: np.ndarray
    rho_f2: np.ndarray
    rho_u: np.ndarray
    rho_u_next: np.ndarray
    rho_current: np.ndarray
    rho_current_next: np.ndarray
    rho_state: np.ndarray
    array_scratch: np.ndarray
    matrix_scratch: np.ndarray

    target_root_fields: np.ndarray
    alpha_state: np.ndarray
    pressure_state: np.ndarray

    def __init__(self, *, nr: int, nt: int, source_execution: SourceExecutionABI) -> None:
        """Allocate source-stage runtime memory."""

        needs_psin_query = bool(source_execution.requires_psin_query_workspace)
        self.cache_key = None

        self.barycentric_weights = np.empty(0, dtype=np.float64)
        self.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
        self.pprime_spline_coeff = np.empty((0, 4), dtype=np.float64)
        self.driver_spline_coeff = np.empty((0, 4), dtype=np.float64)
        self.pressure_spline_coeff = np.empty((0, 4), dtype=np.float64)
        self.source_coordinate_nodes = np.empty(0, dtype=np.float64)
        self.source_coordinate_weights = np.empty(0, dtype=np.float64)

        self.psin_query = (
            np.empty(nr, dtype=np.float64) if needs_psin_query else np.empty(0, dtype=np.float64)
        )
        self.parameter_query = (
            np.empty(nr, dtype=np.float64)
            if source_execution.requires_source_parameter_query
            else self.psin_query
        )
        self.materialized_pprime_input = np.empty(nr, dtype=np.float64)
        # Preserve the public derivative while a psin route temporarily lowers
        # P_psin to the conventional P_psi consumed by legacy route kernels.
        self.pressure_derivative_work = np.empty(nr, dtype=np.float64)
        # PF/psin lowers FF_psin to the conventional FF_psi with the same
        # alpha2 iterate used for P_psin -> P_psi.
        self.driver_derivative_work = np.empty(nr, dtype=np.float64)
        self.materialized_driver_input = np.empty(nr, dtype=np.float64)
        self.materialized_driver_derivative = (
            np.empty(nr, dtype=np.float64)
            if source_execution.route_key in {
                ("PP", "r", "explicit"),
                ("PI", "r", "explicit"),
            }
            else np.empty(0, dtype=np.float64)
        )
        if source_execution.requires_rho_closure:
            self.rho_query = np.empty(nr, dtype=np.float64)
            self.rho_query_next = np.empty(nr, dtype=np.float64)
            self.rho_derivative = np.empty(nr, dtype=np.float64)
            self.rho_derivative_next = np.empty(nr, dtype=np.float64)
            self.rho_pprime = np.empty(nr, dtype=np.float64)
            self.rho_driver = np.empty(nr, dtype=np.float64)
            self.rho_f = np.empty(nr, dtype=np.float64)
            self.rho_f2 = np.empty(nr, dtype=np.float64)
            self.rho_u = np.empty(nr, dtype=np.float64)
            self.rho_u_next = np.empty(nr, dtype=np.float64)
            self.rho_current = np.empty(nr, dtype=np.float64)
            self.rho_current_next = np.empty(nr, dtype=np.float64)
            # [iterations, combined, coordinate value, coordinate derivative,
            #  PJ2/PJ3 strict-physics defect]
            self.rho_state = np.zeros(5, dtype=np.float64)
        else:
            empty = np.empty(0, dtype=np.float64)
            self.rho_query = empty
            self.rho_query_next = empty
            self.rho_derivative = empty
            self.rho_derivative_next = empty
            self.rho_pprime = empty
            self.rho_driver = empty
            self.rho_f = empty
            self.rho_f2 = empty
            self.rho_u = empty
            self.rho_u_next = empty
            self.rho_current = empty
            self.rho_current_next = empty
            self.rho_state = np.empty(0, dtype=np.float64)
        # The extra ``nr`` rows after the named scratch slots are reserved for
        # route-local dense systems such as strict PQ solves.
        self.array_scratch = np.empty((8 + nr, nr), dtype=np.float64)
        self.matrix_scratch = np.empty((1, nr, nt), dtype=np.float64)

        self.target_root_fields = (
            np.empty((3, nr), dtype=np.float64)
            if source_execution.requires_target_root_fields
            else np.empty((3, 0), dtype=np.float64)
        )
        self.alpha_state = np.zeros(2, dtype=np.float64)
        # [0] is the effective mu0*p0 after any beta scaling; [1] is the
        # common pressure-profile multiplier applied to pprime and p0.
        self.pressure_state = np.array([0.0, 1.0], dtype=np.float64)
