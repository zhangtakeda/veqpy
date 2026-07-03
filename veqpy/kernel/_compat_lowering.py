"""Temporary lowering from KernelTypes to the legacy VEQPy runtime.

KernelTopology is the source of truth. This lowering exists only to reuse legacy
VEQPy Operator/Solver during NumbaKernel bootstrap.
"""

from __future__ import annotations

from veqlib.facade import KernelBoundary, KernelSource, KernelTopology
from veqpy.model import Boundary, Grid, Problem
from veqpy.operator import Operator


def build_legacy_operator(
    topology: KernelTopology,
    boundary: KernelBoundary,
    source: KernelSource,
) -> Operator:
    """Build a one-case legacy Operator from Kernel runtime values."""

    grid = _legacy_grid_from_topology(topology)
    problem = _legacy_problem_from_case(topology, boundary, source)
    operator = Operator(grid, problem)
    if operator.x_size != topology.x_size:
        raise RuntimeError(
            "KernelTopology legacy lowering produced incompatible x_size: "
            f"operator={operator.x_size}, topology={topology.x_size}"
        )
    return operator


def _legacy_grid_from_topology(topology: KernelTopology) -> Grid:
    return Grid(
        Nr=topology.Nr,
        Nt=topology.Nt,
        L_max=topology.L_max,
        M_max=topology.M_max,
        K_max=topology.K_max,
        quadrature_scheme=topology.quadrature,
        calculus_scheme=topology.calculus,
    )


def _legacy_problem_from_case(
    topology: KernelTopology,
    boundary: KernelBoundary,
    source: KernelSource,
) -> Problem:
    return Problem(
        route=topology.route,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        active_profiles=_legacy_active_profiles_from_topology(topology),
        boundary=_legacy_boundary_from_kernel(boundary),
        heat_input=source.heat_profile,
        current_input=source.current_profile,
        Ip=source.Ip,
        beta=source.beta,
    )


def _legacy_active_profiles_from_topology(topology: KernelTopology) -> dict[str, int]:
    return dict(topology.active_profiles)


def _legacy_boundary_from_kernel(boundary: KernelBoundary) -> Boundary:
    return Boundary(
        a=boundary.a,
        R0=boundary.R0,
        Z0=boundary.Z0,
        B0=boundary.B0,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )
