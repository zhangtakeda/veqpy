from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks._common import (
    ROUTE_BENCHMARK_CONSTRAINTS,
    RouteBenchmarkSpec,
    route_kernel_case,
)
from veqpy import Kernel, KernelRecipe, KernelSource, KernelTopology
from veqpy.kernels.abi.source_semantics import MU0, materialize_kernel_source
from veqpy.kernels.numba_kernel import numba_source
from veqpy.numerics import make_quadrature

ROUTE_CONSTRAINT_CASES = tuple(
    (route, constraint)
    for route, constraints in ROUTE_BENCHMARK_CONSTRAINTS.items()
    for constraint in constraints
)


@pytest.mark.parametrize(("route", "constraint"), ROUTE_CONSTRAINT_CASES)
def test_rho_uniform_routes_close_every_legal_constraint(
    route: str,
    constraint: str,
) -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec(route, "rho", "uniform", constraint),
        nr=16,
        nt=8,
        sample_count=25,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        equilibrium = kernel.build_equilibrium()
        workspace = kernel._impl._solver.runtime.source_workspace

        assert result.success
        assert 1 <= workspace.rho_state[0] <= 16
        assert workspace.rho_state[1] <= 1.0e-6
        assert_allclose(
            workspace.rho_query_next,
            equilibrium.rho,
            rtol=0.0,
            # The model snapshot reconstructs toroidal flux through its own
            # formula path.  Keep this gate tighter than the 1e-6 local
            # closure tolerance without requiring identical summation order.
            atol=1.0e-7,
        )
        assert_allclose(
            workspace.rho_derivative_next,
            equilibrium.rho_r,
            rtol=3.0e-7,
            atol=1.0e-8,
        )
    finally:
        kernel.close()


@pytest.mark.parametrize("route", tuple(ROUTE_BENCHMARK_CONSTRAINTS))
def test_rho_grid_routes_close_on_coordinate_gauss_nodes(route: str) -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec(route, "rho", "grid", "ip"),
        nr=16,
        nt=8,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        workspace = kernel._impl._solver.runtime.source_workspace
        nodes, _ = make_quadrature(case.topology.Nr, scheme=case.topology.quadrature)

        assert result.success
        assert workspace.rho_state[1] <= 1.0e-6
        assert_array_equal(workspace.source_coordinate_nodes, nodes)
    finally:
        kernel.close()


@pytest.mark.parametrize("route", tuple(ROUTE_BENCHMARK_CONSTRAINTS))
def test_rho_and_r_routes_recover_the_same_physical_solution(route: str) -> None:
    equilibria = []
    for coordinate in ("r", "rho"):
        case = route_kernel_case(
            RouteBenchmarkSpec(route, coordinate, "uniform", "ip"),
            nr=16,
            nt=8,
            sample_count=25,
        )
        kernel = Kernel(
            topology=case.topology,
            recipe=KernelRecipe(backend="numba"),
            config=case.config,
        )
        try:
            result = kernel.solve(case.boundary, case.source)
            assert result.success
            equilibria.append(kernel.build_equilibrium())
        finally:
            kernel.close()

    r_equilibrium, rho_equilibrium = equilibria
    for name in ("psin", "F", "P"):
        reference = np.asarray(getattr(r_equilibrium, name), dtype=np.float64)
        current = np.asarray(getattr(rho_equilibrium, name), dtype=np.float64)
        scale = max(float(np.max(np.abs(reference))), 1.0e-14)
        assert np.max(np.abs(current - reference)) / scale <= 3.0e-3
    for name, tolerance in (("q", 2.0e-3), ("jtor", 2.0e-2)):
        reference = np.asarray(getattr(r_equilibrium, name), dtype=np.float64)
        current = np.asarray(getattr(rho_equilibrium, name), dtype=np.float64)
        scale = max(float(np.max(np.abs(reference))), 1.0e-14)
        assert np.max(np.abs(current - reference)) / scale <= tolerance


def test_rho_residual_is_independent_of_evaluation_history() -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec("PJ3", "rho", "uniform", "ip"),
        nr=16,
        nt=8,
        sample_count=25,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        first = kernel.residual(result.x, case.boundary, case.source)
        first_state = kernel._impl._solver.runtime.source_workspace.rho_state.copy()
        kernel.residual(np.zeros_like(result.x), case.boundary, case.source)
        second = kernel.residual(result.x.copy(), case.boundary, case.source)
        second_state = kernel._impl._solver.runtime.source_workspace.rho_state.copy()

        assert result.success
        assert_array_equal(second, first)
        assert_array_equal(second_state, first_state)
    finally:
        kernel.close()


@pytest.mark.parametrize(("route", "expected_max_iterations"), (("PJ2", 7), ("PJ3", 9)))
def test_rho_pj23_reports_one_joint_coordinate_and_physics_defect(
    route: str,
    expected_max_iterations: int,
) -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec(route, "rho", "uniform", "ip"),
        nr=24,
        nt=12,
        sample_count=39,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        state = kernel._impl._solver.runtime.source_workspace.rho_state

        assert result.success
        assert state.shape == (5,)
        assert 1 <= state[0] <= expected_max_iterations
        assert state[4] > 0.0
        assert state[1] == max(state[2], state[3], state[4])
        assert state[1] <= 1.0e-6
    finally:
        kernel.close()


def test_rho_nonconvergence_is_an_explicit_source_stage_failure(monkeypatch) -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec("PJ1", "rho", "uniform", "ip"),
        nr=16,
        nt=8,
        sample_count=25,
    )
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        monkeypatch.setattr(numba_source, "RHO_FIXED_POINT_MAX_ITER", 1)
        with pytest.raises(ValueError, match=r"did not reach tolerance within iteration limit"):
            kernel.residual(
                np.zeros(case.topology.x_size, dtype=np.float64),
                case.boundary,
                case.source,
            )
    finally:
        kernel.close()


@pytest.mark.parametrize("nodes", ("uniform", "grid"))
def test_absolute_pressure_is_differentiated_in_rho(nodes: str) -> None:
    nr = 8
    sample_count = nr if nodes == "grid" else 9
    topology = KernelTopology(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=0,
        F_count=0,
        c_counts=(),
        s_counts=(1,),
        Nr=nr,
        Nt=8,
        route="PQ",
        coordinate="rho",
        nodes=nodes,
        constraint="none",
        sample_count=sample_count,
    )
    coordinate = (
        make_quadrature(nr, scheme=topology.quadrature)[0]
        if nodes == "grid"
        else np.linspace(0.0, 1.0, sample_count)
    )
    edge_pressure = 6408.706536
    amplitude = 2200.0
    pressure = edge_pressure + amplitude * (1.0 - coordinate * coordinate)
    source = KernelSource(
        p=pressure,
        q=1.71 + 0.16 * coordinate * coordinate,
    )

    materialized = materialize_kernel_source(topology, source)

    assert_allclose(
        materialized.scaled_pprime / MU0,
        -2.0 * amplitude * coordinate,
        rtol=2.0e-13,
        atol=1.0e-8,
    )
    assert materialized.scaled_p0 / MU0 == pytest.approx(edge_pressure, abs=1.0e-8)


@pytest.mark.parametrize(("field", "value"), (("psin_count", 1), ("F_count", 1)))
def test_rho_topology_owns_no_outer_coordinate_profile(
    field: str,
    value: int,
) -> None:
    kwargs = {"psin_count": 0, "F_count": 0, field: value}
    with pytest.raises(ValueError, match=field):
        KernelTopology(
            h_count=1,
            v_count=0,
            kappa_count=0,
            c_counts=(),
            s_counts=(1,),
            Nr=8,
            Nt=8,
            route="PJ2",
            coordinate="rho",
            nodes="uniform",
            constraint="ip",
            sample_count=9,
            **kwargs,
        )


def test_rho_coordinate_rejects_the_unimplemented_cxx_backend() -> None:
    case = route_kernel_case(
        RouteBenchmarkSpec("PJ1", "rho", "uniform", "ip"),
        nr=8,
        nt=8,
        sample_count=9,
    )
    with pytest.raises(NotImplementedError, match="supported only by backend='numba'"):
        Kernel(topology=case.topology, recipe=KernelRecipe(backend="cxx"))
