from __future__ import annotations

import numpy as np
import pytest
from helpers import MU0, pf_reference_profiles
from numpy.linalg import norm
from numpy.testing import assert_allclose

from veqlib.facade import KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqlib.facade.source_semantics import materialize_kernel_source
from veqpy.engine import validate_route
from veqpy.kernel import NumbaKernel
from veqpy.kernel.result import solve_result_from_legacy
from veqpy.model import Boundary, Grid, Problem
from veqpy.operator import Operator
from veqpy.operator.source_plan import build_source_plan
from veqpy.solver import SolverResult
from veqpy.solver.residual_scale import make_residual_scale

ROUTE_PARITY_CASES = (
    ("PF", "psin", "uniform"),
    ("PP", "psin", "uniform"),
    ("PI", "rho", "uniform"),
    ("PJ1", "psin", "uniform"),
    ("PJ2", "psin", "uniform"),
    ("PQ", "rho", "grid"),
)


def make_kernel_topology(**overrides: object) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 3,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
        "Nr": 8,
        "Nt": 8,
        "route": "PF",
        "coordinate": "psin",
        "nodes": "uniform",
        "ip_constraint": True,
        "sample_count": 9,
    }
    params.update(overrides)
    return KernelTopology(**params)  # type: ignore[arg-type]


def tiny_kernel_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
    )


def tiny_kernel_source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    current_profile, scaled_heat = pf_reference_profiles(psin)
    return KernelSource(
        heat_profile=scaled_heat / MU0,
        current_profile=current_profile,
        Ip=3.0e6,
    )


def route_kernel_topology(route: str, coordinate: str, nodes: str) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 0,
        "psin_count": 0,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
        "Nr": 8,
        "Nt": 8,
        "route": route,
        "coordinate": coordinate,
        "nodes": nodes,
        "ip_constraint": True,
        "sample_count": 8 if nodes == "grid" else 9,
    }
    if route == "PJ2":
        params["F_count"] = 2
    elif coordinate == "psin" and nodes == "uniform":
        params["psin_count"] = 2
    return KernelTopology(**params)  # type: ignore[arg-type]


def route_kernel_source(route: str, sample_count: int) -> KernelSource:
    heat_profile = np.linspace(1.0e6, 1.4e6, sample_count, dtype=np.float64)
    if route in {"PI", "PJ1", "PJ2"}:
        current_profile = np.linspace(1.0e6, 3.0e6, sample_count, dtype=np.float64)
    else:
        current_profile = np.linspace(1.0, 3.0, sample_count, dtype=np.float64)
    return KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=3.0e6,
    )


def tiny_legacy_operator(topology: KernelTopology) -> Operator:
    source = tiny_kernel_source()
    boundary = tiny_kernel_boundary()
    return legacy_operator_from_kernel_case(topology, boundary, source)


def legacy_operator_from_kernel_case(
    topology: KernelTopology,
    boundary: KernelBoundary,
    source: KernelSource,
) -> Operator:
    return Operator(
        Grid(
            Nr=topology.Nr,
            Nt=topology.Nt,
            L_max=topology.L_max,
            M_max=topology.M_max,
            K_max=topology.K_max,
            quadrature_scheme=topology.quadrature,
            calculus_scheme=topology.calculus,
        ),
        Problem(
            route=topology.route,
            coordinate=topology.coordinate,
            nodes=topology.nodes,
            active_profiles=dict(topology.active_profiles),
            boundary=legacy_boundary_from_kernel(boundary),
            heat_input=source.heat_profile,
            current_input=source.current_profile,
            Ip=source.Ip,
            beta=source.beta,
        ),
    )


def legacy_problem_from_kernel_case(
    topology: KernelTopology,
    boundary: KernelBoundary,
    source: KernelSource,
) -> Problem:
    return Problem(
        route=topology.route,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        active_profiles=dict(topology.active_profiles),
        boundary=legacy_boundary_from_kernel(boundary),
        heat_input=source.heat_profile,
        current_input=source.current_profile,
        Ip=source.Ip,
        beta=source.beta,
    )


def legacy_boundary_from_kernel(boundary: KernelBoundary) -> Boundary:
    return Boundary(
        a=boundary.a,
        R0=boundary.R0,
        Z0=boundary.Z0,
        B0=boundary.B0,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )


def test_numba_kernel_recipe_validation_and_public_surface() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)

    assert kernel.x_size == topology.x_size
    assert kernel.recipe.backend == "numba"
    assert kernel.recipe.layout == "degree"
    assert kernel.prepare() is None
    assert kernel.prepare(force=True, dry_run=True) is None
    assert kernel.close() is None
    assert kernel.close() is None

    with pytest.raises(ValueError, match="layout='degree'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="numba", layout="family"))
    with pytest.raises(ValueError, match="backend='numba'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="cxx", layout="degree"))


def test_numba_kernel_residual_matches_legacy_operator_and_validates_buffers() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    legacy_operator = tiny_legacy_operator(topology)
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual = kernel.residual(x, boundary, source)
    assert_allclose(residual, legacy_operator.residual_var(x))

    out = np.empty(kernel.x_size, dtype=np.float64)
    kernel.residual_into(out, x.tolist(), boundary, source)
    assert_allclose(out, residual)

    with pytest.raises(ValueError, match="x must have shape"):
        kernel.residual(np.zeros(kernel.x_size + 1, dtype=np.float64), boundary, source)
    with pytest.raises(TypeError, match="dtype float64"):
        kernel.residual_into(np.empty(kernel.x_size, dtype=np.float32), x, boundary, source)
    with pytest.raises(ValueError, match="C-contiguous"):
        noncontiguous_out = np.empty((kernel.x_size, 2), dtype=np.float64)[:, 0]
        kernel.residual_into(noncontiguous_out, x, boundary, source)


@pytest.mark.parametrize(("route", "coordinate", "nodes"), ROUTE_PARITY_CASES)
def test_kernel_source_materialization_matches_legacy_source_plan_route_matrix(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = route_kernel_topology(route, coordinate, nodes)
    boundary = tiny_kernel_boundary()
    source = route_kernel_source(route, topology.sample_count)
    problem = legacy_problem_from_kernel_case(topology, boundary, source)
    source_route_spec = validate_route(problem.route, problem.coordinate, problem.nodes)

    source_plan = build_source_plan(problem=problem, source_route_spec=source_route_spec)
    materialized = materialize_kernel_source(topology, source)

    assert source_plan.route_key == topology.source_route_key
    assert source_plan.parameterization == topology.source_parameterization
    assert_allclose(source_plan.scaled_heat, materialized.scaled_heat)
    assert_allclose(source_plan.scaled_current, materialized.scaled_current)
    assert source_plan.scaled_Ip == materialized.scaled_Ip
    assert_allclose([source_plan.beta], [materialized.beta], equal_nan=True)
    assert not source_plan.scaled_heat.flags.writeable
    assert not source_plan.scaled_current.flags.writeable


@pytest.mark.parametrize(("route", "coordinate", "nodes"), ROUTE_PARITY_CASES)
def test_numba_kernel_residual_matches_legacy_operator_for_route_matrix(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = route_kernel_topology(route, coordinate, nodes)
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = route_kernel_source(route, topology.sample_count)
    legacy_operator = legacy_operator_from_kernel_case(topology, boundary, source)
    x = np.zeros(kernel.x_size, dtype=np.float64)

    assert_allclose(kernel.residual(x, boundary, source), legacy_operator.residual_var(x))


def test_numba_kernel_build_equilibrium_runtime_state_rules() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)

    with pytest.raises(RuntimeError, match="previous NumbaKernel runtime case"):
        kernel.build_equilibrium(x)

    kernel.residual(x, boundary, source)
    equilibrium = kernel.build_equilibrium(x)
    assert np.isfinite(equilibrium.Ip)

    with pytest.raises(RuntimeError, match="previous solve result"):
        kernel.build_equilibrium()


def test_numba_kernel_solve_result_lifecycle_and_equilibrium_snapshot() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()

    result = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_evaluations=2,
        ),
    )

    assert result.x.shape == (kernel.x_size,)
    assert result.raw.shape == (kernel.x_size,)
    assert result.scaled.shape == (kernel.x_size,)
    assert result.alpha.shape == (2,)
    assert kernel.result is result
    assert kernel.history == [result]
    assert_allclose(kernel.build_equilibrium(result.x).Ip, kernel.build_equilibrium().Ip)
    assert result.info == int(result.success)
    assert result.callbacks == 0
    assert result.jvp_evaluations == 0
    assert result.jacobian_component_evaluations == 0

    kernel.clear()
    assert kernel.result is None
    assert kernel.history == []


@pytest.mark.parametrize("norm_mode", ["none", "fast", "balanced"])
def test_numba_kernel_solve_result_scaled_uses_solver_reference_state(norm_mode: str) -> None:
    operator = tiny_legacy_operator(make_kernel_topology())
    x0 = np.zeros(operator.x_size, dtype=np.float64)
    x_final = np.ones(operator.x_size, dtype=np.float64)
    solver_result = SolverResult(
        x0=x0,
        x=x_final,
        success=False,
        message="synthetic terminal state",
        residual_norm_final=0.0,
        function_evaluations=3,
        jacobian_evaluations=1,
        iterations=2,
        elapsed=1250.0,
    )
    config = KernelConfig(norm=norm_mode)

    result = solve_result_from_legacy(solver_result, operator, config)
    expected_raw = operator.residual_var(x_final)
    expected_scaled = _expected_scaled_from_reference(expected_raw, x0, operator, config)

    assert_allclose(result.raw, expected_raw)
    assert_allclose(result.scaled, expected_scaled)
    assert result.elapsed_ms == 1.25
    assert result.info == 0
    assert result.nfev == 3
    assert result.njev == 1
    assert result.linear_iterations == 2
    assert result.callbacks == 0
    assert result.jvp_evaluations == 0
    assert result.jacobian_component_evaluations == 0
    assert result.raw_norm == pytest.approx(float(norm(result.raw)))
    assert result.scaled_norm == pytest.approx(float(norm(result.scaled)))

    if norm_mode != "none":
        final_based_scaled = _expected_scaled_from_reference(
            expected_raw,
            x_final,
            operator,
            config,
        )
        assert not np.allclose(result.scaled, final_based_scaled)


def test_numba_kernel_warm_continuation_passes_previous_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    first = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_evaluations=2,
        ),
    )
    captured: dict[str, np.ndarray | None] = {}

    def fake_solve(
        boundary_arg: KernelBoundary,
        source_arg: KernelSource,
        config_arg: KernelConfig,
        *,
        x0: np.ndarray | None,
    ):
        del boundary_arg, source_arg, config_arg
        captured["x0"] = None if x0 is None else x0.copy()
        return first

    monkeypatch.setattr(kernel._solver, "solve", fake_solve)
    second = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            continuation="warm",
            norm="none",
            max_evaluations=2,
        ),
    )

    assert second is first
    assert captured["x0"] is not None
    assert_allclose(captured["x0"], first.x)


def test_numba_kernel_jvp_and_jacobian_are_explicitly_unimplemented() -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)

    with pytest.raises(NotImplementedError):
        kernel.jvp(x, x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jvp_into(np.empty_like(x), x, x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jacobian(x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jacobian_into(np.empty((kernel.x_size, kernel.x_size)), x, boundary, source)


def _expected_scaled_from_reference(
    raw: np.ndarray,
    x_reference: np.ndarray,
    operator: Operator,
    config: KernelConfig,
) -> np.ndarray:
    if config.norm == "none":
        return raw.copy()
    reference_raw = operator.residual_var(x_reference)
    params: dict[str, object] = {}
    if config.norm == "balanced":
        params = {
            "floor": config.residual_normalization_floor,
            "max_ratio": config.residual_normalization_max_ratio,
            "huber_tau": config.residual_normalization_huber_tau,
        }
    scale = make_residual_scale(
        config.norm,
        reference_raw,
        operator.residual_block_lengths(),
        **params,
    )
    if scale is None:
        scale = np.ones_like(raw)
    return raw / scale
