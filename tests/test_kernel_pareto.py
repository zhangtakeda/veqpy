from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource
from veqpy.api import pareto as pareto_api
from veqpy.kernels.numba_kernel.pareto_runtime import sample_r_surface
from veqpy.kernels.pareto import (
    KernelParetoSignature,
    ParetoSample,
    adaptive_seed_candidate_count,
    coefficient_blocks_from_packed_state,
    generate_adaptive_refinement_signatures,
    generate_pareto_signatures,
    normalize_pareto_by,
    normalize_pareto_max_candidates,
    normalize_pareto_metric,
    normalize_pareto_strategy,
    normalize_shape_error_thresholds,
    pareto_frontier,
    pareto_sample_complexity,
    pareto_shape_error,
    select_pareto_thresholds,
    topology_from_pareto_signature,
)
from veqpy.kernels.types import KernelTopology, SolveResult


def make_topology(**overrides: object) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 1,
        "kappa_count": 1,
        "psin_count": 1,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (),
        "Nr": 8,
        "Nt": 8,
        "route": "PF",
        "coordinate": "psin",
        "nodes": "uniform",
        "ip_constraint": True,
        "sample_count": 8,
    }
    params.update(overrides)
    return KernelTopology(**params)  # type: ignore[arg-type]


def make_result(
    *,
    topology: KernelTopology,
    success: bool = True,
    nfev: int = 0,
    jvp_evaluations: int = 0,
    jacobian_component_evaluations: int = 0,
    linear_iterations: int = 0,
) -> SolveResult:
    x = np.zeros(topology.x_size, dtype=np.float64)
    return SolveResult(
        elapsed_ms=1.0,
        success=success,
        info=1 if success else 0,
        nfev=nfev,
        njev=0,
        callbacks=0,
        jacobian_component_evaluations=jacobian_component_evaluations,
        jvp_evaluations=jvp_evaluations,
        linear_iterations=linear_iterations,
        raw_norm=0.0,
        scaled_norm=0.0,
        x=x,
        raw=x.copy(),
        scaled=x.copy(),
        alpha=np.zeros(2, dtype=np.float64),
    )


def tiny_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=(float(np.arcsin(0.2)),),
    )


def tiny_source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    return KernelSource(
        heat_profile=1.0e6 + 0.2e6 * psin,
        current_profile=1.0 + 0.1 * psin,
        Ip=3.0e6,
    )


def make_sample(
    *,
    topology: KernelTopology,
    counts: int,
    time: float,
    complexity: int,
    shape_error: float,
    success: bool = True,
) -> ParetoSample:
    return ParetoSample(
        topology=topology,
        signature=KernelParetoSignature.from_topology(topology),
        counts=counts,
        time=time,
        complexity=complexity,
        shape_error=shape_error,
        result=make_result(topology=topology, success=success),
    )


def test_pareto_parameter_normalization() -> None:
    assert normalize_pareto_by("COUNTS") == "counts"
    assert normalize_pareto_metric("Max") == "max"
    assert normalize_pareto_strategy(" adaptive ") == "adaptive"
    assert normalize_pareto_max_candidates(0) == 0
    assert normalize_pareto_max_candidates(500) == 500
    assert normalize_shape_error_thresholds(None) == ()
    assert normalize_shape_error_thresholds(1.0e-3) == (1.0e-3,)
    assert normalize_shape_error_thresholds([1.0e-3, 2.0e-3]) == (1.0e-3, 2.0e-3)

    with pytest.raises(ValueError, match="pareto_by"):
        normalize_pareto_by("elapsed")
    with pytest.raises(ValueError, match="metric"):
        normalize_pareto_metric("mean")
    with pytest.raises(ValueError, match="strategy"):
        normalize_pareto_strategy("random")
    with pytest.raises(TypeError, match="max_candidates"):
        normalize_pareto_max_candidates(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        normalize_pareto_max_candidates(-1)
    with pytest.raises(ValueError, match="finite"):
        normalize_shape_error_thresholds(float("inf"))
    with pytest.raises(ValueError, match="non-negative"):
        normalize_shape_error_thresholds([-1.0e-3])


def test_pareto_sample_complexity_uses_fixed_formula() -> None:
    topology = make_topology(h_count=1, v_count=1, kappa_count=1, psin_count=1)
    result = make_result(
        topology=topology,
        nfev=3,
        jvp_evaluations=2,
        jacobian_component_evaluations=1,
        linear_iterations=4,
    )

    nx = topology.x_size
    expected = 3 * nx + (2 + 1 + 4) * nx * nx
    complexity = pareto_sample_complexity(result, topology)
    assert isinstance(complexity, int)
    assert complexity == expected


def test_pareto_shape_error_measures_r_surface_in_meters() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    candidate = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=np.float64)

    assert pareto_shape_error(reference, candidate, metric="rms") == pytest.approx(np.sqrt(2.5))
    assert pareto_shape_error(reference, candidate, metric="max") == 2.0

    with pytest.raises(ValueError, match="shape mismatch"):
        pareto_shape_error(reference, candidate[:, :1], metric="rms")


def test_coefficient_blocks_decode_degree_first_layout() -> None:
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    blocks = coefficient_blocks_from_packed_state(
        x,
        profile_names=("h", "psin"),
        profile_L=np.array([1, 0], dtype=np.int64),
        coeff_index=np.array([[0, 2], [1, -1]], dtype=np.int64),
    )

    assert_allclose(blocks["h"], [10.0, 30.0])
    assert_allclose(blocks["psin"], [20.0])


def test_balanced_pareto_strategy_is_deterministic_and_capacity_bounded() -> None:
    topology = make_topology(h_count=5, v_count=4, kappa_count=3, psin_count=3)

    first = generate_pareto_signatures(
        topology,
        strategy="balanced",
        max_candidates=10,
    )
    second = generate_pareto_signatures(
        topology,
        strategy="balanced",
        max_candidates=10,
    )

    assert first == second
    assert len(first) == 10
    assert KernelParetoSignature.from_topology(topology) not in first
    for signature in first:
        candidate = topology_from_pareto_signature(topology, signature)
        assert candidate.x_size <= topology.x_size
        assert signature.psin_count >= 1


def test_tail_strategy_keeps_more_coefficients_for_large_max_tail() -> None:
    topology = make_topology(h_count=5, v_count=5, kappa_count=0, psin_count=2)
    coefficients = {
        "h": np.array([1.0, 0.01, 0.001, 0.0001, 0.00001], dtype=np.float64),
        "v": np.ones(5, dtype=np.float64),
        "psin": np.ones(2, dtype=np.float64),
    }

    signatures = generate_pareto_signatures(
        topology,
        strategy="tail",
        coefficients_by_profile=coefficients,
        max_candidates=20,
    )

    assert signatures
    assert any(signature.h_count < signature.v_count for signature in signatures)
    assert all(signature.psin_count >= 1 for signature in signatures)


def test_energy_strategy_keeps_more_coefficients_for_large_l2_tail() -> None:
    topology = make_topology(h_count=5, v_count=5, kappa_count=0, psin_count=2)
    coefficients = {
        "h": np.ones(5, dtype=np.float64),
        "v": np.array([1.0, 0.01, 0.001, 0.0001, 0.00001], dtype=np.float64),
        "psin": np.ones(2, dtype=np.float64),
    }

    signatures = generate_pareto_signatures(
        topology,
        strategy="energy",
        coefficients_by_profile=coefficients,
        max_candidates=20,
    )

    assert signatures
    assert any(signature.h_count > signature.v_count for signature in signatures)
    assert all(signature.psin_count >= 1 for signature in signatures)


def test_adaptive_strategy_starts_from_combined_seed_candidates() -> None:
    topology = make_topology(h_count=4, v_count=4, kappa_count=2, psin_count=2)
    coefficients = {
        "h": np.array([1.0, 0.01, 0.001, 0.0001], dtype=np.float64),
        "v": np.ones(4, dtype=np.float64),
        "k": np.array([1.0, 0.1], dtype=np.float64),
        "psin": np.ones(2, dtype=np.float64),
    }

    adaptive = generate_pareto_signatures(
        topology,
        strategy="adaptive",
        coefficients_by_profile=coefficients,
        max_candidates=6,
    )
    balanced = generate_pareto_signatures(topology, strategy="balanced", max_candidates=6)

    assert len(adaptive) == 6
    assert set(adaptive) != set(balanced)
    assert len(set(adaptive)) == len(adaptive)


def test_adaptive_refinement_generates_unseen_local_neighbors() -> None:
    topology = make_topology(h_count=4, v_count=3, kappa_count=0, psin_count=2)
    signature = KernelParetoSignature(
        h_count=2,
        v_count=2,
        kappa_count=0,
        psin_count=1,
        F_count=0,
        c_counts=(),
        s_counts=(),
    )
    sample = make_sample(
        topology=topology_from_pareto_signature(topology, signature),
        counts=5,
        time=1.0,
        complexity=5,
        shape_error=0.1,
    )

    neighbors = generate_adaptive_refinement_signatures(
        topology,
        frontier=(sample,),
        seen_signatures={KernelParetoSignature.from_topology(topology), signature},
        max_candidates=10,
    )

    assert neighbors
    assert signature not in neighbors
    assert len(neighbors) == len(set(neighbors))
    assert all(neighbor.psin_count >= 1 for neighbor in neighbors)
    assert any(neighbor.h_count == 1 for neighbor in neighbors)
    assert any(neighbor.h_count == 3 for neighbor in neighbors)


def test_numba_pareto_r_sampler_matches_equilibrium_without_history_mutation() -> None:
    topology = make_topology()
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    boundary = tiny_boundary()
    source = tiny_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)

    kernel.residual(x, boundary, source)
    assert kernel.result is None
    assert kernel.history == []

    impl = kernel._impl
    r_surface = sample_r_surface(
        impl._solver.runtime,
        x,
        impl._kernel_boundary(boundary),
        impl._kernel_source(source, case_name=None),
    )
    equilibrium = kernel.build_equilibrium(x)

    assert_allclose(r_surface, equilibrium.R)
    assert kernel.result is None
    assert kernel.history == []


def pareto_smoke_config() -> KernelConfig:
    return KernelConfig(
        method="powell",
        initial="cold-zeros",
        norm="none",
        max_residual=1.0e12,
        max_evaluations=1,
    )


def test_kernel_pareto_balanced_runs_and_restores_public_state() -> None:
    topology = make_topology(h_count=4, v_count=3, kappa_count=2, psin_count=3)
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    boundary = tiny_boundary()
    source = tiny_source()
    config = pareto_smoke_config()
    existing = kernel.solve(boundary, source, config=config)
    history_before = list(kernel.history)

    result = kernel.pareto(
        boundary,
        source,
        config=config,
        max_shape_error=[1.0e6],
        pareto_by="counts",
        strategy="balanced",
        metric="rms",
        max_candidates=3,
    )

    assert result.reference.shape_error == 0.0
    assert len(result.samples) <= 3
    assert result.frontier
    assert result.selected[1.0e6].result.success
    assert result.pareto_by == "counts"
    assert result.metric == "rms"
    assert result.strategy == "balanced"
    assert kernel.topology == topology
    assert kernel.result is existing
    assert kernel.history == history_before


@pytest.mark.parametrize("pareto_by", ["counts", "time", "complexity"])
@pytest.mark.parametrize("metric", ["rms", "max"])
def test_kernel_pareto_cost_axes_and_metrics_are_accepted(pareto_by: str, metric: str) -> None:
    topology = make_topology(h_count=3, v_count=2, kappa_count=1, psin_count=2)
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    result = kernel.pareto(
        tiny_boundary(),
        tiny_source(),
        config=pareto_smoke_config(),
        max_shape_error=1.0e6,
        pareto_by=pareto_by,
        strategy="balanced",
        metric=metric,
        max_candidates=1,
    )

    assert result.pareto_by == pareto_by
    assert result.metric == metric
    assert len(result.samples) <= 1
    assert 1.0e6 in result.selected


def test_kernel_pareto_adaptive_uses_seed_budget_for_small_searches() -> None:
    kernel = Kernel(
        topology=make_topology(h_count=3, v_count=2, kappa_count=1, psin_count=2),
        recipe=KernelRecipe(backend="numba"),
    )

    result = kernel.pareto(
        tiny_boundary(),
        tiny_source(),
        config=pareto_smoke_config(),
        strategy="adaptive",
        max_candidates=1,
        max_shape_error=1.0e6,
    )

    assert adaptive_seed_candidate_count(1) == 1
    assert len(result.samples) == 1
    assert len({sample.signature for sample in result.samples}) == len(result.samples)
    assert 1.0e6 in result.selected


def test_kernel_pareto_adaptive_refines_after_seed_frontier() -> None:
    max_candidates = 5
    kernel = Kernel(
        topology=make_topology(h_count=4, v_count=3, kappa_count=2, psin_count=3),
        recipe=KernelRecipe(backend="numba"),
    )

    result = kernel.pareto(
        tiny_boundary(),
        tiny_source(),
        config=pareto_smoke_config(),
        strategy="adaptive",
        max_candidates=max_candidates,
        max_shape_error=1.0e6,
    )

    assert adaptive_seed_candidate_count(max_candidates) == 3
    assert len(result.samples) > 3
    assert len(result.samples) <= max_candidates
    assert len({sample.signature for sample in result.samples}) == len(result.samples)
    assert result.frontier


def test_function_api_pareto_defaults_to_numba_backend() -> None:
    result = pareto_api(
        tiny_boundary(),
        tiny_source(),
        topology=make_topology(h_count=3, v_count=2, kappa_count=1, psin_count=2),
        config=pareto_smoke_config(),
        max_shape_error=1.0e6,
        strategy="balanced",
        max_candidates=0,
    )

    assert result.reference.result.success
    assert result.samples == ()
    assert result.frontier == (result.reference,)
    assert result.selected[1.0e6] is result.reference


def test_kernel_pareto_is_numba_only_for_now() -> None:
    kernel = Kernel(topology=make_topology(), recipe=KernelRecipe(backend="cxx"))

    with pytest.raises(NotImplementedError, match="Numba backend"):
        kernel.pareto(tiny_boundary(), tiny_source(), config=pareto_smoke_config())


def test_pareto_frontier_filters_failed_and_dominated_samples() -> None:
    topology = make_topology()
    samples = (
        make_sample(topology=topology, counts=5, time=5.0, complexity=50, shape_error=0.30),
        make_sample(topology=topology, counts=6, time=4.0, complexity=40, shape_error=0.20),
        make_sample(topology=topology, counts=7, time=3.0, complexity=30, shape_error=0.25),
        make_sample(topology=topology, counts=8, time=2.0, complexity=20, shape_error=0.10),
        make_sample(
            topology=topology,
            counts=9,
            time=1.0,
            complexity=10,
            shape_error=0.05,
            success=False,
        ),
        make_sample(
            topology=topology,
            counts=10,
            time=0.5,
            complexity=5,
            shape_error=float("nan"),
        ),
    )

    by_counts = pareto_frontier(samples, pareto_by="counts")
    assert [sample.counts for sample in by_counts] == [5, 6, 8]

    by_time = pareto_frontier(samples, pareto_by="time")
    assert [sample.time for sample in by_time] == [2.0]

    by_complexity = pareto_frontier(samples, pareto_by="complexity")
    assert [sample.complexity for sample in by_complexity] == [20]


def test_select_pareto_thresholds_chooses_lowest_cost_under_error() -> None:
    topology = make_topology()
    frontier = (
        make_sample(topology=topology, counts=5, time=5.0, complexity=50, shape_error=0.30),
        make_sample(topology=topology, counts=6, time=4.0, complexity=40, shape_error=0.20),
        make_sample(topology=topology, counts=8, time=2.0, complexity=20, shape_error=0.10),
    )

    selected = select_pareto_thresholds(frontier, [0.25, 0.15, 0.05], pareto_by="counts")

    assert selected[0.25].counts == 6
    assert selected[0.15].counts == 8
    assert 0.05 not in selected
