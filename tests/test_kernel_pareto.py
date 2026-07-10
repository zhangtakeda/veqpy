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
    coefficient_blocks_from_packed_state,
    generate_adaptive_refinement_signatures,
    generate_pareto_signatures,
    normalize_pareto_candidates,
    normalize_pareto_metric,
    normalize_pareto_neighborhood_size,
    normalize_pareto_strategy,
    normalize_pareto_target,
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
        "constraint": "ip",
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
    assert normalize_pareto_target("COUNTS") == "counts"
    assert normalize_pareto_metric("Max") == "max"
    assert normalize_pareto_strategy(" adaptive ") == "adaptive"
    assert normalize_pareto_neighborhood_size(0) == 0
    assert normalize_pareto_neighborhood_size(2) == 2
    assert normalize_shape_error_thresholds(None) == ()
    assert normalize_shape_error_thresholds(1.0e-3) == (1.0e-3,)
    assert normalize_shape_error_thresholds([1.0e-3, 2.0e-3]) == (1.0e-3, 2.0e-3)

    with pytest.raises(ValueError, match="target"):
        normalize_pareto_target("elapsed")
    with pytest.raises(ValueError, match="metric"):
        normalize_pareto_metric("mean")
    with pytest.raises(ValueError, match="strategy"):
        normalize_pareto_strategy("random")
    with pytest.raises(TypeError, match="neighborhood_size"):
        normalize_pareto_neighborhood_size(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        normalize_pareto_neighborhood_size(-1)
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


def test_pareto_candidates_normalize_mappings_topologies_and_signatures() -> None:
    topology = make_topology(h_count=4, v_count=3, psin_count=2, c_counts=(2,), s_counts=(2,))
    candidate_topology = topology_from_pareto_signature(
        topology,
        KernelParetoSignature(
            h_count=2,
            v_count=1,
            kappa_count=1,
            psin_count=1,
            F_count=0,
            c_counts=(1,),
            s_counts=(1,),
        ),
    )

    signatures = normalize_pareto_candidates(
        topology,
        (
            {"h": 2, "v_count": 1, "kappa": 1, "psin": 1, "c_counts": (1,), "s1": 1},
            candidate_topology,
            KernelParetoSignature.from_topology(topology),
        ),
    )

    assert signatures == (KernelParetoSignature.from_topology(candidate_topology),)

    with pytest.raises(ValueError, match="unknown"):
        normalize_pareto_candidates(topology, {"unknown": 1})


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
    )
    second = generate_pareto_signatures(
        topology,
        strategy="balanced",
    )

    assert first == second
    assert first
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
    )
    balanced = generate_pareto_signatures(topology, strategy="balanced")

    assert adaptive
    assert set(adaptive) != set(balanced)
    assert len(set(adaptive)) == len(adaptive)


def test_adaptive_seed_includes_structural_floor_neighborhood() -> None:
    topology = make_topology(h_count=3, v_count=0, kappa_count=3, psin_count=3, s_counts=(3,))
    coefficients = {
        "h": np.ones(3, dtype=np.float64),
        "k": np.ones(3, dtype=np.float64),
        "psin": np.ones(3, dtype=np.float64),
        "s1": np.ones(3, dtype=np.float64),
    }

    adaptive = generate_pareto_signatures(
        topology,
        strategy="adaptive",
        coefficients_by_profile=coefficients,
    )

    assert KernelParetoSignature(
        h_count=1,
        v_count=0,
        kappa_count=1,
        psin_count=1,
        F_count=0,
        c_counts=(),
        s_counts=(1,),
    ) in adaptive
    assert KernelParetoSignature(
        h_count=1,
        v_count=0,
        kappa_count=2,
        psin_count=1,
        F_count=0,
        c_counts=(),
        s_counts=(1,),
    ) in adaptive


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
        neighborhood_size=1,
    )

    assert neighbors
    assert signature not in neighbors
    assert len(neighbors) == len(set(neighbors))
    assert all(neighbor.psin_count >= 1 for neighbor in neighbors)
    assert any(neighbor.h_count == 1 for neighbor in neighbors)
    assert any(neighbor.h_count == 3 for neighbor in neighbors)


def test_adaptive_refinement_moves_fourier_harmonic_pairs() -> None:
    topology = make_topology(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=1,
        c_counts=(3, 2),
        s_counts=(3, 2),
    )
    signature = KernelParetoSignature(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=1,
        F_count=0,
        c_counts=(2, 1),
        s_counts=(2, 1),
    )
    sample = make_sample(
        topology=topology_from_pareto_signature(topology, signature),
        counts=6,
        time=1.0,
        complexity=6,
        shape_error=0.1,
    )

    neighbors = generate_adaptive_refinement_signatures(
        topology,
        frontier=(sample,),
        seen_signatures={KernelParetoSignature.from_topology(topology), signature},
        neighborhood_size=1,
    )

    assert KernelParetoSignature(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=1,
        F_count=0,
        c_counts=(3, 1),
        s_counts=(3, 1),
    ) in neighbors
    assert KernelParetoSignature(
        h_count=1,
        v_count=0,
        kappa_count=0,
        psin_count=1,
        F_count=0,
        c_counts=(2,),
        s_counts=(2,),
    ) in neighbors


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
    candidates = generate_pareto_signatures(topology, strategy="balanced")[:4]

    result = kernel.pareto(
        boundary,
        source,
        candidates=candidates,
        config=config,
        reference=existing,
        target="counts",
        metric="rms",
    )

    assert result.reference.shape_error == 0.0
    assert result.samples
    assert result.frontier
    assert result.target == "counts"
    assert result.metric == "rms"
    assert kernel.topology == topology
    assert kernel.result is existing
    assert kernel.history == history_before


def test_numba_pareto_private_progress_callback_reports_candidate_steps() -> None:
    topology = make_topology(h_count=4, v_count=3, kappa_count=2, psin_count=3)
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    events: list[tuple[str, int, int]] = []
    candidates = generate_pareto_signatures(topology, strategy="balanced")[:3]

    result = kernel._impl.pareto(  # type: ignore[attr-defined]
        tiny_boundary(),
        tiny_source(),
        candidates=candidates,
        config=pareto_smoke_config(),
        _progress_callback=lambda phase, completed, total: events.append(
            (phase, completed, total)
        ),
    )

    assert result.samples
    assert ("ref", 0, 0) in events
    assert events[-1] == ("run", len(result.samples), len(result.samples))


@pytest.mark.parametrize("target", ["counts", "time", "complexity"])
@pytest.mark.parametrize("metric", ["rms", "max"])
def test_kernel_pareto_cost_axes_and_metrics_are_accepted(target: str, metric: str) -> None:
    topology = make_topology(h_count=3, v_count=2, kappa_count=1, psin_count=2)
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    candidates = generate_pareto_signatures(topology, strategy="balanced")[:2]
    result = kernel.pareto(
        tiny_boundary(),
        tiny_source(),
        candidates=candidates,
        config=pareto_smoke_config(),
        target=target,
        metric=metric,
    )

    assert result.target == target
    assert result.metric == metric
    assert result.samples
    assert len({sample.signature for sample in result.samples}) == len(result.samples)


def test_function_api_pareto_defaults_to_numba_backend() -> None:
    topology = make_topology(h_count=3, v_count=2, kappa_count=1, psin_count=2)
    candidates = generate_pareto_signatures(topology, strategy="balanced")[:2]
    result = pareto_api(
        tiny_boundary(),
        tiny_source(),
        topology=topology,
        candidates=candidates,
        config=pareto_smoke_config(),
        target="counts",
    )

    assert result.reference.result.success
    assert result.samples
    assert result.frontier
    assert result.target == "counts"


def test_kernel_pareto_is_numba_only_for_now() -> None:
    kernel = Kernel(topology=make_topology(), recipe=KernelRecipe(backend="cxx"))

    with pytest.raises(NotImplementedError, match="Numba backend"):
        kernel.pareto(tiny_boundary(), tiny_source(), candidates=(), config=pareto_smoke_config())


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

    by_counts = pareto_frontier(samples, target="counts")
    assert [sample.counts for sample in by_counts] == [5, 6, 8]

    by_time = pareto_frontier(samples, target="time")
    assert [sample.time for sample in by_time] == [2.0]

    by_complexity = pareto_frontier(samples, target="complexity")
    assert [sample.complexity for sample in by_complexity] == [20]


def test_select_pareto_thresholds_chooses_lowest_cost_under_error() -> None:
    topology = make_topology()
    frontier = (
        make_sample(topology=topology, counts=5, time=5.0, complexity=50, shape_error=0.30),
        make_sample(topology=topology, counts=6, time=4.0, complexity=40, shape_error=0.20),
        make_sample(topology=topology, counts=8, time=2.0, complexity=20, shape_error=0.10),
    )

    selected = select_pareto_thresholds(frontier, [0.25, 0.15, 0.05], target="counts")

    assert selected[0.25].counts == 6
    assert selected[0.15].counts == 8
    assert 0.05 not in selected
