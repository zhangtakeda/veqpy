from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy import Kernel, KernelBoundary, KernelRecipe, KernelSource
from veqpy.kernels.numba_kernel.pareto_runtime import sample_r_surface
from veqpy.kernels.pareto import (
    KernelParetoSignature,
    ParetoSample,
    normalize_pareto_by,
    normalize_pareto_max_candidates,
    normalize_pareto_metric,
    normalize_pareto_strategy,
    normalize_shape_error_thresholds,
    pareto_frontier,
    pareto_sample_complexity,
    pareto_shape_error,
    select_pareto_thresholds,
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
    complexity: float,
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
    assert pareto_sample_complexity(result, topology) == float(expected)


def test_pareto_shape_error_measures_r_surface_in_meters() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    candidate = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=np.float64)

    assert pareto_shape_error(reference, candidate, metric="rms") == pytest.approx(np.sqrt(2.5))
    assert pareto_shape_error(reference, candidate, metric="max") == 2.0

    with pytest.raises(ValueError, match="shape mismatch"):
        pareto_shape_error(reference, candidate[:, :1], metric="rms")


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


def test_pareto_frontier_filters_failed_and_dominated_samples() -> None:
    topology = make_topology()
    samples = (
        make_sample(topology=topology, counts=5, time=5.0, complexity=50.0, shape_error=0.30),
        make_sample(topology=topology, counts=6, time=4.0, complexity=40.0, shape_error=0.20),
        make_sample(topology=topology, counts=7, time=3.0, complexity=30.0, shape_error=0.25),
        make_sample(topology=topology, counts=8, time=2.0, complexity=20.0, shape_error=0.10),
        make_sample(
            topology=topology,
            counts=9,
            time=1.0,
            complexity=10.0,
            shape_error=0.05,
            success=False,
        ),
        make_sample(
            topology=topology,
            counts=10,
            time=0.5,
            complexity=5.0,
            shape_error=float("nan"),
        ),
    )

    by_counts = pareto_frontier(samples, pareto_by="counts")
    assert [sample.counts for sample in by_counts] == [5, 6, 8]

    by_time = pareto_frontier(samples, pareto_by="time")
    assert [sample.time for sample in by_time] == [2.0]

    by_complexity = pareto_frontier(samples, pareto_by="complexity")
    assert [sample.complexity for sample in by_complexity] == [20.0]


def test_select_pareto_thresholds_chooses_lowest_cost_under_error() -> None:
    topology = make_topology()
    frontier = (
        make_sample(topology=topology, counts=5, time=5.0, complexity=50.0, shape_error=0.30),
        make_sample(topology=topology, counts=6, time=4.0, complexity=40.0, shape_error=0.20),
        make_sample(topology=topology, counts=8, time=2.0, complexity=20.0, shape_error=0.10),
    )

    selected = select_pareto_thresholds(frontier, [0.25, 0.15, 0.05], pareto_by="counts")

    assert selected[0.25].counts == 6
    assert selected[0.15].counts == 8
    assert 0.05 not in selected
