from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import MU0, pf_reference_profiles
from numpy.linalg import norm
from numpy.testing import assert_allclose

from veqlib.facade import Kernel as UnifiedKernel
from veqlib.facade import KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqlib.facade.source_semantics import materialize_kernel_source
from veqpy.kernel import NumbaKernel
from veqpy.kernel.residual_scale import make_residual_scale

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


def test_numba_kernel_recipe_validation_and_public_surface() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)

    assert kernel.x_size == topology.x_size
    assert kernel.recipe.backend == "numba"
    assert kernel.recipe.layout == "degree"
    dry_run = kernel.prepare(dry_run=True)
    assert dry_run.topology is topology
    assert dry_run.recipe is kernel.recipe
    assert dry_run.backend == "numba"
    assert dry_run.prepared is False
    assert dry_run.artifact is None
    assert dry_run.warmed is False
    assert dry_run.dry_run is True
    assert np.isnan(dry_run.raw_norm)

    prepared = kernel.prepare()
    assert prepared.topology is topology
    assert prepared.recipe is kernel.recipe
    assert prepared.x_size == topology.x_size
    assert prepared.residual_size == topology.x_size
    assert prepared.backend == "numba"
    assert prepared.prepared is True
    assert prepared.artifact is None
    assert prepared.warmed is True
    assert prepared.dry_run is False
    assert np.isfinite(prepared.raw_norm)
    assert kernel.prepare() is prepared
    refreshed = kernel.prepare(force=True)
    assert refreshed is not prepared
    assert refreshed.warmed is True
    assert kernel.close() is None
    assert kernel.close() is None

    with pytest.raises(ValueError, match="layout='degree'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="numba", layout="family"))
    with pytest.raises(ValueError, match="backend='numba'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="cxx", layout="degree"))


def test_unified_kernel_selects_numba_backend_and_rejects_native_options(tmp_path) -> None:
    topology = make_kernel_topology()
    kernel = UnifiedKernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba", layout="degree"),
    )

    assert type(kernel).__name__ == "Kernel"
    assert kernel.recipe.backend == "numba"
    assert kernel.x_size == topology.x_size
    prepared = kernel.prepare(dry_run=True)
    assert prepared.backend == "numba"
    assert prepared.prepared is False
    assert prepared.artifact is None

    with pytest.raises(ValueError, match="native-only option"):
        UnifiedKernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba", layout="degree"),
            cache_root=tmp_path,
        )


def test_numba_kernel_residual_is_repeatable_and_validates_buffers() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual = kernel.residual(x, boundary, source)
    assert residual.shape == (kernel.x_size,)
    assert np.all(np.isfinite(residual))
    assert_allclose(residual, kernel.residual(x.copy(), boundary, source))

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
def test_kernel_source_materialization_route_matrix(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = route_kernel_topology(route, coordinate, nodes)
    source = route_kernel_source(route, topology.sample_count)
    materialized = materialize_kernel_source(topology, source)

    assert topology.source_route_key == (route, coordinate, nodes)
    assert materialized.scaled_heat.shape == (topology.sample_count,)
    assert materialized.scaled_current.shape == (topology.sample_count,)
    assert_allclose(materialized.scaled_heat, source.heat_profile * MU0)
    if route in {"PI", "PJ1", "PJ2"}:
        assert_allclose(materialized.scaled_current, source.current_profile * MU0)
    else:
        assert_allclose(materialized.scaled_current, source.current_profile)
    assert materialized.scaled_Ip == pytest.approx(source.Ip * MU0)
    assert_allclose([materialized.beta], [source.beta], equal_nan=True)
    assert not materialized.scaled_heat.flags.writeable
    assert not materialized.scaled_current.flags.writeable


@pytest.mark.parametrize(("route", "coordinate", "nodes"), ROUTE_PARITY_CASES)
def test_numba_kernel_residual_route_matrix_is_finite_and_repeatable(
    route: str,
    coordinate: str,
    nodes: str,
) -> None:
    topology = route_kernel_topology(route, coordinate, nodes)
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = route_kernel_source(route, topology.sample_count)
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual = kernel.residual(x, boundary, source)
    out = np.empty(kernel.x_size, dtype=np.float64)
    kernel.residual_into(out, x, boundary, source)

    assert residual.shape == (kernel.x_size,)
    assert np.all(np.isfinite(residual))
    assert_allclose(out, residual)
    assert_allclose(kernel.residual(x.copy(), boundary, source), residual)


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


def test_numba_kernel_solve_records_runtime_result() -> None:
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
    assert_allclose(result.raw, kernel.residual(result.x, boundary, source))
    assert kernel.result is result
    assert kernel.history == [result]


def test_numba_kernel_success_is_raw_residual_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()

    def fake_least_squares(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            x=np.full(kernel.x_size, 1.0e4, dtype=np.float64),
            success=True,
            message="synthetic optimizer success",
            nfev=1,
            njev=0,
            nit=0,
        )

    kernel_solver = import_module("veqpy.kernel.solver")
    monkeypatch.setattr(kernel_solver, "least_squares", fake_least_squares)

    result = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_residual=1.0e-12,
            max_evaluations=2,
        ),
    )

    assert not result.success
    assert result.info == 0


def test_numba_kernel_powell_uses_hybr_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    calls: list[str] = []
    budgets: list[int] = []

    def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append("hybr")
        options = kwargs["options"]
        budgets.append(int(options["maxfev"]))
        return SimpleNamespace(
            x=np.zeros(kernel.x_size, dtype=np.float64),
            success=True,
            message="synthetic root success",
            nfev=3,
            njev=0,
            nit=0,
        )

    def fake_least_squares(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        raise AssertionError("powell solve should not invoke least_squares")

    kernel_solver = import_module("veqpy.kernel.solver")
    monkeypatch.setattr(kernel_solver, "root", fake_root)
    monkeypatch.setattr(kernel_solver, "least_squares", fake_least_squares)

    result = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="powell",
            initial="cold-zeros",
            norm="none",
            max_residual=1.0,
            max_evaluations=2,
        ),
    )

    assert calls == ["hybr"]
    assert budgets == [2]
    assert result.nfev == 3
    assert result.success


def test_numba_kernel_build_equilibrium_uses_direct_runtime(
) -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)
    kernel.residual(x, boundary, source)

    equilibrium = kernel.build_equilibrium(x)
    assert np.isfinite(equilibrium.Ip)
    assert equilibrium.shape_profiles
    assert "h" in equilibrium.shape_profiles


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
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    config = KernelConfig(norm=norm_mode)
    x_final = np.ones(kernel.x_size, dtype=np.float64)

    def fake_least_squares(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            x=x_final.copy(),
            success=True,
            message="synthetic terminal state",
            nfev=3,
            njev=1,
            nit=2,
        )

    kernel_solver = import_module("veqpy.kernel.solver")
    original_least_squares = kernel_solver.least_squares
    kernel_solver.least_squares = fake_least_squares
    try:
        result = kernel.solve(
            boundary,
            source,
            config=KernelConfig(
                method="levenberg-marquardt",
                initial="cold-zeros",
                norm=norm_mode,
                max_evaluations=2,
            ),
        )
    finally:
        kernel_solver.least_squares = original_least_squares

    x0 = np.zeros(kernel.x_size, dtype=np.float64)
    expected_raw = kernel.residual(x_final, boundary, source)
    expected_scaled = _expected_scaled_from_reference(expected_raw, x0, kernel, config)

    assert_allclose(result.raw, expected_raw)
    assert_allclose(result.scaled, expected_scaled)
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
            kernel,
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


def test_numba_kernel_jvp_and_jacobian_match_native_finite_differences() -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)
    v = np.linspace(1.0, 2.0, kernel.x_size, dtype=np.float64)

    base = kernel.residual(x, boundary, source)
    eps = np.sqrt(1.0e-12) * (1.0 + norm(x)) / norm(v)
    expected_jvp = (kernel.residual(x + eps * v, boundary, source) - base) / eps

    jvp = kernel.jvp(x, v, boundary, source)
    assert_allclose(jvp, expected_jvp)

    jvp_out = np.empty_like(x)
    kernel.jvp_into(jvp_out, x.tolist(), v.tolist(), boundary, source)
    assert_allclose(jvp_out, expected_jvp)

    zero_jvp = kernel.jvp(x, np.zeros_like(v), boundary, source)
    assert_allclose(zero_jvp, np.zeros_like(zero_jvp))

    expected_jacobian = np.empty((kernel.x_size, kernel.x_size), dtype=np.float64)
    for col, saved in enumerate(x):
        step = 1.0e-7 * max(1.0, abs(float(saved)))
        x_plus = x.copy()
        x_plus[col] = saved + step
        expected_jacobian[:, col] = (kernel.residual(x_plus, boundary, source) - base) / step

    jacobian = kernel.jacobian(x, boundary, source)
    assert_allclose(jacobian, expected_jacobian)

    jacobian_out = np.empty_like(expected_jacobian)
    kernel.jacobian_into(jacobian_out, x.tolist(), boundary, source)
    assert_allclose(jacobian_out, expected_jacobian)

    with pytest.raises(ValueError, match="v must have shape"):
        kernel.jvp(x, np.ones(kernel.x_size + 1, dtype=np.float64), boundary, source)
    with pytest.raises(ValueError, match="out must have shape"):
        kernel.jacobian_into(
            np.empty((kernel.x_size, kernel.x_size + 1), dtype=np.float64),
            x,
            boundary,
            source,
        )


def _expected_scaled_from_reference(
    raw: np.ndarray,
    x_reference: np.ndarray,
    kernel: NumbaKernel,
    config: KernelConfig,
) -> np.ndarray:
    if config.norm == "none":
        return raw.copy()
    if kernel._last_boundary is None or kernel._last_source is None:
        raise RuntimeError("kernel case must be set before computing expected scale")
    reference_raw = kernel.residual(x_reference, kernel._last_boundary, kernel._last_source)
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
        kernel._solver.runtime.residual_block_lengths(),
        **params,
    )
    if scale is None:
        scale = np.ones_like(raw)
    return raw / scale
