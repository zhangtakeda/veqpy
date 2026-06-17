from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from helpers import tiny_grid, tiny_operator, tiny_pf_problem

from veqpy.model import Boundary
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig, SolverResult
from veqpy.solver.residual_scale import DEFAULT_RESIDUAL_NORMALIZATION
from veqpy.solver.solver import _AUTO_CURVE_STRAIN_THRESHOLD, _boundary_curve_strain


def test_solver_config_normalizes_aliases_and_validates_methods() -> None:
    config = SolverConfig(
        method=None,
        initial_policy="warmstart",
        fallback_methods=["lm", "lm", "trf"],
        residual_normalization="BLOCK_RMS",
    )

    assert config.method == "hybr"
    assert config.initial_policy == "warm"
    assert config.fallback_methods == ("lm", "trf")
    assert config.residual_normalization == "block-rms"
    assert SolverConfig(initial_policy="geometric").initial_policy == "geometric"
    assert SolverConfig(initial_policy="geometric-refined").initial_policy == "geometric-refined"
    assert SolverConfig(initial_policy="legacy-geometric").initial_policy == "legacy-geometric"
    assert SolverConfig(initial_policy="auto").initial_policy == "auto"

    with pytest.raises(ValueError, match="Unsupported solver method"):
        SolverConfig(method="not-a-method")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="tangent")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="geometric-no-psin")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="geom-psin")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="geom")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="geometric_refined")
    with pytest.raises(ValueError, match="Unsupported initial_policy"):
        SolverConfig(initial_policy="homothetic")
    with pytest.raises(ValueError, match="collocation_weight"):
        SolverConfig(collocation_weight=1.5)
    with pytest.raises(ValueError, match="max_residual"):
        SolverConfig(max_residual=0.0)


def test_solver_warm_start_interface_uses_initial_policy_only() -> None:
    assert "enable_warmstart" not in SolverConfig.__dataclass_fields__
    assert "enable_warmstart" not in inspect.signature(Solver.solve).parameters


def test_solver_default_normalization_is_single_source() -> None:
    assert DEFAULT_RESIDUAL_NORMALIZATION == "fast"
    assert SolverConfig().residual_normalization == DEFAULT_RESIDUAL_NORMALIZATION
    assert SolverConfig(residual_normalization=None).residual_normalization == (
        DEFAULT_RESIDUAL_NORMALIZATION
    )


def test_solver_homothetic_lambda_interface_is_removed() -> None:
    assert "initial_homothetic_lambda" not in SolverConfig.__dataclass_fields__
    assert "initial_homothetic_lambda" not in inspect.signature(Solver.solve).parameters


def test_solver_result_copies_input_arrays_and_validates_rank() -> None:
    x0 = np.array([1.0, 2.0])
    x = np.array([3.0, 4.0])
    result = SolverResult(
        x0=x0,
        x=x,
        success=True,
        message="ok",
        residual_norm_final=0.0,
        function_evaluations=1,
        jacobian_evaluations=0,
        iterations=1,
        elapsed=10.0,
    )

    x0[0] = 99.0
    x[0] = 99.0
    assert result.x0.tolist() == [1.0, 2.0]
    assert result.x.tolist() == [3.0, 4.0]
    assert result.total_elapsed == result.elapsed

    with pytest.raises(ValueError, match="x0 must be 1D"):
        SolverResult(
            x0=np.zeros((1, 1)),
            x=np.zeros(1),
            success=True,
            message="ok",
            residual_norm_final=0.0,
            function_evaluations=1,
            jacobian_evaluations=0,
            iterations=1,
            elapsed=10.0,
        )


def test_solver_facade_initial_state_and_history_lifecycle() -> None:
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))

    assert solver.result is None
    assert solver.history == []
    assert solver.x0.shape == (solver.operator.x_size,)

    solver.x0.fill(1.0)
    solver.reset()
    assert np.all(solver.x0 == 0.0)

    solver.history.append(object())
    solver.clear()
    assert solver.history == []


def test_solver_replace_case_is_problem_compatibility_alias() -> None:
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))
    replacement = tiny_pf_problem().copy()

    solver.replace_case(replacement)

    assert solver.operator.problem is replacement


def test_solver_attempt_label_reflects_initial_policy() -> None:
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))
    nonzero_guess = np.ones(solver.operator.x_size, dtype=np.float64)

    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="geometric"),
            x0_was_provided=False,
        )
        == "geometric-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="geometric-refined"),
            x0_was_provided=False,
        )
        == "geometric-refined-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="legacy-geometric"),
            x0_was_provided=False,
        )
        == "legacy-geometric-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="auto"),
            x0_was_provided=False,
        )
        == "auto-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="zeros"),
            x0_was_provided=False,
        )
        == "zero-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy=None),
            x0_was_provided=False,
        )
        == "encoded-start"
    )
    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy=None),
            x0_was_provided=True,
        )
        == "warm-start"
    )


def test_solver_solve_config_initial_policy_override_is_temporary() -> None:
    solver = Solver(
        operator=tiny_operator(),
        config=SolverConfig(initial_policy="geometric", enable_history=False),
    )
    base_kwargs = dict(
        method=None,
        max_residual=None,
        max_evaluations=None,
        enable_fallback=None,
        fallback_methods=None,
        enable_verbose=None,
        enable_history=None,
        residual_normalization=None,
        residual_normalization_floor=None,
        residual_normalization_max_ratio=None,
        residual_normalization_huber_tau=None,
        residual_normalization_probe_count=None,
        residual_normalization_probe_step=None,
        residual_normalization_sensitivity_lambda=None,
        enable_collocation=None,
        collocation_method=None,
        collocation_weight=None,
        collocation_max_residual=None,
        collocation_max_evaluations=None,
    )

    inherited = solver._resolve_solve_config(initial_policy=None, **base_kwargs)
    overridden = solver._resolve_solve_config(initial_policy="zeros", **base_kwargs)

    assert inherited.initial_policy == "geometric"
    assert overridden.initial_policy == "zeros"
    assert solver.config.initial_policy == "geometric"


def test_solver_solve_records_solve_and_total_elapsed_with_geometric_policy(monkeypatch) -> None:
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))

    def fake_solve_with_fallbacks(
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        assert solve_config.initial_policy == "geometric"
        assert residual_kind == "variational"
        assert not x0_was_provided
        return x_guess.copy(), True, "ok", 1, 0, 1, 0.0

    monkeypatch.setattr(solver, "_solve_with_fallbacks", fake_solve_with_fallbacks)

    solver.solve(initial_policy="geometric")

    assert solver.result is not None
    assert solver.result.elapsed > 0.0
    assert solver.result.total_elapsed >= solver.result.elapsed


def test_solver_elapsed_includes_internal_initial_state(monkeypatch) -> None:
    solver_module = importlib.import_module("veqpy.solver.solver")
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))
    ticks = iter([0.0, 10.0, 12.0, 15.0, 16.0])

    monkeypatch.setattr(solver_module, "perf_counter", lambda: next(ticks))

    def fake_build_initial_state(operator, solve_config: SolverConfig) -> np.ndarray:
        assert solve_config.initial_policy == "geometric"
        solver_module.perf_counter()
        return operator.zero_state()

    def fake_solve_with_fallbacks(
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        assert residual_kind == "variational"
        assert not x0_was_provided
        return x_guess.copy(), True, "ok", 1, 0, 1, 0.0

    monkeypatch.setattr(solver_module, "_build_initial_state", fake_build_initial_state)
    monkeypatch.setattr(solver, "_solve_with_fallbacks", fake_solve_with_fallbacks)

    solver.solve(initial_policy="geometric")

    assert solver.result is not None
    assert solver.result.elapsed == pytest.approx(5.0e6)
    assert solver.result.total_elapsed == pytest.approx(16.0e6)


def test_solver_geometric_ablation_policies_are_built_inside_solve(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    for policy in ("geometric", "geometric-refined", "legacy-geometric"):
        solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))

        def fake_solve_with_fallbacks(
            x_guess: np.ndarray,
            *,
            solve_config: SolverConfig,
            residual_kind: str,
            x0_was_provided: bool,
            policy_key: str = policy,
        ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
            assert solve_config.initial_policy == policy_key
            assert residual_kind == "variational"
            assert not x0_was_provided
            captured[policy_key] = x_guess.copy()
            return x_guess.copy(), True, "ok", 1, 0, 1, 0.0

        monkeypatch.setattr(solver, "_solve_with_fallbacks", fake_solve_with_fallbacks)
        solver.solve(initial_policy=policy)

    probe = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))
    no_psin_coeffs = probe.operator.unpack_coefficients(captured["geometric"])
    psin_coeffs = probe.operator.unpack_coefficients(captured["geometric-refined"])
    legacy_coeffs = probe.operator.unpack_coefficients(captured["legacy-geometric"])

    assert np.linalg.norm(no_psin_coeffs["psin"]) == pytest.approx(0.0)
    assert np.linalg.norm(psin_coeffs["psin"]) > 0.0
    assert legacy_coeffs["h"][0] != pytest.approx(no_psin_coeffs["h"][0])


def test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold(
    monkeypatch,
) -> None:
    def capture_x_guess(problem, initial_policy: str) -> np.ndarray:
        solver = Solver(
            operator=Operator(tiny_grid(), problem),
            config=SolverConfig(enable_history=False),
        )
        captured: dict[str, np.ndarray] = {}

        def fake_solve_with_fallbacks(
            x_guess: np.ndarray,
            *,
            solve_config: SolverConfig,
            residual_kind: str,
            x0_was_provided: bool,
        ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
            assert solve_config.initial_policy == initial_policy
            assert residual_kind == "variational"
            assert not x0_was_provided
            captured["x_guess"] = x_guess.copy()
            return x_guess.copy(), True, "ok", 1, 0, 1, 0.0

        monkeypatch.setattr(solver, "_solve_with_fallbacks", fake_solve_with_fallbacks)
        solver.solve(initial_policy=initial_policy)
        return captured["x_guess"]

    moderate_problem = tiny_pf_problem()
    assert _boundary_curve_strain(moderate_problem.boundary) < _AUTO_CURVE_STRAIN_THRESHOLD
    moderate_auto = capture_x_guess(moderate_problem, "auto")
    moderate_zeros = capture_x_guess(moderate_problem, "zeros")
    assert np.array_equal(moderate_auto, moderate_zeros)

    boundary = moderate_problem.boundary
    large_projection_problem = moderate_problem.replace(
        boundary=Boundary(
            a=boundary.a,
            R0=boundary.R0,
            Z0=boundary.Z0,
            B0=boundary.B0,
            ka=boundary.ka,
            c_offsets=np.array([0.0, 0.4], dtype=np.float64),
        )
    )
    assert _boundary_curve_strain(large_projection_problem.boundary) < _AUTO_CURVE_STRAIN_THRESHOLD
    large_projection_auto = capture_x_guess(large_projection_problem, "auto")
    large_projection_zeros = capture_x_guess(large_projection_problem, "zeros")
    assert np.array_equal(large_projection_auto, large_projection_zeros)

    high_s_problem = moderate_problem.replace(
        boundary=Boundary(
            a=boundary.a,
            R0=boundary.R0,
            Z0=boundary.Z0,
            B0=boundary.B0,
            ka=boundary.ka,
            s_offsets=np.array([0.0, np.arcsin(0.9)], dtype=np.float64),
        )
    )
    assert _boundary_curve_strain(high_s_problem.boundary) >= _AUTO_CURVE_STRAIN_THRESHOLD
    high_s_auto = capture_x_guess(high_s_problem, "auto")
    high_s_refined = capture_x_guess(high_s_problem, "geometric-refined")
    assert np.allclose(high_s_auto, high_s_refined)

    high_c_problem = moderate_problem.replace(
        boundary=Boundary(
            a=boundary.a,
            R0=boundary.R0,
            Z0=boundary.Z0,
            B0=boundary.B0,
            ka=boundary.ka,
            c_offsets=np.array([0.0, 0.8], dtype=np.float64),
        )
    )
    assert _boundary_curve_strain(high_c_problem.boundary) >= _AUTO_CURVE_STRAIN_THRESHOLD
    high_c_auto = capture_x_guess(high_c_problem, "auto")
    high_c_refined = capture_x_guess(high_c_problem, "geometric-refined")
    assert np.allclose(high_c_auto, high_c_refined)

    ellipse_problem = moderate_problem.replace(
        boundary=Boundary(
            a=boundary.a,
            R0=boundary.R0,
            Z0=boundary.Z0,
            B0=boundary.B0,
            ka=boundary.ka,
            c_offsets=np.array([5.0e-4], dtype=np.float64),
            s_offsets=np.array([0.0, 5.0e-4], dtype=np.float64),
        )
    )
    ellipse_auto = capture_x_guess(ellipse_problem, "auto")
    ellipse_zeros = capture_x_guess(ellipse_problem, "zeros")

    assert _boundary_curve_strain(ellipse_problem.boundary) < _AUTO_CURVE_STRAIN_THRESHOLD
    assert np.array_equal(ellipse_auto, ellipse_zeros)
