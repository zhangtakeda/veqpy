from __future__ import annotations

import numpy as np
import pytest
from helpers import tiny_operator

from veqpy.solver import Solver, SolverConfig, SolverResult


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

    with pytest.raises(ValueError, match="Unsupported solver method"):
        SolverConfig(method="not-a-method")
    with pytest.raises(ValueError, match="collocation_weight"):
        SolverConfig(collocation_weight=1.5)
    with pytest.raises(ValueError, match="max_residual"):
        SolverConfig(max_residual=0.0)


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


def test_solver_attempt_label_reflects_initial_policy() -> None:
    solver = Solver(operator=tiny_operator(), config=SolverConfig(enable_history=False))
    nonzero_guess = np.ones(solver.operator.x_size, dtype=np.float64)
    zero_guess = np.zeros(solver.operator.x_size, dtype=np.float64)

    assert (
        solver._display_start_kind(
            nonzero_guess,
            solve_config=SolverConfig(initial_policy="homothetic"),
            x0_was_provided=False,
        )
        == "homothetic-start"
    )
    assert (
        solver._display_start_kind(
            zero_guess,
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
