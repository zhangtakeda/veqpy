from __future__ import annotations

import pytest

from veqpy.cpp import (
    INITIAL_POLICY_COLD,
    INITIAL_POLICY_COLD_GEOMETRIC,
    INITIAL_POLICY_COLD_ZEROS,
    INITIAL_POLICY_WARM_CLONE,
    RESIDUAL_NORMALIZATION_BALANCED,
    RESIDUAL_NORMALIZATION_FAST,
    RESIDUAL_NORMALIZATION_NONE,
    RESIDUAL_NORMALIZATION_SAFE,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_NEWTON,
    SOLVER_METHOD_NEWTON_KRYLOV,
    SOLVER_METHOD_NEWTON_RAPHSON,
    SOLVER_METHOD_POWELL,
    initial_policy_code,
    residual_normalization_code,
    solver_method_code,
)


def test_cpp_solver_method_strings_are_exact_canonical_tokens() -> None:
    assert solver_method_code("powell") == SOLVER_METHOD_POWELL
    assert solver_method_code("levenberg-marquardt") == SOLVER_METHOD_LEVENBERG_MARQUARDT
    assert solver_method_code("newton") == SOLVER_METHOD_NEWTON
    assert solver_method_code("newton-krylov") == SOLVER_METHOD_NEWTON_KRYLOV
    assert solver_method_code("newton-raphson") == SOLVER_METHOD_NEWTON_RAPHSON

    for alias in ("lm", "Powell", "newton_krylov", "hybrd"):
        with pytest.raises(ValueError):
            solver_method_code(alias)


def test_cpp_initial_policy_strings_have_no_transitional_aliases() -> None:
    assert initial_policy_code("cold-zeros") == INITIAL_POLICY_COLD_ZEROS
    assert initial_policy_code("cold-geometric") == INITIAL_POLICY_COLD_GEOMETRIC
    assert initial_policy_code("cold") == INITIAL_POLICY_COLD
    assert initial_policy_code("warm-clone") == INITIAL_POLICY_WARM_CLONE

    for alias in ("zeros", "geometric", "geometric-refined", "auto", "warm", "warm_start"):
        with pytest.raises(ValueError):
            initial_policy_code(alias)


def test_cpp_residual_normalization_strings_are_four_exact_modes() -> None:
    assert residual_normalization_code("none") == RESIDUAL_NORMALIZATION_NONE
    assert residual_normalization_code("fast") == RESIDUAL_NORMALIZATION_FAST
    assert residual_normalization_code("balanced") == RESIDUAL_NORMALIZATION_BALANCED
    assert residual_normalization_code("safe") == RESIDUAL_NORMALIZATION_SAFE

    for alias in ("block-rms", "block_huber", "balance", "Fast"):
        with pytest.raises(ValueError):
            residual_normalization_code(alias)
