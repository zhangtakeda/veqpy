from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    MU0,
    pf_reference_profiles,
    tiny_boundary,
    tiny_grid,
    tiny_operator,
    tiny_pf_problem,
)
from numpy.testing import assert_allclose

from veqpy.model import Problem
from veqpy.operator import Operator


def test_operator_residuals_coefficients_and_equilibrium_snapshot() -> None:
    operator = tiny_operator()
    x0 = operator.zero_state()

    assert x0.shape == (operator.x_size,)
    residual = operator.residual_var(x0)
    assert residual.shape == (operator.x_size,)
    assert np.all(np.isfinite(residual))

    out = np.empty_like(residual)
    operator.residual_var_into(x0.tolist(), out)
    assert_allclose(out, residual)

    collocation = operator.residual_collocation(x0)
    assert collocation.ndim == 1
    assert collocation.size > operator.x_size
    assert np.all(np.isfinite(collocation))

    coeffs = operator.build_coeffs(x0)
    assert set(coeffs) == {"h", "k", "s1", "psin"}

    equilibrium = operator.build_equilibrium(x0)
    assert equilibrium.grid.Nr == tiny_grid().Nr
    assert np.isfinite(equilibrium.Ip)

    with pytest.raises(ValueError, match="Expected x to have shape"):
        operator.residual_var(np.zeros(operator.x_size + 1, dtype=np.float64))
    with pytest.raises(TypeError, match="dtype float64"):
        operator.residual_var_into(x0, np.empty(operator.x_size, dtype=np.float32))


def test_operator_problem_alias_and_compatible_replacement() -> None:
    problem = tiny_pf_problem()
    operator = Operator(tiny_grid(), case=problem)

    assert operator.problem is problem
    assert operator.case is problem

    replacement = problem.copy()
    operator.replace_case(replacement)
    assert operator.problem is replacement
    assert operator.case is replacement

    assigned = problem.copy()
    operator.case = assigned
    assert operator.problem is assigned

    with pytest.raises(TypeError, match="either problem or case"):
        Operator(tiny_grid(), problem, case=replacement)


def test_operator_rejects_invalid_user_route_profile_combinations() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)

    with pytest.raises(ValueError, match="rho/uniform does not accept an active psin profile"):
        Operator(
            tiny_grid(),
            Problem(
                route="PF",
                coordinate="rho",
                nodes="uniform",
                active_profiles={"h": 2, "k": 2, "s1": 2, "psin": 2},
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.ones_like(rho),
            ),
        )

    with pytest.raises(ValueError, match="active F profile.*only supported for PJ2"):
        Operator(
            tiny_grid(),
            Problem(
                route="PF",
                coordinate="rho",
                nodes="uniform",
                active_profiles={"h": 2, "k": 2, "s1": 2, "F": 2},
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.ones_like(rho),
            ),
        )

    with pytest.raises(ValueError, match="PJ2 requires an active F profile"):
        Operator(
            tiny_grid(),
            Problem(
                route="PJ2",
                coordinate="rho",
                nodes="uniform",
                active_profiles={"h": 2, "k": 2, "s1": 2},
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.full_like(rho, 1.0e6),
            ),
        )


def test_operator_keeps_problem_inputs_raw_and_rejects_pre_scaled_ip() -> None:
    heat_input = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    current_input = np.array([0.0, 2.0e6, 3.0e6], dtype=np.float64)
    problem = Problem(
        route="PI",
        coordinate="rho",
        active_profiles={"h": 2},
        boundary=tiny_boundary(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )
    Operator(tiny_grid(), problem)

    assert_allclose(problem.heat_input, heat_input)
    assert_allclose(problem.current_input, current_input)

    bad_problem = Problem(
        route="PF",
        coordinate="rho",
        active_profiles={"h": 2},
        boundary=tiny_boundary(),
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.ones(3, dtype=np.float64),
        Ip=MU0 * 3.0e6,
    )
    with pytest.warns(RuntimeWarning, match="Rejected setup input magnitude"):
        with pytest.raises(ValueError, match="Rejected setup input magnitude"):
            Operator(tiny_grid(), bad_problem)

    bad_current_problem = Problem(
        route="PF",
        coordinate="rho",
        active_profiles={"h": 2},
        boundary=tiny_boundary(),
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.full(3, 1.0e6, dtype=np.float64),
    )
    with pytest.warns(RuntimeWarning, match="Pass unnormalized setup values to Problem"):
        with pytest.raises(ValueError, match="current_input max_abs"):
            Operator(tiny_grid(), bad_current_problem)


def test_pf_rho_grid_unconstrained_equilibrium_uses_positive_flux_branch() -> None:
    grid = tiny_grid()
    rho = np.asarray(grid.rho, dtype=np.float64)
    ffn_psin, pn_psin = pf_reference_profiles(rho * rho)
    problem = Problem(
        route="PF",
        coordinate="rho",
        nodes="grid",
        active_profiles={"h": 1, "k": 1, "s1": 1},
        boundary=tiny_boundary(),
        heat_input=pn_psin * (2.0 * rho) / MU0,
        current_input=ffn_psin * (2.0 * rho),
    )

    operator = Operator(grid, problem)
    equilibrium = operator.build_equilibrium(operator.zero_state())

    assert equilibrium.alpha2 > 0.0
