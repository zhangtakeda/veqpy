from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers import tiny_grid
from jax_helpers import tiny_pf_rho_grid_problem

from veqpy.model.equilibrium import Equilibrium
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX solver smoke requires the optional JAX dependency",
)


def test_jax_solver_smoke_pf_rho_grid_uses_existing_scipy_solver() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    operator = Operator(grid, problem, backend="jax")
    solver = Solver(
        operator=operator,
        config=SolverConfig(
            method="lm",
            max_residual=1.0e-5,
            max_evaluations=20,
            enable_fallback=False,
            enable_history=False,
            residual_normalization="none",
            initial_policy="zeros",
        ),
    )

    x = solver.solve()

    assert isinstance(x, np.ndarray)
    assert type(x).__module__.startswith("numpy")
    assert solver.result is not None
    assert isinstance(solver.result.x, np.ndarray)
    assert np.isfinite(float(solver.result.residual_norm_final))


def test_jax_build_equilibrium_snapshot_is_numpy_backed() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    operator = Operator(grid, problem, backend="jax")
    x = operator.zero_state()

    equilibrium = operator.build_equilibrium(x)

    assert isinstance(equilibrium, Equilibrium)
    assert isinstance(equilibrium.psin, np.ndarray)
    assert isinstance(equilibrium.FFn_psin, np.ndarray)
    assert isinstance(equilibrium.Pn_psin, np.ndarray)
    assert type(equilibrium.psin).__module__.startswith("numpy")
