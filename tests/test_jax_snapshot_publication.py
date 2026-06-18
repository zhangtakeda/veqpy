from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers import tiny_grid
from jax_helpers import tiny_pf_rho_grid_problem
from numpy.testing import assert_allclose

from veqpy.engine.backend import SnapshotNotPublishedError, UnsupportedBackendFeature
from veqpy.model.equilibrium import Equilibrium
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX snapshot publication tests require the optional JAX dependency",
)


def _x() -> np.ndarray:
    return np.array([0.02, -0.01, 0.03], dtype=np.float64)


def _jax_operator() -> Operator:
    grid = tiny_grid()
    return Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")


def test_jax_residual_hot_path_does_not_publish_snapshot() -> None:
    operator = _jax_operator()
    runtime = operator.layout.backend_runtime

    for _ in range(3):
        residual = operator.residual_var(_x())
        assert isinstance(residual, np.ndarray)

    assert runtime.snapshot_publish_count == 0
    assert runtime.residual_call_count == 3
    assert runtime.residual_host_copy_count == 3
    assert runtime.residual_host_bytes == 3 * operator.x_size * np.dtype(np.float64).itemsize
    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha1


def test_jax_residual_into_writes_caller_owned_output_without_snapshot() -> None:
    operator = _jax_operator()
    runtime = operator.layout.backend_runtime
    out = np.empty(operator.x_size, dtype=np.float64)

    returned = operator.residual_var_into(_x(), out)

    assert returned is None
    assert isinstance(out, np.ndarray)
    assert runtime.snapshot_publish_count == 0
    with pytest.raises(SnapshotNotPublishedError):
        operator.build_equilibrium()


def test_jax_publish_snapshot_returns_numpy_host_state() -> None:
    operator = _jax_operator()

    snapshot = operator.publish_snapshot(_x())

    assert isinstance(snapshot.x, np.ndarray)
    assert isinstance(snapshot.profile_fields, np.ndarray)
    assert isinstance(snapshot.geometry_surface_fields, np.ndarray)
    assert isinstance(snapshot.root_fields, np.ndarray)
    assert isinstance(snapshot.alpha_state, np.ndarray)
    assert isinstance(operator.alpha1, float)
    assert isinstance(operator.alpha2, float)


def test_jax_build_equilibrium_publishes_snapshot_for_requested_x() -> None:
    operator = _jax_operator()
    runtime = operator.layout.backend_runtime

    equilibrium = operator.build_equilibrium(_x())

    assert isinstance(equilibrium, Equilibrium)
    assert isinstance(equilibrium.psin, np.ndarray)
    assert runtime.snapshot_publish_count == 1

    same_equilibrium = operator.build_equilibrium(_x())
    assert isinstance(same_equilibrium, Equilibrium)
    assert runtime.snapshot_publish_count == 1


def test_jax_snapshot_not_reused_for_different_x() -> None:
    operator = _jax_operator()
    runtime = operator.layout.backend_runtime
    x1 = _x()
    x2 = np.array([0.021, -0.01, 0.03], dtype=np.float64)

    operator.publish_snapshot(x1)
    operator.publish_snapshot(x2)

    assert runtime.snapshot_publish_count == 2
    assert np.array_equal(operator._published_snapshot_x, x2)


def test_jax_replace_problem_invalidates_snapshot() -> None:
    grid = tiny_grid()
    operator = Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
    operator.publish_snapshot(_x())

    operator.replace_problem(tiny_pf_rho_grid_problem(grid))

    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha1


def test_jax_alpha_requires_snapshot_and_is_read_only() -> None:
    operator = _jax_operator()

    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha1
    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha2

    operator.publish_snapshot(_x())

    assert isinstance(operator.alpha1, float)
    assert isinstance(operator.alpha2, float)
    with pytest.raises(UnsupportedBackendFeature):
        operator.alpha1 = 1.0
    with pytest.raises(UnsupportedBackendFeature):
        operator.alpha2 = 1.0


def test_jax_published_state_matches_numba_required_equilibrium_fields() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    x = _x()
    op_numba = Operator(grid, problem, backend="numba")
    op_jax = Operator(grid, problem, backend="jax")

    op_numba.residual_var(x)
    op_jax.publish_snapshot(x)

    assert_allclose(
        op_jax.residual_workspace.root_fields,
        op_numba.residual_workspace.root_fields,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert_allclose(
        op_jax.source_workspace.alpha_state,
        op_numba.source_workspace.alpha_state,
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_jax_solver_publishes_only_final_snapshot() -> None:
    operator = _jax_operator()
    runtime = operator.layout.backend_runtime
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

    assert runtime.snapshot_publish_count == 1
    assert solver.result is not None
    assert np.array_equal(operator._published_snapshot_x, solver.result.x)
    assert np.array_equal(operator._published_snapshot_x, x)

    equilibrium = solver.build_equilibrium()
    assert isinstance(equilibrium, Equilibrium)
    assert runtime.snapshot_publish_count == 1
