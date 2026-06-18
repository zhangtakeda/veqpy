from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers import tiny_grid
from jax_helpers import tiny_pf_rho_grid_problem

from veqpy.engine.backend import SnapshotNotPublishedError, UnsupportedBackendFeature
from veqpy.operator import Operator

JAX_INSTALLED = importlib.util.find_spec("jax") is not None

@pytest.mark.skipif(not JAX_INSTALLED, reason="JAX residual contract requires JAX installed")
def test_jax_public_residual_methods_return_numpy_for_supported_route() -> None:
    grid = tiny_grid()
    operator = Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
    x = operator.zero_state()
    out = np.empty(operator.x_size, dtype=np.float64)

    residual = operator.residual_var(x)
    operator.residual_var_into(x, out)

    assert isinstance(residual, np.ndarray)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, residual)


@pytest.mark.skipif(not JAX_INSTALLED, reason="JAX staged contract requires JAX installed")
def test_jax_public_collocation_and_stage_methods_raise_explicit_unsupported(monkeypatch) -> None:
    del monkeypatch
    grid = tiny_grid()
    operator = Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
    x = operator.zero_state()
    collocation_out = np.empty(
        operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt,
        dtype=np.float64,
    )

    with pytest.raises(UnsupportedBackendFeature):
        operator.residual_collocation(x)
    with pytest.raises(UnsupportedBackendFeature):
        operator.residual_collocation_into(x, collocation_out)
    with pytest.raises(UnsupportedBackendFeature):
        operator.stage_a_profile(x)
    with pytest.raises(UnsupportedBackendFeature):
        operator.stage_b_geometry()
    with pytest.raises(UnsupportedBackendFeature):
        operator.stage_c_source()
    with pytest.raises(UnsupportedBackendFeature):
        operator.stage_d_residual()


@pytest.mark.skipif(not JAX_INSTALLED, reason="JAX snapshot contract requires JAX installed")
def test_jax_public_snapshot_is_supported_and_alpha_requires_snapshot() -> None:
    grid = tiny_grid()
    operator = Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
    x = operator.zero_state()

    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha1
    with pytest.raises(SnapshotNotPublishedError):
        _ = operator.alpha2

    equilibrium = operator.build_equilibrium(x)

    assert isinstance(equilibrium.psin, np.ndarray)
    assert isinstance(operator.alpha1, float)
    assert isinstance(operator.alpha2, float)
    with pytest.raises(UnsupportedBackendFeature):
        operator.alpha1 = 1.0
    with pytest.raises(UnsupportedBackendFeature):
        operator.alpha2 = 1.0


@pytest.mark.skipif(not JAX_INSTALLED, reason="JAX replace-problem contract requires JAX installed")
def test_jax_replace_problem_revalidates_backend_capability(monkeypatch) -> None:
    del monkeypatch
    grid = tiny_grid()
    operator = Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")

    with pytest.raises(UnsupportedBackendFeature, match="does not support route"):
        operator.replace_problem(tiny_pf_rho_grid_problem(tiny_grid()).replace(coordinate="psin"))
