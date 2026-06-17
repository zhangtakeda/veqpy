from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers import tiny_grid
from jax_helpers import tiny_pf_rho_grid_problem
from numpy.testing import assert_allclose

from veqpy.engine.jax.compile import JaxCompileCache, global_residual_cache_size
from veqpy.engine.jax.state import JaxStaticSpec
from veqpy.operator import Operator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX residual parity tests require the optional JAX dependency",
)


def _spec() -> JaxStaticSpec:
    return JaxStaticSpec(
        route_key=("PF", "rho", "grid"),
        nr=8,
        nt=8,
        k_max=1,
        l_max=3,
        m_max=2,
        x_size=9,
        profile_names=("h", "k", "s1"),
        active_profile_ids=(0, 2, 6),
        active_lengths=(2, 2, 2),
        residual_block_codes=(0, 2, 6),
        residual_block_orders=(0, 0, 1),
        residual_block_radial_powers=(0, 0, 1),
    )


def test_jax_compile_cache_is_keyed_by_static_spec() -> None:
    cache = JaxCompileCache()
    spec = _spec()
    compiled = object()
    cache.put(spec, compiled)

    assert cache.get(_spec()) is compiled
    assert len(cache) == 1


def test_jax_residual_parity_pf_rho_grid() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    op_numba = Operator(grid, problem, backend="numba")
    op_jax = Operator(grid, problem, backend="jax")
    x = np.array([0.03, -0.04, 0.02], dtype=np.float64)

    r_numba = op_numba.residual_var(x)
    r_jax = op_jax.residual_var(x)

    assert isinstance(r_jax, np.ndarray)
    assert r_jax.dtype == np.float64
    assert_allclose(r_jax, r_numba, rtol=1.0e-8, atol=1.0e-8)


def test_jax_residual_var_into_writes_caller_owned_numpy_array() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    op_numba = Operator(grid, problem, backend="numba")
    op_jax = Operator(grid, problem, backend="jax")
    x = np.array([0.01, 0.02, -0.01], dtype=np.float64)
    expected = op_numba.residual_var(x)
    out = np.empty_like(expected)

    returned = op_jax.residual_var_into(x, out)

    assert returned is None
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert_allclose(out, expected, rtol=1.0e-8, atol=1.0e-8)


def test_jax_residual_public_output_contains_no_device_array() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    operator = Operator(grid, problem, backend="jax")
    residual = operator.residual_var(operator.zero_state())

    assert isinstance(residual, np.ndarray)
    assert type(residual).__module__.startswith("numpy")


def test_jax_residual_compile_cache_reuses_static_signature() -> None:
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    before = global_residual_cache_size()
    Operator(grid, problem, backend="jax")
    after_one = global_residual_cache_size()
    Operator(grid, problem, backend="jax")
    after_two = global_residual_cache_size()

    assert after_one <= before + 1
    assert after_two == after_one
