from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers import tiny_grid
from jax_helpers import tiny_pf_rho_grid_problem
from numpy.testing import assert_allclose

from veqpy.engine.jax.config import require_jax
from veqpy.engine.jax.geometry import evaluate_geometry_stage_pf_rho_grid
from veqpy.engine.jax.operator import build_pf_rho_grid_runtime
from veqpy.engine.jax.profile import evaluate_profile_stage_pf_rho_grid
from veqpy.engine.jax.source import evaluate_source_stage_pf_rho_grid
from veqpy.operator import Operator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX stage parity tests require the optional JAX dependency",
)


def test_jax_stage_parity_pf_rho_grid() -> None:
    jax = require_jax()
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    operator = Operator(grid, problem, backend="numba")
    x = np.array([0.03, -0.04, 0.02], dtype=np.float64)

    runtime = build_pf_rho_grid_runtime(
        jax_module=jax,
        plan=operator.plan,
        problem=operator.problem,
        profile_workspace=operator.profile_workspace,
        grid_workspace=operator.plan.grid_workspace,
        source_workspace=operator.source_workspace,
        residual_binding_layout=operator.plan.residual_binding_layout,
        c_effective_order=operator.c_effective_order,
        s_effective_order=operator.s_effective_order,
        fix_rho=operator.fix_rho,
    )

    operator.stage_a_profile(x)
    expected_profile = operator.profile_workspace.profile_fields.copy()
    profile_fields = evaluate_profile_stage_pf_rho_grid(
        jax,
        runtime.device_state.leaves,
        runtime.static_spec,
        jax.device_put(x),
    )
    assert_allclose(np.asarray(profile_fields), expected_profile, rtol=1.0e-10, atol=1.0e-10)

    operator.stage_b_geometry()
    expected_surface = operator.geometry_workspace.surface_fields.copy()
    expected_radial = operator.geometry_workspace.radial_fields.copy()
    surface_fields, radial_fields, _, _ = evaluate_geometry_stage_pf_rho_grid(
        jax,
        runtime.device_state.leaves,
        runtime.static_spec,
        profile_fields,
    )
    assert_allclose(np.asarray(surface_fields), expected_surface, rtol=1.0e-10, atol=1.0e-10)
    assert_allclose(np.asarray(radial_fields), expected_radial, rtol=1.0e-10, atol=1.0e-10)

    expected_alpha = np.asarray(operator.layout.run_source(), dtype=np.float64)
    expected_root = operator.residual_workspace.root_fields.copy()
    root_fields, alpha_state, _ = evaluate_source_stage_pf_rho_grid(
        jax,
        runtime.device_state.leaves,
        runtime.static_spec,
        radial_fields,
        surface_fields,
    )
    assert_allclose(np.asarray(root_fields), expected_root, rtol=1.0e-10, atol=1.0e-10)
    assert_allclose(np.asarray(alpha_state), expected_alpha, rtol=1.0e-10, atol=1.0e-10)


def test_jax_stage_evaluation_does_not_mutate_numba_workspace() -> None:
    jax = require_jax()
    grid = tiny_grid()
    problem = tiny_pf_rho_grid_problem(grid)
    operator = Operator(grid, problem, backend="numba")
    x = np.array([0.01, 0.02, -0.01], dtype=np.float64)
    profile_before = operator.profile_workspace.profile_fields.copy()
    geometry_before = operator.geometry_workspace.surface_fields.copy()

    runtime = build_pf_rho_grid_runtime(
        jax_module=jax,
        plan=operator.plan,
        problem=operator.problem,
        profile_workspace=operator.profile_workspace,
        grid_workspace=operator.plan.grid_workspace,
        source_workspace=operator.source_workspace,
        residual_binding_layout=operator.plan.residual_binding_layout,
        c_effective_order=operator.c_effective_order,
        s_effective_order=operator.s_effective_order,
        fix_rho=operator.fix_rho,
    )
    profile_fields = evaluate_profile_stage_pf_rho_grid(
        jax,
        runtime.device_state.leaves,
        runtime.static_spec,
        jax.device_put(x),
    )
    evaluate_geometry_stage_pf_rho_grid(
        jax,
        runtime.device_state.leaves,
        runtime.static_spec,
        profile_fields,
    )

    assert_allclose(operator.profile_workspace.profile_fields, profile_before)
    assert_allclose(operator.geometry_workspace.surface_fields, geometry_before)
