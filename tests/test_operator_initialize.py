from __future__ import annotations

import numpy as np
import pytest
from helpers import tiny_grid, tiny_operator, tiny_pf_problem

from veqpy.operator import Operator
from veqpy.operator.initialize import build_boundary_slope_initial_state, estimate_axis_shift_h0


def _active_profile_coeff_indices(operator: Operator, name: str) -> np.ndarray:
    profile_id = operator.profile_workspace.profile_id_for(name)
    slot = operator.profile_workspace.active_slot_for_profile_id(profile_id)
    length = int(operator.profile_workspace.active_lengths[slot])
    return operator.profile_workspace.active_coeff_index_rows[slot, :length]


def _rms_residual(operator: Operator, x: np.ndarray) -> float:
    residual = np.asarray(operator.residual_var(x), dtype=np.float64)
    return float(np.sqrt(np.mean(residual * residual)))


def test_axis_shift_estimate_is_continuous_at_uniform_sources() -> None:
    base = tiny_pf_problem()
    uniform = np.ones_like(base.heat_input)
    uniform_problem = base.replace(heat_input=uniform, current_input=uniform)
    roundoff_problem = base.replace(
        heat_input=uniform,
        current_input=uniform + 1.0e-7 * np.linspace(-1.0, 1.0, uniform.size),
    )
    near_uniform_problem = base.replace(
        heat_input=uniform + 1.0e-3 * np.linspace(-1.0, 1.0, uniform.size),
        current_input=uniform,
    )

    assert estimate_axis_shift_h0(uniform_problem) == pytest.approx(0.0)
    assert estimate_axis_shift_h0(roundoff_problem) == pytest.approx(0.0)
    assert 0.0 < estimate_axis_shift_h0(near_uniform_problem)
    assert estimate_axis_shift_h0(near_uniform_problem) < estimate_axis_shift_h0(base)


def test_geometric_initial_state_delegates_axis_and_boundary_shape_terms() -> None:
    operator = tiny_operator()
    x0 = operator.build_boundary_slope_initial_state()
    coeffs = operator.unpack_coefficients(x0)

    assert coeffs["h"][0] == pytest.approx(estimate_axis_shift_h0(operator.problem))
    assert coeffs["s1"][0] != pytest.approx(0.0)
    assert np.linalg.norm(coeffs["psin"]) > 0.0
    assert coeffs["psin"][0] != pytest.approx(0.0)
    assert coeffs["psin"][1:] == pytest.approx(np.zeros_like(coeffs["psin"][1:]))
    assert abs(float(coeffs["psin"][0])) < 1.0


def test_geometric_initial_state_predicts_active_psin_coefficients() -> None:
    operator = tiny_operator()
    x0 = operator.build_boundary_slope_initial_state()
    psin_indices = _active_profile_coeff_indices(operator, "psin")

    x_without_psin_seed = x0.copy()
    x_without_psin_seed[psin_indices] = 0.0

    assert _rms_residual(operator, x0) < _rms_residual(operator, x_without_psin_seed)


def test_initial_source_psin_target_matches_residual_refresh() -> None:
    operator = tiny_operator()
    x0 = build_boundary_slope_initial_state(
        problem=operator.problem,
        plan=operator.plan,
        profile_workspace=operator.profile_workspace,
        source_psin_target=None,
    )

    operator.residual_var(x0)
    expected = operator.source_workspace.target_root_fields[0].copy()
    actual = operator._source_psin_target_for_initial_state(x0)

    np.testing.assert_allclose(actual, expected)


def test_geometric_initial_state_preserves_positive_psin_radial_derivative() -> None:
    operator = tiny_operator()
    x0 = operator.build_boundary_slope_initial_state()

    operator.stage_a_profile(x0)

    assert float(np.min(operator.profile_workspace.radial_derivative_for("psin"))) > 0.0


@pytest.mark.parametrize("target_factor", [100.0, -10.0])
def test_psin0_source_projection_is_clipped_to_positive_radial_derivative(
    target_factor: float,
) -> None:
    operator = tiny_operator()
    rho = operator.plan.grid_workspace.rho

    def source_psin_target(_: np.ndarray) -> np.ndarray:
        return rho * rho * (1.0 + target_factor * (1.0 - rho * rho))

    x0 = build_boundary_slope_initial_state(
        problem=operator.problem,
        plan=operator.plan,
        profile_workspace=operator.profile_workspace,
        source_psin_target=source_psin_target,
    )

    coeffs = operator.unpack_coefficients(x0)
    operator.stage_a_profile(x0)

    assert abs(float(coeffs["psin"][0])) < 1.0
    assert float(np.min(operator.profile_workspace.radial_derivative_for("psin"))) > 0.0


def test_boundary_shape_terms_are_not_gated_by_uniform_axis_shift() -> None:
    problem = tiny_pf_problem()
    uniform_heat = np.full_like(problem.heat_input, np.mean(np.abs(problem.heat_input)))
    uniform_current = np.full_like(problem.current_input, np.mean(np.abs(problem.current_input)))
    uniform_problem = problem.replace(heat_input=uniform_heat, current_input=uniform_current)
    operator = Operator(tiny_grid(), uniform_problem)

    x0 = operator.build_boundary_slope_initial_state()
    coeffs = operator.unpack_coefficients(x0)

    assert coeffs["h"][0] == pytest.approx(0.0)
    assert coeffs["s1"][0] != pytest.approx(0.0)
