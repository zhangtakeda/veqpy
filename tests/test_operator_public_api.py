from __future__ import annotations

import numpy as np
import pytest
from helpers import MU0, pf_reference_profiles, tiny_boundary, tiny_grid, tiny_operator
from numpy.testing import assert_allclose

from veqpy.engine.numba_source import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL as ENGINE_LOCAL_BARYCENTRIC_STENCIL,
)
from veqpy.engine.numba_source import (
    _dense_solve_one_rhs_inplace,
    _dense_solve_two_rhs_inplace,
)
from veqpy.math import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    SOURCE_INTERP_DEFAULT,
    normalize_source_interpolation_kind,
)
from veqpy.model import Grid
from veqpy.operator import Operator, OperatorCase


def test_operator_residual_interfaces_and_in_place_outputs() -> None:
    operator = tiny_operator()
    x0 = operator.encode_initial_state()

    assert x0.shape == (operator.x_size,)
    assert operator.active_profile_ids.ndim == 1
    assert "h" in operator.profile_names

    residual = operator.residual_var(x0)
    assert residual.shape == (operator.x_size,)
    assert np.all(np.isfinite(residual))
    assert_allclose(operator.residual_var(x0.tolist()), residual)

    out = np.empty_like(residual)
    operator.residual_var_into(x0, out)
    assert_allclose(out, residual)

    unchecked_residual = operator.residual_var(x0, check=False)
    assert_allclose(unchecked_residual, residual)
    unchecked_out = np.empty_like(residual)
    operator.residual_var_into(x0, unchecked_out, check=False)
    assert_allclose(unchecked_out, residual)

    collocation = operator.residual_collocation(x0)
    assert collocation.shape == (
        operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt,
    )
    assert np.all(np.isfinite(collocation))
    assert_allclose(operator.residual_collocation(x0.tolist()), collocation)

    collocation_out = np.empty_like(collocation)
    operator.residual_collocation_into(x0, collocation_out)
    assert_allclose(collocation_out, collocation)

    unchecked_collocation = operator.residual_collocation(x0, check=False)
    assert_allclose(unchecked_collocation, collocation)
    unchecked_collocation_out = np.empty_like(collocation)
    operator.residual_collocation_into(x0, unchecked_collocation_out, check=False)
    assert_allclose(unchecked_collocation_out, collocation)

    residual_stage = operator.stage_d_residual()
    assert residual_stage.shape == (operator.x_size,)


def test_fused_residual_matches_explicit_stage_chain() -> None:
    fused_operator = tiny_operator()
    fused_x = fused_operator.encode_initial_state()
    fused = fused_operator.residual_var(fused_x)

    staged_operator = tiny_operator()
    staged_x = staged_operator.encode_initial_state()
    staged_operator.stage_a_profile(staged_x)
    staged_operator.stage_b_geometry()
    staged_operator.stage_c_source()
    staged = staged_operator.stage_d_residual()

    assert_allclose(staged, fused)


def test_source_interpolation_default_is_shared() -> None:
    assert normalize_source_interpolation_kind(None) == SOURCE_INTERP_DEFAULT
    source_interpolation_field = Operator.__dataclass_fields__["source_interpolation_kind"]
    assert source_interpolation_field.default == SOURCE_INTERP_DEFAULT
    assert ENGINE_LOCAL_BARYCENTRIC_STENCIL == DEFAULT_LOCAL_BARYCENTRIC_STENCIL


def test_operator_validation_and_snapshot_helpers() -> None:
    operator = tiny_operator()
    x0 = operator.encode_initial_state()

    with pytest.raises(ValueError, match="Expected x to have shape"):
        operator.coerce_x(np.zeros(operator.x_size + 1))
    with pytest.raises(TypeError, match="dtype float64"):
        operator.residual_var_into(x0, np.empty(operator.x_size, dtype=np.float32))
    with pytest.raises(TypeError, match="dtype float64"):
        collocation_size = operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt
        operator.residual_collocation_into(x0, np.empty(collocation_size, dtype=np.float32))
    noncontiguous = np.empty(operator.x_size * 2, dtype=np.float64)[::2]
    noncontiguous[:] = x0
    assert not noncontiguous.flags.c_contiguous
    assert operator.coerce_x(noncontiguous).flags.c_contiguous

    coeffs = operator.build_coeffs(x0, include_none=False)
    assert set(coeffs) == {"h", "k", "s1", "psin"}

    equilibrium = operator.build_equilibrium(x0)
    assert equilibrium.grid.Nr == operator.plan.grid_workspace.Nr
    assert np.isfinite(equilibrium.Ip)


def test_pf_rho_unconstrained_cases_use_positive_flux_branch() -> None:
    grid = tiny_grid()
    rho = np.asarray(grid.rho, dtype=np.float64)
    ffn_psin, pn_psin = pf_reference_profiles(rho * rho)
    common_kwargs = {
        "route": "PF",
        "coordinate": "rho",
        "nodes": "grid",
        "profile_coeffs": {"h": [0.0], "k": [0.0], "s1": [0.0]},
        "boundary": tiny_boundary(),
        "current_input": ffn_psin * (2.0 * rho),
    }

    null_case = OperatorCase(
        **common_kwargs,
        heat_input=pn_psin * (2.0 * rho) / MU0,
    )
    null_operator = Operator(grid, null_case)
    null_eq = null_operator.build_equilibrium(null_operator.encode_initial_state())

    beta_case = OperatorCase(
        **common_kwargs,
        heat_input=pn_psin * (2.0 * rho) / MU0,
        beta=float(null_eq.beta_t),
    )
    beta_operator = Operator(grid, beta_case)
    beta_eq = beta_operator.build_equilibrium(beta_operator.encode_initial_state())

    assert null_eq.alpha2 > 0.0
    assert beta_eq.alpha2 > 0.0


def test_active_f_profile_is_only_supported_by_pj2() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    with pytest.raises(ValueError, match="active F profile.*only supported for PJ2"):
        Operator(
            tiny_grid(),
            OperatorCase(
                route="PF",
                coordinate="rho",
                nodes="uniform",
                profile_coeffs={
                    "h": [0.0, 0.0],
                    "k": [0.0, 0.0],
                    "s1": [0.0, 0.0],
                    "F": [0.0, 0.0],
                },
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.ones_like(rho),
            ),
        )


def test_pj2_requires_active_f_profile() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    with pytest.raises(ValueError, match="PJ2 requires an active F profile"):
        Operator(
            tiny_grid(),
            OperatorCase(
                route="PJ2",
                coordinate="rho",
                nodes="uniform",
                profile_coeffs={
                    "h": [0.0, 0.0],
                    "k": [0.0, 0.0],
                    "s1": [0.0, 0.0],
                },
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.full_like(rho, 1.0e6),
            ),
        )


def test_rho_routes_reject_active_psin_profile() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    with pytest.raises(ValueError, match="rho/uniform does not accept an active psin profile"):
        Operator(
            tiny_grid(),
            OperatorCase(
                route="PF",
                coordinate="rho",
                nodes="uniform",
                profile_coeffs={
                    "h": [0.0, 0.0],
                    "k": [0.0, 0.0],
                    "s1": [0.0, 0.0],
                    "psin": [0.0, 0.0],
                },
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.ones_like(rho),
            ),
        )


def test_psin_grid_routes_reject_active_psin_profile() -> None:
    grid = tiny_grid()
    with pytest.raises(ValueError, match="psin/grid does not accept an active psin profile"):
        Operator(
            grid,
            OperatorCase(
                route="PF",
                coordinate="psin",
                nodes="grid",
                profile_coeffs={
                    "h": [0.0, 0.0],
                    "k": [0.0, 0.0],
                    "s1": [0.0, 0.0],
                    "psin": [0.0, 0.0],
                },
                boundary=tiny_boundary(),
                heat_input=np.full(grid.Nr, 1.0e6),
                current_input=np.ones(grid.Nr),
            ),
        )


def test_active_f_and_psin_profiles_are_mutually_exclusive() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    with pytest.raises(ValueError, match="F and active psin profiles are mutually exclusive"):
        Operator(
            tiny_grid(),
            OperatorCase(
                route="PJ2",
                coordinate="rho",
                nodes="uniform",
                profile_coeffs={
                    "h": [0.0, 0.0],
                    "k": [0.0, 0.0],
                    "s1": [0.0, 0.0],
                    "psin": [0.0, 0.0],
                    "F": [0.0, 0.0],
                },
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.full_like(rho, 1.0e6),
            ),
        )


def test_replace_case_rejects_active_f_on_non_pj2_route() -> None:
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    profile_coeffs = {
        "h": [0.0, 0.0],
        "k": [0.0, 0.0],
        "s1": [0.0, 0.0],
        "F": [0.0, 0.0],
    }
    operator = Operator(
        tiny_grid(),
        OperatorCase(
            route="PJ2",
            coordinate="rho",
            nodes="uniform",
            profile_coeffs=profile_coeffs,
            boundary=tiny_boundary(),
            heat_input=np.full_like(rho, 1.0e6),
            current_input=np.full_like(rho, 1.0e6),
        ),
    )

    with pytest.raises(ValueError, match="active F profile.*only supported for PJ2"):
        operator.replace_case(
            OperatorCase(
                route="PF",
                coordinate="rho",
                nodes="uniform",
                profile_coeffs=profile_coeffs,
                boundary=tiny_boundary(),
                heat_input=np.full_like(rho, 1.0e6),
                current_input=np.ones_like(rho),
            )
        )


def test_pj2_uses_profile_f_derivative_for_active_f_profile() -> None:
    grid = Grid(Nr=8, Nt=8, L_max=3, M_max=1, K_max=1, quadrature_scheme="legendre")
    case = OperatorCase(
        route="PJ2",
        coordinate="rho",
        nodes="grid",
        profile_coeffs={
            "h": [0.0, 0.0],
            "k": [0.0, 0.0],
            "s1": [0.0, 0.0],
            "F": [0.2, -0.1, 0.05, 0.03],
        },
        boundary=tiny_boundary(),
        heat_input=np.full(grid.Nr, 1.0e6),
        current_input=np.full(grid.Nr, 1.0e6),
    )
    operator = Operator(grid, case)

    operator.residual_var(operator.encode_initial_state())

    root_fields = operator.residual_workspace.root_fields
    f_fields = operator.profile_workspace.fields_for("F")
    psin_r = root_fields[1]
    ffn_psin = root_fields[3]
    expected_from_profile = f_fields[0] * f_fields[1] / (operator.alpha1 * operator.alpha2 * psin_r)
    matrix_f_r = operator.plan.grid_workspace.differentiator @ f_fields[0]
    expected_from_matrix = f_fields[0] * matrix_f_r / (operator.alpha1 * operator.alpha2 * psin_r)

    first_unregularized = int(np.searchsorted(operator.plan.grid_workspace.rho, operator.fix_rho))
    assert_allclose(
        ffn_psin[first_unregularized:],
        expected_from_profile[first_unregularized:],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.max(np.abs(ffn_psin - expected_from_matrix)) > 1.0e-3


def test_pq_dense_two_rhs_solve_matches_two_one_rhs_solves() -> None:
    rng = np.random.default_rng(20240611)
    n = 6
    A = rng.normal(size=(n, n))
    A += np.eye(n) * 5.0
    b0 = rng.normal(size=n)
    b1 = rng.normal(size=n)

    A0 = np.ascontiguousarray(A.copy())
    A1 = np.ascontiguousarray(A.copy())
    x0_expected = np.ascontiguousarray(b0.copy())
    x1_expected = np.ascontiguousarray(b1.copy())
    _dense_solve_one_rhs_inplace(A0, x0_expected, n, 1.0e-12)
    _dense_solve_one_rhs_inplace(A1, x1_expected, n, 1.0e-12)

    A2 = np.ascontiguousarray(A.copy())
    x0_actual = np.ascontiguousarray(b0.copy())
    x1_actual = np.ascontiguousarray(b1.copy())
    _dense_solve_two_rhs_inplace(A2, x0_actual, x1_actual, n, 1.0e-12)

    assert_allclose(x0_actual, x0_expected, rtol=0.0, atol=1.0e-14)
    assert_allclose(x1_actual, x1_expected, rtol=0.0, atol=1.0e-14)
