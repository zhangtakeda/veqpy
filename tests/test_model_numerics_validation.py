from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.model import Grid, Profile
from veqpy.numerics import (
    apply_differentiation,
    build_uniform_source_interpolation_matrix,
    interpolation_matrix,
    make_calculus,
    make_quadrature,
)


@pytest.mark.parametrize("scheme", ("legendre", "lobatto", "radau", "chebyshev", "uniform"))
def test_unit_interval_quadrature_integrates_low_order_polynomials(scheme: str) -> None:
    nodes, weights = make_quadrature(16, scheme=scheme)

    assert nodes.shape == weights.shape == (16,)
    assert np.all(np.isfinite(nodes))
    assert np.all(np.isfinite(weights))
    assert_allclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-13)
    assert_allclose(np.sum(weights * nodes), 0.5, rtol=0.0, atol=1.0e-12)
    assert_allclose(np.sum(weights * nodes**2), 1.0 / 3.0, rtol=0.0, atol=2.0e-3)


@pytest.mark.parametrize("scheme", ("spectral", "compact", "cfd33", "cfd35", "cfd55"))
def test_calculus_differentiates_smooth_radial_profile(scheme: str) -> None:
    nodes = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    _, differentiator = make_calculus(nodes, scheme=scheme)
    profile = nodes**4 - 0.5 * nodes**2 + 2.0
    expected = 4.0 * nodes**3 - nodes
    out = np.empty_like(nodes)

    apply_differentiation(out, profile, differentiator)

    assert_allclose(out[2:-2], expected[2:-2], rtol=0.0, atol=2.0e-3)


def test_barycentric_interpolation_matrix_reproduces_polynomial_samples() -> None:
    source = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    target = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    values = 1.0 - 2.0 * source + 3.0 * source**2 - source**3
    expected = 1.0 - 2.0 * target + 3.0 * target**2 - target**3

    matrix = interpolation_matrix(source, target)

    assert_allclose(matrix @ values, expected, rtol=0.0, atol=1.0e-12)
    assert_allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=1.0e-13)


@pytest.mark.parametrize("kind", ("barycentric", "linear", "quadratic", "cubic", "not-a-knot"))
def test_uniform_source_interpolation_matrix_preserves_constant_profiles(kind: str) -> None:
    matrix = build_uniform_source_interpolation_matrix(np.linspace(0.0, 1.0, 19), 11, kind=kind)

    assert matrix.shape == (19, 11)
    assert_allclose(matrix @ np.ones(11, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12)


def test_grid_and_profile_reactive_fields_match_analytic_derivatives() -> None:
    grid = Grid(Nr=20, Nt=16, quadrature_scheme="legendre")
    profile = Profile(scale=2.5, power=3, envelope_power=0, offset=1.0, grid=grid)

    assert_allclose(profile.value, 2.5 * grid.rho**3)
    assert_allclose(profile.derivative, 7.5 * grid.rho**2)
    assert_allclose(profile.second_derivative, 15.0 * grid.rho)
    assert_allclose(profile.fields[0], profile.value)
    assert_allclose(profile.fields[1], profile.derivative)
    assert_allclose(profile.fields[2], profile.second_derivative)
