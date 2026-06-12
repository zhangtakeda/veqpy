from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.engine import RHO_AXIS, THETA_AXIS
from veqpy.math import DEFAULT_CALCULUS, DEFAULT_QUADRATURE
from veqpy.model import Grid


def test_grid_shapes_quadrature_and_read_only_arrays() -> None:
    grid = Grid(Nr=6, Nt=8, L_max=3, M_max=2, K_max=1, quadrature_scheme="legendre")

    assert grid.rho.shape == (6,)
    assert grid.weights.shape == (6,)
    assert grid.theta.shape == (8,)
    assert grid.T_fields.shape == (3, 4, 6)
    assert grid.cos_mtheta.shape == (3, 8)
    assert grid.K_values.tolist() == [0, 1, 1]
    assert_allclose(float(np.sum(grid.weights)), 1.0)

    assert not grid.rho.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        grid.rho[0] = 0.5


def test_grid_defaults_are_shared_with_math_layer() -> None:
    grid = Grid(Nr=6, Nt=8)

    assert grid.quadrature_scheme == DEFAULT_QUADRATURE
    assert grid.calculus_scheme == DEFAULT_CALCULUS


def test_grid_integrates_radial_poloidal_and_full_fields() -> None:
    grid = Grid(Nr=6, Nt=8)
    field = np.ones((grid.Nr, grid.Nt), dtype=np.float64)

    assert_allclose(grid.integrate(np.ones(grid.Nr)), 1.0)
    assert_allclose(grid.integrate(field, axis=RHO_AXIS), np.ones(grid.Nt))
    assert_allclose(grid.integrate(field, axis=THETA_AXIS), np.full(grid.Nr, 2.0 * np.pi))
    assert_allclose(grid.integrate(field), 2.0 * np.pi)

    with pytest.raises(ValueError, match="Unsupported quadrature axis"):
        grid.integrate(field, axis=99)
    with pytest.raises(ValueError, match="Expected a 1D or 2D array"):
        grid.integrate(np.ones((2, 2, 2)))


def test_grid_validation_errors() -> None:
    with pytest.raises(ValueError, match="Nr must be at least 4"):
        Grid(Nr=3, Nt=8)
    with pytest.raises(ValueError, match="Nt must be at least 4"):
        Grid(Nr=4, Nt=3)
    with pytest.raises(ValueError, match="L_max must be non-negative"):
        Grid(Nr=4, Nt=4, L_max=-1)
    with pytest.raises(ValueError, match="K_max must be non-negative"):
        Grid(Nr=4, Nt=4, K_max=-1)
