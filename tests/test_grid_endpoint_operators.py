from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from veqpy import Equilibrium, Grid
from veqpy.kernels.numba_kernel.numba_source import _update_psin_coordinate


def test_gauss_grid_endpoint_operators_do_not_alias_interior_nodes() -> None:
    grid = Grid(Nr=12, Nt=8, quadrature_scheme="legendre")
    rho = grid.rho
    profile = 2.0 - 3.0 * rho + 5.0 * rho**2 - 7.0 * rho**3

    assert 0.0 < rho[0] < rho[-1] < 1.0
    assert_allclose(grid.axis_eval(profile), 2.0, rtol=0.0, atol=2.0e-13)
    assert_allclose(grid.edge_eval(profile), -3.0, rtol=0.0, atol=2.0e-13)
    assert_allclose(grid.full_integral(profile), 5.0 / 12.0, rtol=0.0, atol=2.0e-13)
    assert abs(profile[0] - grid.axis_eval(profile)) > 1.0e-3
    assert abs(profile[-1] - grid.edge_eval(profile)) > 1.0e-3


def test_psin_coordinate_retains_gauss_node_values_and_true_endpoints() -> None:
    grid = Grid(Nr=12, Nt=8, quadrature_scheme="legendre")
    out = np.empty(grid.Nr, dtype=np.float64)
    psin_r = 2.0 * grid.rho

    _update_psin_coordinate(out, psin_r, grid.accumulator, grid.weights)

    assert_allclose(out, grid.rho**2, rtol=0.0, atol=5.0e-13)
    assert 0.0 < out[0] < out[-1] < 1.0
    assert_allclose(grid.axis_eval(out), 0.0, rtol=0.0, atol=5.0e-13)
    assert_allclose(grid.edge_eval(out), 1.0, rtol=0.0, atol=5.0e-13)


def test_equilibrium_resample_augments_physical_axis_and_edge() -> None:
    source_grid = Grid(Nr=12, Nt=8, quadrature_scheme="legendre")
    rho = source_grid.rho
    equilibrium = Equilibrium(
        R0=3.0,
        Z0=0.0,
        B0=2.0,
        a=1.0,
        grid=source_grid,
        shape_profiles={},
        psin=rho**2,
        psin_r=2.0 * rho,
        psin_rr=np.full_like(rho, 2.0),
        FFn_psin=2.0 + 3.0 * rho,
        Pn_psin=-1.0 + 2.0 * rho,
    )
    target_grid = Grid(Nr=15, Nt=8, quadrature_scheme="uniform")

    resampled = equilibrium.resample(target_grid)

    assert_allclose(resampled.psin, target_grid.rho**2, rtol=0.0, atol=5.0e-3)
    assert_allclose(resampled.psin[[0, -1]], [0.0, 1.0], rtol=0.0, atol=1.0e-14)
    assert_allclose(resampled.psin_r, 2.0 * target_grid.rho, rtol=0.0, atol=2.0e-13)
    assert_allclose(resampled.FFn_psin, 2.0 + 3.0 * target_grid.rho, atol=2.0e-13)
    assert_allclose(resampled.Pn_psin, -1.0 + 2.0 * target_grid.rho, atol=2.0e-13)
