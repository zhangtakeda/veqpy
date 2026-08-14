from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.model import Geqdsk, Grid, Profile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_grid_user_arrays_and_integration() -> None:
    grid = Grid(Nr=6, Nt=8, quadrature_scheme="legendre")
    field = np.ones((grid.Nr, grid.Nt), dtype=np.float64)

    assert grid.r.shape == (6,)
    assert grid.theta.shape == (8,)
    radial_coordinate, poloidal_coordinate = grid.coordinates
    assert_allclose(radial_coordinate, grid.r)
    assert_allclose(poloidal_coordinate, grid.theta)
    assert grid.weights.shape == (6,)
    assert_allclose(float(np.sum(grid.weights)), 1.0)
    assert not grid.r.flags.writeable
    with pytest.raises(AttributeError):
        _ = grid.rho
    assert_allclose(grid.integrate(np.ones(grid.Nr)), 1.0)
    assert_allclose(grid.integrate(field), 2.0 * np.pi)

    with pytest.raises(ValueError, match="Nr must be at least 4"):
        Grid(Nr=3, Nt=8)
    with pytest.raises(ValueError, match="Expected a 1D or 2D array"):
        grid.integrate(np.ones((2, 2, 2)))


def test_profile_normalizes_user_inputs() -> None:
    coeff = np.array([1.0, 2.0], dtype=np.float64)
    profile = Profile(scale=2, power=3.0, envelope_power=1.0, coeff=coeff)
    coeff[:] = -1.0

    assert profile.scale == 2.0
    assert profile.power == 3
    assert_allclose(profile.coeff, [1.0, 2.0])
    assert not profile.coeff.flags.writeable
    with pytest.raises(ValueError, match="coeff must be 1D"):
        Profile(coeff=np.ones((1, 1)))


def test_profile_fields_are_reactive_when_grid_is_bound() -> None:
    grid = Grid(Nr=6, Nt=8, quadrature_scheme="legendre")
    profile = Profile(scale=3.0, power=2, envelope_power=0, offset=2.0)

    with pytest.raises(RuntimeError, match="Profile.grid is required"):
        _ = profile.value

    profile.grid = grid
    assert_allclose(profile.value, 6.0 * grid.r**2)
    assert_allclose(profile.derivative, 12.0 * grid.r)
    assert_allclose(profile.second_derivative, np.full(grid.Nr, 12.0))
    assert profile.fields.shape == (3, grid.Nr)

    profile.scale = 4.0
    assert_allclose(profile.value, 8.0 * grid.r**2)


def test_geqdsk_roundtrip(tmp_path: Path) -> None:
    source = Geqdsk(DATA_DIR / "SOLOVEV.geqdsk")
    source.check()
    geqdsk_path = tmp_path / "SOLOVEV.geqdsk"
    source.write(geqdsk_path)
    restored = Geqdsk(geqdsk_path)
    restored.check()
    assert restored.NR == source.NR
    assert restored.NZ == source.NZ
    assert_allclose(restored.F, source.F, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.psi, source.psi, rtol=1e-8, atol=1e-8)

    invalid = Geqdsk(NR=2, NZ=2, F=[1.0], P=[1.0], FF_psi=[1.0], P_psi=[1.0], q=[1.0])
    with pytest.raises(ValueError, match="F must have length"):
        invalid.check()
