from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.model import Boundary, Geqdsk, Grid, Profile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_grid_user_arrays_and_integration() -> None:
    grid = Grid(Nr=6, Nt=8, quadrature_scheme="legendre")
    field = np.ones((grid.Nr, grid.Nt), dtype=np.float64)

    assert grid.rho.shape == (6,)
    assert grid.theta.shape == (8,)
    assert grid.weights.shape == (6,)
    assert_allclose(float(np.sum(grid.weights)), 1.0)
    assert not grid.rho.flags.writeable
    assert_allclose(grid.integrate(np.ones(grid.Nr)), 1.0)
    assert_allclose(grid.integrate(field), 2.0 * np.pi)

    with pytest.raises(ValueError, match="Nr must be at least 4"):
        Grid(Nr=3, Nt=8)
    with pytest.raises(ValueError, match="Expected a 1D or 2D array"):
        grid.integrate(np.ones((2, 2, 2)))


def test_boundary_and_profile_normalize_user_inputs() -> None:
    coeff = np.array([1.0, 2.0], dtype=np.float64)
    profile = Profile(scale=2, power=3.0, envelope_power=1.0, coeff=coeff)
    coeff[:] = -1.0

    assert profile.scale == 2.0
    assert profile.power == 3
    assert_allclose(profile.coeff, [1.0, 2.0])
    assert not profile.coeff.flags.writeable
    with pytest.raises(FrozenInstanceError):
        profile.scale = 3.0
    with pytest.raises(ValueError, match="coeff must be 1D"):
        Profile(coeff=np.ones((1, 1)))

    boundary = Boundary(a=1, R0=2, Z0=0, B0=3, c_offsets=[0.1], s_offsets=[9.0, 0.3])
    assert boundary.a == 1.0
    assert_allclose(boundary.s_offsets, [0.0, 0.3])


def test_boundary_and_geqdsk_roundtrip(tmp_path: Path) -> None:
    boundary = Boundary(
        a=0.5,
        R0=1.2,
        Z0=-0.1,
        B0=2.5,
        ka=1.6,
        c_offsets=np.array([0.1, 0.2], dtype=np.float64),
        s_offsets=np.array([0.0, -0.2], dtype=np.float64),
    )
    boundary_path = tmp_path / "boundary.json"
    boundary.write(boundary_path)
    loaded_boundary = Boundary.load(boundary_path)
    assert loaded_boundary.a == boundary.a
    assert_allclose(loaded_boundary.c_offsets, boundary.c_offsets)

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
