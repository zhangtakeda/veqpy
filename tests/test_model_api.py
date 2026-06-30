from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from helpers import tiny_boundary
from numpy.testing import assert_allclose

from veqpy.model import Boundary, Geqdsk, Grid, Problem, Profile

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


def test_boundary_profile_and_problem_normalize_user_inputs() -> None:
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

    heat_input = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    current_input = np.array([0.0, 2.0e6, 3.0e6], dtype=np.float64)
    problem = Problem(
        route="pi",
        coordinate="RHO",
        nodes="UNIFORM",
        active_profiles={"h": 2},
        boundary=tiny_boundary(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )

    heat_input[:] = -1.0
    current_input[:] = -2.0
    assert problem.route == "PI"
    assert problem.coordinate == "rho"
    assert problem.nodes == "uniform"
    assert problem.active_profiles == {"h": 2}
    assert_allclose(problem.heat_input, [1.0e6, 1.2e6, 1.4e6])
    assert_allclose(problem.current_input, [0.0, 2.0e6, 3.0e6])
    assert not problem.heat_input.flags.writeable

    clone = problem.copy()
    assert clone is not problem
    assert_allclose(clone.current_input, problem.current_input)
    assert not np.shares_memory(clone.current_input, problem.current_input)

    with pytest.raises(TypeError, match="length must be int"):
        Problem(
            route="pf",
            coordinate="rho",
            active_profiles={"h": np.zeros(2, dtype=np.float64)},
            boundary=tiny_boundary(),
            heat_input=np.ones(3),
            current_input=np.ones(3),
        )
    with pytest.raises(ValueError, match="heat_input and current_input"):
        Problem(
            route="pf",
            coordinate="rho",
            active_profiles={"h": 1},
            boundary=tiny_boundary(),
            heat_input=np.ones(3),
            current_input=np.ones(2),
        )


def test_boundary_problem_and_geqdsk_roundtrip(tmp_path: Path) -> None:
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

    problem = Problem(
        route="pf",
        coordinate="rho",
        active_profiles={"h": 2},
        boundary=boundary,
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.ones(3, dtype=np.float64),
        Ip=3.0e6,
    )
    problem_path = tmp_path / "problem.json"
    problem.write(problem_path)
    loaded_problem = Problem.load(problem_path)
    assert loaded_problem.route == "PF"
    assert loaded_problem.active_profiles == {"h": 2}
    assert_allclose(loaded_problem.heat_input, problem.heat_input)

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
