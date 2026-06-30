from __future__ import annotations

import tomllib
from pathlib import Path

import veqpy
import veqpy.model as model
import veqpy.operator as operator
import veqpy.solver as solver


def test_package_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]

    assert veqpy.__version__ == expected


def test_core_public_imports_are_available() -> None:
    assert model.Grid
    assert model.Boundary
    assert model.Profile
    assert model.Problem
    assert model.Geqdsk
    assert model.Equilibrium
    assert operator.Operator
    assert solver.Solver
    assert solver.SolverConfig
    assert solver.SolverResult
