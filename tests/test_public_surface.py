from __future__ import annotations

import tomllib
from pathlib import Path

import veqpy
import veqpy.model as model


def test_package_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]

    assert veqpy.__version__ == expected


def test_core_public_imports_are_available() -> None:
    assert model.Grid
    assert model.Profile
    assert model.Geqdsk
    assert model.Equilibrium
    assert veqpy.Reactive
    assert veqpy.Registry
    assert veqpy.Serial
    assert veqpy.depends_on
    assert veqpy.read_serializer
    assert veqpy.write_serializer
    assert veqpy.Grid
    assert veqpy.Profile
    assert veqpy.Geqdsk
    assert veqpy.Equilibrium
    assert veqpy.Kernel
    assert veqpy.KernelRecipe
    assert veqpy.KernelTopology
    assert veqpy.KernelBoundary
    assert veqpy.KernelSource
    assert veqpy.ParetoResult
    assert veqpy.ParetoSample
    assert veqpy.fit
    assert veqpy.pareto
