from __future__ import annotations

import tomllib
from pathlib import Path

import veqpy
import veqpy.model as model
import veqpy.operator as operator
import veqpy.solver as solver
from veqpy.model import Equilibrium


def test_package_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]
    assert veqpy.__version__ == expected


def test_model_public_exports_are_stable() -> None:
    assert model.__all__ == [
        "Equilibrium",
        "Grid",
        "Geqdsk",
        "Boundary",
        "Profile",
        "Problem",
        "Reactive",
        "Serial",
    ]
    assert not hasattr(Equilibrium, "compare")


def test_operator_public_exports_are_stable() -> None:
    assert operator.__all__ == [
        "Operator",
        "PACKED_LAYOUT_PROFILE_FIRST",
        "build_active_profile_metadata",
        "build_fourier_profile_names",
        "build_profile_index",
        "build_profile_layout",
        "build_profile_names",
        "build_shape_profile_names",
        "decode_packed_blocks",
        "encode_packed_state",
        "get_prefix_profile_names",
        "packed_size",
        "validate_packed_state",
    ]


def test_solver_public_exports_are_stable() -> None:
    assert solver.__all__ == [
        "Solver",
        "SolverConfig",
        "SolverRecord",
        "SolverResult",
    ]
