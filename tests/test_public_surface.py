from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import veqpy


def test_package_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]
    assert veqpy.__version__ == expected


def test_public_kernel_surface_is_four_buffer_contract() -> None:
    assert set(veqpy.kernels.__all__) == {
        "Kernel",
        "KernelTopology",
        "KernelInput",
        "KernelConfig",
        "KernelOutput",
    }
    assert set(veqpy.__all__) == {
        "Geqdsk",
        "Kernel",
        "KernelTopology",
        "KernelInput",
        "KernelConfig",
        "KernelOutput",
        "VEQ",
        "VEQRecord",
    }
    assert set(veqpy.model.__all__) == {"Geqdsk"}


def test_function_entrypoints_match_current_api() -> None:
    api = importlib.import_module("veqpy.api")
    assert api.__all__ == ["build", "solve"]
    assert api.build is not None
    assert api.solve is not None


def test_core_import_does_not_load_matplotlib() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys, veqpy; assert 'matplotlib' not in sys.modules"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_internal_model_modules_are_not_reexported() -> None:
    model = importlib.import_module("veqpy.model")
    assert not hasattr(model, "Grid")
    assert not hasattr(model, "Profile")
    assert not hasattr(model, "Equilibrium")
