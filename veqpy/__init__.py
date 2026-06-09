"""
Module: veqpy

Role:
- Define package metadata and top-level package exports.

Public API:
- engine
- model
- operator
- solver

Notes:
- Package roots are the only modules that declare ``__all__``.
- Subpackages own their narrower public export surfaces.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_tree_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


try:
    __version__ = version("veqpy")
except PackageNotFoundError:
    __version__ = _source_tree_version()

if __name__ == "__main__":
    print(
        f"VEQPy-v{__version__}:\n"
        "a fast parametric Grad--Shafranov solver for \n"
        "fixed-boundary, axisymmetric tokamak equilibria"
    )
