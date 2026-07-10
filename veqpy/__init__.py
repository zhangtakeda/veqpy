"""
Package: veqpy

Role:
- Expose VEQPy's canonical base, Kernel, and model surfaces from one package root.

Public API:
- Reactive, serialization, and registry infrastructure.
- build, fit, pareto, and solve function-style entrypoints.
- Kernel and KernelRecipe.
- KernelTopology, KernelBoundary, KernelSource, and KernelConfig.
- SolveResult, ParetoResult, and ParetoSample result records.
- Grid, Profile, Geqdsk, and Equilibrium model objects.

Dependencies:
- veqpy.api for function-style entrypoints.
- veqpy.base for shared reactive and serialization infrastructure.
- veqpy.kernels for Kernel dispatch, public Kernel dataclasses, and errors.
- veqpy.model for equilibrium data models.

Downstream:
- Examples, benchmarks, docs, and user code can import the public contract from
  this package root.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from veqpy.api import build, fit, pareto, solve
from veqpy.base import (
    Reactive,
    Registry,
    Serial,
    depends_on,
    read_serializer,
    write_serializer,
)
from veqpy.kernels import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    ParetoResult,
    ParetoSample,
    SolveResult,
)
from veqpy.model import (
    Equilibrium,
    Geqdsk,
    Grid,
    Profile,
)

__all__ = [
    "Reactive",
    "Registry",
    "Serial",
    "depends_on",
    "read_serializer",
    "write_serializer",
    "build",
    "fit",
    "pareto",
    "solve",
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "ParetoResult",
    "ParetoSample",
    "SolveResult",
    "Equilibrium",
    "Geqdsk",
    "Grid",
    "Profile",
]


def _source_tree_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


try:
    __version__ = version("veqpy")
except PackageNotFoundError:
    __version__ = _source_tree_version()


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if __name__ == "__main__":
    print(
        f"VEQPy-v{__version__}:\n"
        "a fast parametric Grad--Shafranov solver for \n"
        "fixed-boundary, axisymmetric tokamak equilibria"
    )
