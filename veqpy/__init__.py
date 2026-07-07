"""
Package: veqpy

Role:
- Expose the canonical public Kernel surface for VEQPy users.
- Keep package-root imports focused on build/solve entrypoints and Kernel data types.

Public API:
- Kernel and KernelRecipe.
- KernelTopology, KernelBoundary, KernelSource, and KernelConfig.
- KernelPrepareResult, SolveResult, and TopologyError.
- build and solve function-style entrypoints.

Dependencies:
- veqpy.api for function-style entrypoints.
- veqpy.kernels for Kernel dispatch, public Kernel dataclasses, and errors.

Downstream:
- Examples, benchmarks, docs, and user code should import the public Kernel
  contract from this package root when possible.

Design notes:
- Backend implementation classes remain private to veqpy.kernels.
- Model objects are exported from veqpy.model rather than this root.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from veqpy.api import build, solve
from veqpy.kernels import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    TopologyError,
)

__all__ = [
    "Kernel",
    "KernelBoundary",
    "KernelConfig",
    "KernelPrepareResult",
    "KernelRecipe",
    "KernelSource",
    "KernelTopology",
    "SolveResult",
    "TopologyError",
    "build",
    "solve",
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
