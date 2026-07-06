"""VEQPy public package root."""

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
