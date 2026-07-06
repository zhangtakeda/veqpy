"""VEQPy public package root."""

from __future__ import annotations

import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from veqpy.types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
    TopologyError,
)

_LAZY_EXPORTS = {
    "Kernel": ("veqpy.kernels", "Kernel"),
    "build": ("veqpy.api", "build"),
    "solve": ("veqpy.api", "solve"),
}

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


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if __name__ == "__main__":
    print(
        f"VEQPy-v{__version__}:\n"
        "a fast parametric Grad--Shafranov solver for \n"
        "fixed-boundary, axisymmetric tokamak equilibria"
    )
