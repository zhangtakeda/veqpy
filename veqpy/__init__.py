"""VEQPy 2.x equilibrium Module and its four-buffer numerical Kernel."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from veqpy.kernels import Kernel, KernelConfig, KernelInput, KernelOutput, KernelTopology
from veqpy.model.geqdsk import Geqdsk
from veqpy.module import VEQ, VEQRecord

__all__ = [
    "Geqdsk",
    "Kernel",
    "KernelConfig",
    "KernelInput",
    "KernelOutput",
    "KernelTopology",
    "VEQ",
    "VEQRecord",
]


def _source_tree_version() -> str:
    """Read the source checkout version when package metadata is unavailable."""

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


try:
    _checkout_version = _source_tree_version()
except (FileNotFoundError, tomllib.TOMLDecodeError):
    _checkout_version = None

if _checkout_version is not None:
    __version__ = _checkout_version
else:
    try:
        __version__ = version("veqpy")
    except PackageNotFoundError:
        __version__ = "0+unknown"


def __dir__() -> list[str]:
    """Return only the supported package-root names."""

    return sorted({*globals(), *__all__})


if __name__ == "__main__":
    print(f"VEQPy-v{__version__}: fixed-boundary axisymmetric equilibrium Module")
