"""VEQPy 2.x fixed-boundary equilibrium Module public facade."""

from __future__ import annotations

import tomllib as _tomllib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path as _Path

from .api import build, solve
from .module import VEQ, VEQRecord

__all__ = ["VEQ", "VEQRecord", "build", "solve"]


def _source_tree_version() -> str:
    """Read the source checkout version when package metadata is unavailable."""

    pyproject_path = _Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        return str(_tomllib.load(stream)["project"]["version"])


try:
    _checkout_version = _source_tree_version()
except (FileNotFoundError, _tomllib.TOMLDecodeError):
    _checkout_version = None

if _checkout_version is not None:
    __version__ = _checkout_version
else:
    try:
        __version__ = _version("veqpy")
    except _PackageNotFoundError:
        __version__ = "0+unknown"


def __dir__() -> list[str]:
    """Return only supported package-root names."""

    return sorted({*__all__, "__version__"})
