"""Public Kernel wrapper and private backend packages."""

from __future__ import annotations

from importlib import import_module

__all__ = ["Kernel"]


def __getattr__(name: str) -> object:
    if name != "Kernel":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("veqpy.kernels.kernel"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
