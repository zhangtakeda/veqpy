"""
Public VEQlib kernel bridge types.

``KernelTopology`` and ``KernelBuild`` describe compile/artifact identity;
``KernelInput`` and ``KernelSolve`` describe one runtime solve; ``Kernel`` owns
stateful native execution.
"""

from __future__ import annotations

from .kernel import Kernel, build
from .types import KernelBuild, KernelInput, KernelSolve, KernelTopology

__all__ = [
    "Kernel",
    "KernelBuild",
    "KernelInput",
    "KernelSolve",
    "KernelTopology",
    "build",
]
