"""
Package: veqpy.base

Role:
- Provide infrastructure shared by model objects, numerical helpers, and Kernel code.
- Own serialization, registry, and reactive-cache primitives.

Public API:
- Reactive and depends_on.
- Registry.
- Serial, SERIAL_TYPE_REGISTRY, read_serializer, and write_serializer.

Dependencies:
- Python standard-library modules and NumPy where needed by concrete files.

Downstream:
- veqpy.model uses Reactive and Serial for model-layer objects.
- veqpy.numerics and veqpy.kernels use Registry for keyed helper dispatch.

Design notes:
- This package is the bottom layer of the Python dependency graph.
- It should not import model, numerics, or kernels modules.
"""

from __future__ import annotations

from .reactive import Reactive, depends_on
from .registry import Registry
from .serial import SERIAL_TYPE_REGISTRY, Serial, read_serializer, write_serializer

__all__ = [
    "Reactive",
    "depends_on",
    "Registry",
    "Serial",
    "SERIAL_TYPE_REGISTRY",
    "read_serializer",
    "write_serializer",
]
