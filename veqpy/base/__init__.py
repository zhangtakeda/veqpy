"""
Module: base

Role:
- Expose shared base utilities for serialization, registries, and reactive caching.

Public API:
- Reactive
- depends_on
- Registry
- Serial
- SERIAL_TYPE_REGISTRY
- read_serializer
- write_serializer
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
