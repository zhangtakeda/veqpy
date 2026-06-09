"""
Module: base

Role:
- Expose shared base utilities for serialization, registries, and reactive caching.

Public API:
- Reactive
- depends_on
- Registry
- Serial
- read_serializer
- write_serializer
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------
from veqpy.base.reactive import Reactive, depends_on
from veqpy.base.registry import Registry
from veqpy.base.serial import Serial, read_serializer, write_serializer

__all__ = [
    "Reactive",
    "depends_on",
    "Registry",
    "Serial",
    "read_serializer",
    "write_serializer",
]
