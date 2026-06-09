"""
Module: operator.__init__

Role:
- Expose the stable operator-layer public API.

Public API:
- Operator
- OperatorCase
- Packed-state naming/layout helpers for preparing coefficient vectors

Notes:
- Build topology lives in ``veqpy.operator.build_plan``; runtime memory and
  executable stage callables live in ``veqpy.workspace`` and ``veqpy.layout``.
- Engine selection, solver driving, and demo orchestration live outside this package.
"""

from __future__ import annotations

from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import (
    PACKED_LAYOUT_PROFILE_FIRST,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
    decode_packed_blocks,
    encode_packed_state,
    get_prefix_profile_names,
    packed_size,
    validate_packed_state,
)

__all__ = [
    "Operator",
    "OperatorCase",
    "PACKED_LAYOUT_PROFILE_FIRST",
    "build_active_profile_metadata",
    "build_fourier_profile_names",
    "build_profile_index",
    "build_profile_layout",
    "build_profile_names",
    "build_shape_profile_names",
    "decode_packed_blocks",
    "encode_packed_state",
    "get_prefix_profile_names",
    "packed_size",
    "validate_packed_state",
]
