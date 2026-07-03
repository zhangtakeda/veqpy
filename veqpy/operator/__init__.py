"""
Module: operator.__init__

Role:
- Expose the stable operator-layer public API.

Public API:
- Operator
- Packed-state naming/layout helpers for preparing coefficient vectors
- Build/source plan types used by peer submodules

Notes:
- Build topology lives in ``veqpy.operator.build_plan``; runtime memory and
  executable stage callables live in ``veqpy.workspace`` and ``veqpy.layout``.
- Engine selection, solver driving, and demo orchestration live outside this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Operator",
    "OperatorBuildPlan",
    "PACKED_LAYOUT_PROFILE_FIRST",
    "PROFILE_OFFSET_SPECS",
    "PROFILE_STATIC_KWARGS",
    "ResidualBindingLayout",
    "SourcePlan",
    "build_active_profile_metadata",
    "build_fourier_profile_names",
    "build_profile_index",
    "build_profile_layout",
    "build_profile_names",
    "build_boundary_slope_initial_state",
    "build_residual_block_metadata",
    "build_residual_block_radial_powers",
    "build_shape_profile_names",
    "decode_packed_blocks",
    "encode_packed_state",
    "get_prefix_profile_names",
    "packed_size",
    "refresh_fourier_family_metadata",
    "refresh_profile_runtime",
    "refresh_source_runtime",
    "refresh_stage_a_runtime",
    "validate_packed_state",
]

_EXPORTS = {
    "Operator": ("veqpy.operator.operator", "Operator"),
    "OperatorBuildPlan": ("veqpy.operator.build_plan", "OperatorBuildPlan"),
    "ResidualBindingLayout": ("veqpy.operator.build_plan", "ResidualBindingLayout"),
    "SourcePlan": ("veqpy.operator.source_plan", "SourcePlan"),
    "PACKED_LAYOUT_PROFILE_FIRST": (
        "veqpy.operator.packed_layout",
        "PACKED_LAYOUT_PROFILE_FIRST",
    ),
    "PROFILE_OFFSET_SPECS": ("veqpy.operator.packed_layout", "PROFILE_OFFSET_SPECS"),
    "PROFILE_STATIC_KWARGS": ("veqpy.operator.packed_layout", "PROFILE_STATIC_KWARGS"),
    "build_active_profile_metadata": (
        "veqpy.operator.packed_layout",
        "build_active_profile_metadata",
    ),
    "build_fourier_profile_names": (
        "veqpy.operator.packed_layout",
        "build_fourier_profile_names",
    ),
    "build_profile_index": ("veqpy.operator.packed_layout", "build_profile_index"),
    "build_profile_layout": ("veqpy.operator.packed_layout", "build_profile_layout"),
    "build_profile_names": ("veqpy.operator.packed_layout", "build_profile_names"),
    "build_boundary_slope_initial_state": (
        "veqpy.operator.initialize",
        "build_boundary_slope_initial_state",
    ),
    "build_residual_block_metadata": (
        "veqpy.operator.packed_layout",
        "build_residual_block_metadata",
    ),
    "build_residual_block_radial_powers": (
        "veqpy.operator.packed_layout",
        "build_residual_block_radial_powers",
    ),
    "build_shape_profile_names": (
        "veqpy.operator.packed_layout",
        "build_shape_profile_names",
    ),
    "decode_packed_blocks": ("veqpy.operator.packed_layout", "decode_packed_blocks"),
    "encode_packed_state": ("veqpy.operator.packed_layout", "encode_packed_state"),
    "get_prefix_profile_names": ("veqpy.operator.packed_layout", "get_prefix_profile_names"),
    "packed_size": ("veqpy.operator.packed_layout", "packed_size"),
    "refresh_fourier_family_metadata": (
        "veqpy.operator.profile_runtime",
        "refresh_fourier_family_metadata",
    ),
    "refresh_profile_runtime": (
        "veqpy.operator.profile_runtime",
        "refresh_profile_runtime",
    ),
    "refresh_stage_a_runtime": (
        "veqpy.operator.profile_runtime",
        "refresh_stage_a_runtime",
    ),
    "refresh_source_runtime": ("veqpy.operator.source_runtime", "refresh_source_runtime"),
    "validate_packed_state": ("veqpy.operator.packed_layout", "validate_packed_state"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported operator symbols lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
