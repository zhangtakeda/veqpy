"""
Module: workspace.__init__

Role:
- Expose the workspace package-root contract used outside ``veqpy.kernels.numba_kernel.workspace``.

Public API:
- GridWorkspace and allocate_runtime_state for operator construction
- Workspace classes and field-row constants consumed by engine/layout/operator

Notes:
- Scratch semantics and allocation details stay in their owning modules.
- Package roots are the only modules that declare ``__all__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GeometryWorkspace",
    "ProfileWorkspace",
    "ResidualWorkspace",
    "SourceWorkspace",
    "allocate_runtime_state",
]

_EXPORTS = {
    "GEOMETRY_RADIAL_KN": ("veqpy.model.numerics.field_rows", "GEOMETRY_RADIAL_KN"),
    "GEOMETRY_RADIAL_KN_R": ("veqpy.model.numerics.field_rows", "GEOMETRY_RADIAL_KN_R"),
    "GEOMETRY_RADIAL_LN_R": ("veqpy.model.numerics.field_rows", "GEOMETRY_RADIAL_LN_R"),
    "GEOMETRY_RADIAL_S_R": ("veqpy.model.numerics.field_rows", "GEOMETRY_RADIAL_S_R"),
    "GEOMETRY_RADIAL_V_R": ("veqpy.model.numerics.field_rows", "GEOMETRY_RADIAL_V_R"),
    "GEOMETRY_SURFACE_GRTDIVJR_T": (
        "veqpy.model.numerics.field_rows",
        "GEOMETRY_SURFACE_GRTDIVJR_T",
    ),
    "GEOMETRY_SURFACE_GTTDIVJR": (
        "veqpy.model.numerics.field_rows",
        "GEOMETRY_SURFACE_GTTDIVJR",
    ),
    "GEOMETRY_SURFACE_GTTDIVJR_R": (
        "veqpy.model.numerics.field_rows",
        "GEOMETRY_SURFACE_GTTDIVJR_R",
    ),
    "GEOMETRY_SURFACE_J": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_J"),
    "GEOMETRY_SURFACE_JDIVR": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_JDIVR"),
    "GEOMETRY_SURFACE_R": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_R"),
    "GEOMETRY_SURFACE_R_T": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_R_T"),
    "GEOMETRY_SURFACE_SIN_TB": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_SIN_TB"),
    "GEOMETRY_SURFACE_Z_T": ("veqpy.model.numerics.field_rows", "GEOMETRY_SURFACE_Z_T"),
    "GRID_POLOIDAL_COS_MTHETA_START": (
        "veqpy.model.numerics.field_rows",
        "GRID_POLOIDAL_COS_MTHETA_START",
    ),
    "GRID_POLOIDAL_THETA": ("veqpy.model.numerics.field_rows", "GRID_POLOIDAL_THETA"),
    "GRID_RADIAL_RHO": ("veqpy.model.numerics.field_rows", "GRID_RADIAL_RHO"),
    "GRID_RADIAL_RHO_POWERS_START": (
        "veqpy.model.numerics.field_rows",
        "GRID_RADIAL_RHO_POWERS_START",
    ),
    "GRID_RADIAL_Y": ("veqpy.model.numerics.field_rows", "GRID_RADIAL_Y"),
    "PROFILE_R": ("veqpy.model.numerics.field_rows", "PROFILE_R"),
    "PROFILE_RR": ("veqpy.model.numerics.field_rows", "PROFILE_RR"),
    "PROFILE_VALUE": ("veqpy.model.numerics.field_rows", "PROFILE_VALUE"),
    "RESIDUAL_ROOT_FFN_PSIN": ("veqpy.model.numerics.field_rows", "RESIDUAL_ROOT_FFN_PSIN"),
    "RESIDUAL_ROOT_PN_PSIN": ("veqpy.model.numerics.field_rows", "RESIDUAL_ROOT_PN_PSIN"),
    "RESIDUAL_ROOT_PSIN": ("veqpy.model.numerics.field_rows", "RESIDUAL_ROOT_PSIN"),
    "RESIDUAL_ROOT_PSIN_R": ("veqpy.model.numerics.field_rows", "RESIDUAL_ROOT_PSIN_R"),
    "RESIDUAL_ROOT_PSIN_RR": ("veqpy.model.numerics.field_rows", "RESIDUAL_ROOT_PSIN_RR"),
    "RESIDUAL_SURFACE_G": ("veqpy.model.numerics.field_rows", "RESIDUAL_SURFACE_G"),
    "RESIDUAL_SURFACE_GPSIN_R": ("veqpy.model.numerics.field_rows", "RESIDUAL_SURFACE_GPSIN_R"),
    "RESIDUAL_SURFACE_GPSIN_R_SIN_TB": (
        "veqpy.model.numerics.field_rows",
        "RESIDUAL_SURFACE_GPSIN_R_SIN_TB",
    ),
    "RESIDUAL_SURFACE_GPSIN_Z": ("veqpy.model.numerics.field_rows", "RESIDUAL_SURFACE_GPSIN_Z"),
    "GeometryWorkspace": (
        "veqpy.kernels.numba_kernel.workspace.geometry_workspace",
        "GeometryWorkspace",
    ),
    "GridWorkspace": ("veqpy.model.numerics.grid_workspace", "GridWorkspace"),
    "ProfileWorkspace": (
        "veqpy.kernels.numba_kernel.workspace.profile_workspace",
        "ProfileWorkspace",
    ),
    "ResidualWorkspace": (
        "veqpy.kernels.numba_kernel.workspace.residual_workspace",
        "ResidualWorkspace",
    ),
    "SourceWorkspace": ("veqpy.kernels.numba_kernel.workspace.source_workspace", "SourceWorkspace"),
    "allocate_runtime_state": (
        "veqpy.kernels.numba_kernel.workspace.allocation",
        "allocate_runtime_state",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve exported workspace symbols lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
