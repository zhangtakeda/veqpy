"""
Module: model.numerics

Role:
- Host model-facing numerical helpers used by grid and equilibrium objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GridWorkspace",
    "barycentric_log_weights",
    "interpolation_matrix",
    "update_geometry_hot_auto",
    "update_profile",
]

_EXPORTS = {
    "RHO_AXIS": ("veqpy.model.numerics.axes", "RHO_AXIS"),
    "THETA_AXIS": ("veqpy.model.numerics.axes", "THETA_AXIS"),
    "DEFAULT_CALCULUS": ("veqpy.model.numerics.calculus", "DEFAULT_CALCULUS"),
    "apply_accumulation": ("veqpy.model.numerics.calculus", "apply_accumulation"),
    "apply_differentiation": ("veqpy.model.numerics.calculus", "apply_differentiation"),
    "make_calculus": ("veqpy.model.numerics.calculus", "make_calculus"),
    "DEFAULT_LOCAL_BARYCENTRIC_STENCIL": (
        "veqpy.model.numerics.interpolate",
        "DEFAULT_LOCAL_BARYCENTRIC_STENCIL",
    ),
    "SOURCE_INTERP_DEFAULT": ("veqpy.model.numerics.interpolate", "SOURCE_INTERP_DEFAULT"),
    "barycentric_log_weights": ("veqpy.model.numerics.interpolate", "barycentric_log_weights"),
    "build_uniform_source_interpolation_coefficients": (
        "veqpy.model.numerics.interpolate",
        "build_uniform_source_interpolation_coefficients",
    ),
    "build_uniform_source_interpolation_matrix": (
        "veqpy.model.numerics.interpolate",
        "build_uniform_source_interpolation_matrix",
    ),
    "interpolation_matrix": ("veqpy.model.numerics.interpolate", "interpolation_matrix"),
    "normalize_source_interpolation_kind": (
        "veqpy.model.numerics.interpolate",
        "normalize_source_interpolation_kind",
    ),
    "source_interpolation_kind_is_barycentric": (
        "veqpy.model.numerics.interpolate",
        "source_interpolation_kind_is_barycentric",
    ),
    "GridWorkspace": ("veqpy.model.numerics.grid_workspace", "GridWorkspace"),
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
    "GRID_RADIAL_X": ("veqpy.model.numerics.field_rows", "GRID_RADIAL_X"),
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
    "update_geometry_hot_auto": (
        "veqpy.model.numerics.geometry",
        "update_geometry_hot_auto",
    ),
    "update_profile": ("veqpy.model.numerics.profile_eval", "update_profile"),
    "update_profiles_packed_bulk": (
        "veqpy.model.numerics.profile_eval",
        "update_profiles_packed_bulk",
    ),
    "DEFAULT_QUADRATURE": ("veqpy.model.numerics.quadrature", "DEFAULT_QUADRATURE"),
    "make_quadrature": ("veqpy.model.numerics.quadrature", "make_quadrature"),
    "colwise_weighted_sum_into": ("veqpy.model.numerics.fast", "colwise_weighted_sum_into"),
    "copy_into": ("veqpy.model.numerics.fast", "copy_into"),
    "dot": ("veqpy.model.numerics.fast", "dot"),
    "indexed_matvec_into": ("veqpy.model.numerics.fast", "indexed_matvec_into"),
    "matvec_into": ("veqpy.model.numerics.fast", "matvec_into"),
    "product_into": ("veqpy.model.numerics.fast", "product_into"),
    "rowwise_sum_into": ("veqpy.model.numerics.fast", "rowwise_sum_into"),
    "rowwise_weighted_sum_into": ("veqpy.model.numerics.fast", "rowwise_weighted_sum_into"),
    "scale_into": ("veqpy.model.numerics.fast", "scale_into"),
    "scaled_product_into": ("veqpy.model.numerics.fast", "scaled_product_into"),
    "scaled_product_ratio_into": ("veqpy.model.numerics.fast", "scaled_product_ratio_into"),
    "scaled_ratio_into": ("veqpy.model.numerics.fast", "scaled_ratio_into"),
    "weighted_dot": ("veqpy.model.numerics.fast", "weighted_dot"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported math helpers lazily at the package boundary."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
