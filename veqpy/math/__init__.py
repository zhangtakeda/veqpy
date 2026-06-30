"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_CALCULUS",
    "DEFAULT_LOCAL_BARYCENTRIC_STENCIL",
    "DEFAULT_QUADRATURE",
    "SOURCE_INTERP_DEFAULT",
    "RHO_AXIS",
    "THETA_AXIS",
    "apply_accumulation",
    "apply_differentiation",
    "barycentric_log_weights",
    "build_uniform_source_interpolation_coefficients",
    "build_uniform_source_interpolation_matrix",
    "colwise_weighted_sum_into",
    "copy_into",
    "dot",
    "indexed_matvec_into",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
    "matvec_into",
    "normalize_source_interpolation_kind",
    "product_into",
    "rowwise_sum_into",
    "rowwise_weighted_sum_into",
    "scale_into",
    "scaled_product_into",
    "scaled_product_ratio_into",
    "scaled_ratio_into",
    "source_interpolation_kind_is_barycentric",
    "weighted_dot",
]

_EXPORTS = {
    "RHO_AXIS": ("veqpy.math.axes", "RHO_AXIS"),
    "THETA_AXIS": ("veqpy.math.axes", "THETA_AXIS"),
    "DEFAULT_CALCULUS": ("veqpy.math.calculus", "DEFAULT_CALCULUS"),
    "apply_accumulation": ("veqpy.math.calculus", "apply_accumulation"),
    "apply_differentiation": ("veqpy.math.calculus", "apply_differentiation"),
    "make_calculus": ("veqpy.math.calculus", "make_calculus"),
    "DEFAULT_LOCAL_BARYCENTRIC_STENCIL": (
        "veqpy.math.interpolate",
        "DEFAULT_LOCAL_BARYCENTRIC_STENCIL",
    ),
    "SOURCE_INTERP_DEFAULT": ("veqpy.math.interpolate", "SOURCE_INTERP_DEFAULT"),
    "barycentric_log_weights": ("veqpy.math.interpolate", "barycentric_log_weights"),
    "build_uniform_source_interpolation_coefficients": (
        "veqpy.math.interpolate",
        "build_uniform_source_interpolation_coefficients",
    ),
    "build_uniform_source_interpolation_matrix": (
        "veqpy.math.interpolate",
        "build_uniform_source_interpolation_matrix",
    ),
    "interpolation_matrix": ("veqpy.math.interpolate", "interpolation_matrix"),
    "normalize_source_interpolation_kind": (
        "veqpy.math.interpolate",
        "normalize_source_interpolation_kind",
    ),
    "source_interpolation_kind_is_barycentric": (
        "veqpy.math.interpolate",
        "source_interpolation_kind_is_barycentric",
    ),
    "DEFAULT_QUADRATURE": ("veqpy.math.quadrature", "DEFAULT_QUADRATURE"),
    "make_quadrature": ("veqpy.math.quadrature", "make_quadrature"),
    "colwise_weighted_sum_into": ("veqpy.math.fast", "colwise_weighted_sum_into"),
    "copy_into": ("veqpy.math.fast", "copy_into"),
    "dot": ("veqpy.math.fast", "dot"),
    "indexed_matvec_into": ("veqpy.math.fast", "indexed_matvec_into"),
    "matvec_into": ("veqpy.math.fast", "matvec_into"),
    "product_into": ("veqpy.math.fast", "product_into"),
    "rowwise_sum_into": ("veqpy.math.fast", "rowwise_sum_into"),
    "rowwise_weighted_sum_into": ("veqpy.math.fast", "rowwise_weighted_sum_into"),
    "scale_into": ("veqpy.math.fast", "scale_into"),
    "scaled_product_into": ("veqpy.math.fast", "scaled_product_into"),
    "scaled_product_ratio_into": ("veqpy.math.fast", "scaled_product_ratio_into"),
    "scaled_ratio_into": ("veqpy.math.fast", "scaled_ratio_into"),
    "weighted_dot": ("veqpy.math.fast", "weighted_dot"),
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
