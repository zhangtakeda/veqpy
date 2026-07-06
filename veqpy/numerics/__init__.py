"""Shared numerical helpers for model setup and Kernel runtime binding."""

from __future__ import annotations

from .axes import RHO_AXIS, THETA_AXIS
from .calculus import (
    DEFAULT_CALCULUS,
    apply_accumulation,
    apply_differentiation,
    make_calculus,
)
from .interpolate import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    SOURCE_INTERP_DEFAULT,
    barycentric_log_weights,
    build_uniform_source_interpolation_coefficients,
    build_uniform_source_interpolation_matrix,
    interpolation_matrix,
    normalize_source_interpolation_kind,
    source_interpolation_kind_is_barycentric,
)
from .quadrature import DEFAULT_QUADRATURE, make_quadrature

__all__ = [
    "DEFAULT_CALCULUS",
    "DEFAULT_LOCAL_BARYCENTRIC_STENCIL",
    "DEFAULT_QUADRATURE",
    "RHO_AXIS",
    "SOURCE_INTERP_DEFAULT",
    "THETA_AXIS",
    "apply_accumulation",
    "apply_differentiation",
    "barycentric_log_weights",
    "build_uniform_source_interpolation_coefficients",
    "build_uniform_source_interpolation_matrix",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
    "normalize_source_interpolation_kind",
    "source_interpolation_kind_is_barycentric",
]
