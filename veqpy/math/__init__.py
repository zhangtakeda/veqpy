"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.
"""

from __future__ import annotations

from veqpy.math.calculus import make_calculus
from veqpy.math.interpolate import (
    barycentric_log_weights,
    build_uniform_source_interpolation_coefficients,
    build_uniform_source_interpolation_matrix,
    interpolation_matrix,
    normalize_source_interpolation_kind,
    source_interpolation_kind_is_barycentric,
)
from veqpy.math.quadrature import make_quadrature

__all__ = [
    "barycentric_log_weights",
    "build_uniform_source_interpolation_coefficients",
    "build_uniform_source_interpolation_matrix",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
    "normalize_source_interpolation_kind",
    "source_interpolation_kind_is_barycentric",
]
