"""
Package: veqpy.numerics

Role:
- Provide shared mathematical helpers for model reconstruction and Kernel runtime binding.
- Own axes, quadrature, calculus, interpolation, and projection utilities.

Public API:
- RHO_AXIS and THETA_AXIS.
- make_quadrature and DEFAULT_QUADRATURE.
- make_calculus, DEFAULT_CALCULUS, apply_differentiation, and apply_accumulation.
- Interpolation helpers for source remapping and barycentric weights.

Dependencies:
- veqpy.base.Registry for keyed helper selection.
- NumPy and SciPy where used by concrete numerical modules.

Downstream:
- veqpy.model uses numerics for grid tables and model-side derived fields.
- veqpy.kernels uses numerics for source interpolation and backend setup helpers.

Design notes:
- Keep general numerical helpers here.
- Backend workspaces, packed row ABI, and residual/source runtime stages belong under
  veqpy.kernels.
"""

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
