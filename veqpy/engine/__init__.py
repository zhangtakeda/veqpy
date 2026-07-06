"""
Module: engine.__init__

Role:
- Expose the engine package-root contract used by peer submodules.

Notes:
- This package owns model-layer numerical kernels and source helper routines used
  by VEQlib's Numba backend.
- Kernel construction, backend selection, and nonlinear solving live in VEQlib.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RHO_AXIS",
    "THETA_AXIS",
    "backend_abi",
    "COORDINATE_NAMES",
    "COORDINATE_CODES",
    "PSIN_COORDINATE",
    "RHO_COORDINATE",
    "SourceExecutionABI",
    "build_source_remap_cache",
    "validate_route",
    "full_differentiation",
    "full_integration",
    "numba_operator",
    "numba_profile",
    "numba_residual",
    "numba_source",
    "resolve_source_inputs",
    "source_parameterization_for_route_key",
    "update_fourier_family_fields",
    "update_geometry_hot_auto",
    "update_profile",
]

_MODULE_EXPORTS = {
    "backend_abi": "veqpy.engine.backend_abi",
    "numba_operator": "veqpy.engine.numba_operator",
    "numba_profile": "veqpy.engine.numba_profile",
    "numba_residual": "veqpy.engine.numba_residual",
    "numba_source": "veqpy.engine.numba_source",
}

_EXPORTS = {
    "RHO_AXIS": ("veqpy.math", "RHO_AXIS"),
    "THETA_AXIS": ("veqpy.math", "THETA_AXIS"),
    "COORDINATE_NAMES": ("veqpy.engine.numba_source", "COORDINATE_NAMES"),
    "COORDINATE_CODES": ("veqpy.engine.numba_source", "COORDINATE_CODES"),
    "PSIN_COORDINATE": ("veqpy.engine.numba_source", "PSIN_COORDINATE"),
    "RHO_COORDINATE": ("veqpy.engine.numba_source", "RHO_COORDINATE"),
    "SourceExecutionABI": ("veqpy.engine.backend_abi", "SourceExecutionABI"),
    "build_source_remap_cache": ("veqpy.engine.numba_source", "build_source_remap_cache"),
    "validate_route": ("veqpy.engine.numba_source", "validate_route"),
    "full_differentiation": ("veqpy.engine.numba_source", "full_differentiation"),
    "full_integration": ("veqpy.engine.numba_source", "full_integration"),
    "resolve_source_inputs": ("veqpy.engine.numba_source", "resolve_source_inputs"),
    "source_parameterization_for_route_key": (
        "veqpy.engine.numba_source",
        "source_parameterization_for_route_key",
    ),
    "update_fourier_family_fields": (
        "veqpy.engine.numba_source",
        "update_fourier_family_fields",
    ),
    "update_geometry_hot_auto": ("veqpy.engine.numba_geometry", "update_geometry_hot_auto"),
    "update_profile": ("veqpy.engine.numba_profile", "update_profile"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported engine symbols lazily at the package boundary."""

    if name in _MODULE_EXPORTS:
        value = import_module(_MODULE_EXPORTS[name])
        globals()[name] = value
        return value
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return normal module attributes plus lazy package-boundary exports."""

    return sorted(set(globals()) | set(__all__))
