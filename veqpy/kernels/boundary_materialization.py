"""
Module: veqpy.kernels.boundary_materialization

Role:
- Lower public _BoundaryCase inputs into coefficient boundaries for backends.

Notes:
- Explicit coefficient boundaries pass through unchanged.
- R/Z scatter boundaries are fitted here, inside Kernel runtime materialization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from veqpy.kernels.boundary_fit import fit_boundary_params
from veqpy.kernels.types import (
    _BoundaryCase,
    _kernel_boundary_with_fit_metadata,
    _nonnegative_int,
    kernel_boundary_raw_fit_spec,
)


@dataclass(frozen=True, slots=True)
class _MaterializedBoundary:
    """Private boundary lowering result consumed by Kernel backend implementations."""

    boundary: _BoundaryCase
    fit_backend: str
    fit_elapsed_ms: float
    fit_rms: float | None
    fit_max_curve_error: float | None
    fit_c_order: int | None
    fit_s_order: int | None
    fit_method: str | None


BoundaryFitter = Callable[..., dict[str, float | np.ndarray | str]]


def materialize_kernel_boundary(
    boundary: _BoundaryCase,
    *,
    fit_backend: str = "numpy",
    fitter: BoundaryFitter | None = None,
    method: str | None = None,
    c_order: int | None = None,
    s_order: int | None = None,
    maxtol: float | None = None,
) -> _MaterializedBoundary:
    """Return a backend-ready coefficient boundary plus optional fit metadata."""

    if not isinstance(boundary, _BoundaryCase):
        raise TypeError(f"boundary must be _BoundaryCase, got {type(boundary).__name__}")
    raw = kernel_boundary_raw_fit_spec(boundary)
    if raw is None:
        _reject_parameterized_fit_overrides(
            method=method,
            c_order=c_order,
            s_order=s_order,
            maxtol=maxtol,
        )
        return _MaterializedBoundary(
            boundary=boundary,
            fit_backend="explicit",
            fit_elapsed_ms=0.0,
            fit_rms=boundary.fit_rms,
            fit_max_curve_error=boundary.fit_max_curve_error,
            fit_c_order=boundary.fit_c_order,
            fit_s_order=boundary.fit_s_order,
            fit_method=boundary.fit_method,
        )

    R_boundary, Z_boundary, raw_c_order, raw_s_order, raw_fit_maxtol, raw_fit_method = raw
    fit_c_order, fit_s_order, fit_maxtol, fit_method = _resolve_fit_options(
        raw_c_order=raw_c_order,
        raw_s_order=raw_s_order,
        raw_fit_maxtol=raw_fit_maxtol,
        raw_fit_method=raw_fit_method,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
        method=method,
    )
    started = perf_counter()
    fitted = _fit_boundary_params(
        fitter,
        R_boundary,
        Z_boundary,
        fit_backend=fit_backend,
        c_order=fit_c_order,
        s_order=fit_s_order,
        maxtol=fit_maxtol,
        method=fit_method,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    fit_method = str(fitted.get("method", fit_method))
    materialized_boundary = _kernel_boundary_with_fit_metadata(
        _BoundaryCase(
            a=float(fitted["a"]),
            R0=float(fitted["R0"]),
            Z0=float(fitted["Z0"]),
            B0=float(boundary.B0),
            ka=float(fitted["ka"]),
            c_offsets=np.asarray(fitted["c_offsets"], dtype=np.float64),
            s_offsets=np.asarray(fitted["s_offsets"], dtype=np.float64)[1:],
        ),
        fit_rms=float(fitted["rms"]),
        fit_max_curve_error=float(fitted["max_curve_error"]),
        fit_c_order=int(fitted["c_order"]),
        fit_s_order=int(fitted["s_order"]),
        fit_method=fit_method,
    )
    return _MaterializedBoundary(
        boundary=materialized_boundary,
        fit_backend=str(fit_backend),
        fit_elapsed_ms=float(elapsed_ms),
        fit_rms=materialized_boundary.fit_rms,
        fit_max_curve_error=materialized_boundary.fit_max_curve_error,
        fit_c_order=materialized_boundary.fit_c_order,
        fit_s_order=materialized_boundary.fit_s_order,
        fit_method=materialized_boundary.fit_method,
    )


def fit_kernel_boundary(
    boundary: _BoundaryCase,
    *,
    fit_backend: str = "numba",
    method: str | None = None,
    c_order: int | None = None,
    s_order: int | None = None,
    maxtol: float | None = None,
    fitter: BoundaryFitter | None = None,
) -> _BoundaryCase:
    """Return a parameterized _BoundaryCase by explicitly fitting raw R/Z input."""

    return materialize_kernel_boundary(
        boundary,
        fit_backend=fit_backend,
        fitter=fitter,
        method=method,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
    ).boundary


def _reject_parameterized_fit_overrides(
    *,
    method: str | None,
    c_order: int | None,
    s_order: int | None,
    maxtol: float | None,
) -> None:
    if any(value is not None for value in (method, c_order, s_order, maxtol)):
        raise ValueError("fit overrides are only valid when R/Z boundary points are stored")


def _resolve_fit_options(
    *,
    raw_c_order: int,
    raw_s_order: int,
    raw_fit_maxtol: float,
    raw_fit_method: str,
    c_order: int | None,
    s_order: int | None,
    maxtol: float | None,
    method: str | None,
) -> tuple[int, int, float, str]:
    if (c_order is None) != (s_order is None):
        raise ValueError("c_order and s_order must be provided together or both omitted")
    if c_order is None:
        fit_c_order, fit_s_order = raw_c_order, raw_s_order
    else:
        assert s_order is not None
        fit_c_order = _nonnegative_int(c_order, "c_order")
        fit_s_order = _nonnegative_int(s_order, "s_order")
    fit_maxtol = raw_fit_maxtol if maxtol is None else float(maxtol)
    if fit_maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")
    fit_method = raw_fit_method if method is None else str(method)
    return fit_c_order, fit_s_order, fit_maxtol, fit_method


def materialized_boundary_fit_payload(
    materialized: _MaterializedBoundary,
) -> dict[str, float | int | str | np.ndarray | None]:
    """Return script/benchmark-friendly fit metadata for one materialized boundary."""

    boundary = materialized.boundary
    return {
        "fit_backend": materialized.fit_backend,
        "fit_method": materialized.fit_method,
        "fit_elapsed_ms": float(materialized.fit_elapsed_ms),
        "rms": materialized.fit_rms,
        "max_curve_error": materialized.fit_max_curve_error,
        "c_order": materialized.fit_c_order,
        "s_order": materialized.fit_s_order,
        "a": float(boundary.a),
        "R0": float(boundary.R0),
        "Z0": float(boundary.Z0),
        "ka": float(boundary.ka),
        "c_offsets": np.asarray(boundary.c_offsets, dtype=np.float64),
        "s_offsets": np.concatenate(([0.0], np.asarray(boundary.s_offsets, dtype=np.float64))),
    }


def _fit_boundary_params(
    fitter: BoundaryFitter | None,
    R_boundary: Any,
    Z_boundary: Any,
    *,
    fit_backend: str,
    c_order: int,
    s_order: int,
    maxtol: float,
    method: str,
) -> dict[str, float | np.ndarray | str]:
    if fitter is None:
        fitter = _fitter_for_backend(fit_backend)
    return fitter(
        R_boundary,
        Z_boundary,
        c_order=c_order,
        s_order=s_order,
        maxtol=maxtol,
        method=method,
    )


def _fitter_for_backend(fit_backend: str) -> BoundaryFitter:
    backend = str(fit_backend).lower()
    if backend == "numpy":
        return fit_boundary_params
    if backend == "numba":
        from veqpy.kernels.numba_kernel.boundary_fit import fit_boundary_params_numba

        return fit_boundary_params_numba
    if backend == "cxx":
        from veqpy.kernels.cxx_kernel.boundary_fit import fit_boundary_params_cxx

        return fit_boundary_params_cxx
    raise ValueError(f"unsupported boundary fitter backend {fit_backend!r}")
