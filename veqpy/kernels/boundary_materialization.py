"""
Module: veqpy.kernels.boundary_materialization

Role:
- Lower public KernelBoundary inputs into coefficient boundaries for backends.

Notes:
- Explicit coefficient boundaries pass through unchanged.
- R/Z scatter boundaries are fitted here, inside Kernel runtime materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from veqpy.kernels.boundary_fit import fit_boundary_params
from veqpy.kernels.types import (
    KernelBoundary,
    kernel_boundary_raw_fit_spec,
)


@dataclass(frozen=True, slots=True)
class MaterializedKernelBoundary:
    """Private boundary lowering result consumed by Kernel backend implementations."""

    boundary: KernelBoundary
    fit_backend: str
    fit_elapsed_ms: float
    fit_rms: float | None
    fit_max_curve_error: float | None
    fit_c_order: int | None
    fit_s_order: int | None


def materialize_kernel_boundary(boundary: KernelBoundary) -> MaterializedKernelBoundary:
    """Return a backend-ready coefficient boundary plus optional fit metadata."""

    if not isinstance(boundary, KernelBoundary):
        raise TypeError(f"boundary must be KernelBoundary, got {type(boundary).__name__}")
    raw = kernel_boundary_raw_fit_spec(boundary)
    if raw is None:
        return MaterializedKernelBoundary(
            boundary=boundary,
            fit_backend="explicit",
            fit_elapsed_ms=0.0,
            fit_rms=boundary.fit_rms,
            fit_max_curve_error=boundary.fit_max_curve_error,
            fit_c_order=boundary.fit_c_order,
            fit_s_order=boundary.fit_s_order,
        )

    R_boundary, Z_boundary, c_order, s_order, fit_maxtol = raw
    started = perf_counter()
    fitted = fit_boundary_params(
        R_boundary,
        Z_boundary,
        c_order=c_order,
        s_order=s_order,
        maxtol=fit_maxtol,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    materialized_boundary = KernelBoundary(
        a=float(fitted["a"]),
        R0=float(fitted["R0"]),
        Z0=float(fitted["Z0"]),
        B0=float(boundary.B0),
        ka=float(fitted["ka"]),
        c_offsets=np.asarray(fitted["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(fitted["s_offsets"], dtype=np.float64)[1:],
    )
    return MaterializedKernelBoundary(
        boundary=materialized_boundary,
        fit_backend="numpy",
        fit_elapsed_ms=float(elapsed_ms),
        fit_rms=float(fitted["rms"]),
        fit_max_curve_error=float(fitted["max_curve_error"]),
        fit_c_order=int(fitted["c_order"]),
        fit_s_order=int(fitted["s_order"]),
    )


def materialized_boundary_fit_payload(
    materialized: MaterializedKernelBoundary,
) -> dict[str, float | int | str | np.ndarray | None]:
    """Return script/benchmark-friendly fit metadata for one materialized boundary."""

    boundary = materialized.boundary
    return {
        "fit_backend": materialized.fit_backend,
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
