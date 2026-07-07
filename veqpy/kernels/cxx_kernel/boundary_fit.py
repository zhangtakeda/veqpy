"""
Module: veqpy.kernels.cxx_kernel.boundary_fit

Role:
- Load the native boundary phase-QR fitter from a Cxx Kernel artifact.

Notes:
- The fitter is topology-independent, but it lives in the same nanobind module
  as generated Kernel artifacts to avoid a second native build system.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from types import ModuleType
from typing import Any

import numpy as np

from veqpy.kernels.types import KernelRecipe, KernelTopology

from .registry import KernelRegistry


def fit_boundary_params_cxx(
    R_boundary: Any,
    Z_boundary: Any,
    *,
    c_order: int,
    s_order: int,
    maxtol: float = 1.0e-2,
) -> dict[str, float | np.ndarray]:
    """Fit RZ boundary samples with the native phase-QR implementation."""

    R = np.ascontiguousarray(R_boundary, dtype=np.float64)
    Z = np.ascontiguousarray(Z_boundary, dtype=np.float64)
    if R.ndim != 1:
        raise ValueError(f"R_boundary must be 1D, got {R.shape}")
    if Z.ndim != 1:
        raise ValueError(f"Z_boundary must be 1D, got {Z.shape}")
    maxtol = float(maxtol)
    if maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")

    native = _boundary_fit_module()
    payload = native.fit_boundary_qr(R, Z, int(c_order), int(s_order))
    result = {
        "R0": float(payload["R0"]),
        "Z0": float(payload["Z0"]),
        "a": float(payload["a"]),
        "ka": float(payload["ka"]),
        "c_offsets": np.asarray(payload["c_offsets"], dtype=np.float64),
        "s_offsets": np.asarray(payload["s_offsets"], dtype=np.float64),
        "rms": float(payload["rms"]),
        "max_curve_error": float(payload["max_curve_error"]),
        "c_order": int(payload["c_order"]),
        "s_order": int(payload["s_order"]),
    }
    if result["rms"] >= maxtol:
        warnings.warn(
            (
                f"Boundary fit RMS {float(result['rms']):.6e} exceeds maxtol "
                f"{maxtol:.6e} for c/s orders={c_order}/{s_order}"
            ),
            stacklevel=2,
        )
    return result


@lru_cache(maxsize=1)
def _boundary_fit_module() -> ModuleType:
    topology = KernelTopology(
        h_count=1,
        v_count=0,
        kappa_count=1,
        psin_count=0,
        F_count=0,
        c_counts=(1,),
        s_counts=(1,),
        Nr=4,
        Nt=4,
        route="PF",
        coordinate="rho",
        nodes="uniform",
        ip_constraint=True,
        sample_count=8,
    )
    recipe = KernelRecipe(backend="cxx", build="fastmath")
    loaded = KernelRegistry().load_kernel(topology, recipe=recipe)
    return loaded.module
