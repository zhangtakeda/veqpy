from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks._common import CASE_KEYS, CASE_REFERENCE_GFILES
from veqpy.kernels.boundary_fit import fit_boundary_params
from veqpy.kernels.cxx_kernel.boundary_fit import fit_boundary_params_cxx
from veqpy.kernels.numba_kernel.boundary_fit import fit_boundary_params_numba
from veqpy.model import Geqdsk


def _load_boundary(case_key: str) -> tuple[np.ndarray, np.ndarray]:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key])
    boundary = np.asarray(geqdsk.boundary, dtype=np.float64)
    return boundary[:, 0].copy(), boundary[:, 1].copy()


def _coeff_vector(fit: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(fit["R0"]),
            float(fit["Z0"]),
            float(fit["a"]),
            float(fit["ka"]),
            *np.asarray(fit["c_offsets"], dtype=np.float64).tolist(),
            *np.asarray(fit["s_offsets"], dtype=np.float64).tolist(),
        ],
        dtype=np.float64,
    )


def _assert_fit_is_reasonable(case_key: str, fit: dict[str, Any]) -> None:
    rms = float(fit["rms"])
    curve = float(fit["max_curve_error"])
    assert np.isfinite(rms)
    assert np.isfinite(curve)
    assert rms < {"solovev": 2.0e-3, "chease": 1.0e-2, "efit": 5.0e-3}[case_key]
    assert curve < {"solovev": 3.0e-2, "chease": 4.0e-2, "efit": 3.0e-2}[case_key]


def _fit_numpy(R: np.ndarray, Z: np.ndarray, *, method: str) -> dict[str, Any]:
    return fit_boundary_params(R, Z, c_order=10, s_order=10, maxtol=1.0, method=method)


def _fit_numba(R: np.ndarray, Z: np.ndarray, *, method: str) -> dict[str, Any]:
    return fit_boundary_params_numba(R, Z, c_order=10, s_order=10, maxtol=1.0, method=method)


def _fit_cxx(R: np.ndarray, Z: np.ndarray, *, method: str) -> dict[str, Any]:
    try:
        return fit_boundary_params_cxx(
            R,
            Z,
            c_order=10,
            s_order=10,
            maxtol=1.0,
            method=method,
        )
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if any(token in text.lower() for token in ("cmake", "compiler", "nanobind", "build")):
            pytest.skip(f"native boundary fitter unavailable: {text}")
        raise


@pytest.mark.slow
@pytest.mark.parametrize("case_key", CASE_KEYS)
@pytest.mark.parametrize("method", ("qr", "gnqr", "least-square"))
def test_numba_boundary_fitter_matches_numpy(case_key: str, method: str) -> None:
    R, Z = _load_boundary(case_key)

    reference = _fit_numpy(R, Z, method=method)
    fitted = _fit_numba(R, Z, method=method)

    _assert_fit_is_reasonable(case_key, reference)
    _assert_fit_is_reasonable(case_key, fitted)
    if method in {"qr", "gnqr"}:
        assert_allclose(_coeff_vector(fitted), _coeff_vector(reference), rtol=0.0, atol=1.0e-10)
        assert_allclose(fitted["rms"], reference["rms"], rtol=0.0, atol=1.0e-12)
        assert_allclose(
            fitted["max_curve_error"],
            reference["max_curve_error"],
            rtol=0.0,
            atol=1.0e-12,
        )
    else:
        assert float(fitted["rms"]) <= 2.0 * float(reference["rms"])
        assert float(fitted["max_curve_error"]) <= 1.2 * float(reference["max_curve_error"])


@pytest.mark.slow
@pytest.mark.parametrize("case_key", CASE_KEYS)
@pytest.mark.parametrize("method", ("qr", "gnqr", "least-square"))
def test_cxx_boundary_fitter_matches_numpy_or_native_ls(case_key: str, method: str) -> None:
    R, Z = _load_boundary(case_key)

    reference = _fit_numpy(R, Z, method=method)
    fitted = _fit_cxx(R, Z, method=method)

    _assert_fit_is_reasonable(case_key, reference)
    _assert_fit_is_reasonable(case_key, fitted)
    if method in {"qr", "gnqr"}:
        assert_allclose(_coeff_vector(fitted), _coeff_vector(reference), rtol=0.0, atol=1.0e-8)
        assert_allclose(fitted["rms"], reference["rms"], rtol=0.0, atol=1.0e-9)
        assert_allclose(
            fitted["max_curve_error"],
            reference["max_curve_error"],
            rtol=0.0,
            atol=1.0e-8,
        )
    else:
        assert float(fitted["rms"]) <= 2.0 * float(reference["rms"])
        assert float(fitted["max_curve_error"]) <= 1.2 * float(reference["max_curve_error"])
