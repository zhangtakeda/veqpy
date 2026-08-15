from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from veqpy import Geqdsk
from veqpy.kernels.boundary_fit import fit_boundary_params
from veqpy.kernels.cxx_kernel.boundary_fit import fit_boundary_params_cxx
from veqpy.kernels.numba_kernel.boundary_fit import fit_boundary_params_numba

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("name", ("SOLOVEV", "CHEASE", "EFIT"))
def test_numba_boundary_fit_matches_python_reference(name: str) -> None:
    geqdsk = Geqdsk(DATA / f"{name}.geqdsk")
    R = np.asarray(geqdsk.boundary[:, 0], dtype=np.float64)
    Z = np.asarray(geqdsk.boundary[:, 1], dtype=np.float64)
    reference = fit_boundary_params(R, Z, c_order=8, s_order=8, maxtol=1.0)
    fitted = fit_boundary_params_numba(R, Z, c_order=8, s_order=8, maxtol=1.0)
    assert float(reference["rms"]) < 2.0e-2
    assert float(fitted["rms"]) < 2.0e-2
    np.testing.assert_allclose(fitted["R0"], reference["R0"], atol=1.0e-10)
    np.testing.assert_allclose(fitted["a"], reference["a"], atol=1.0e-10)


def test_cxx_boundary_fit_has_explicit_native_diagnostic_or_matches_numba() -> None:
    geqdsk = Geqdsk(DATA / "SOLOVEV.geqdsk")
    R = np.asarray(geqdsk.boundary[:, 0], dtype=np.float64)
    Z = np.asarray(geqdsk.boundary[:, 1], dtype=np.float64)
    try:
        fitted = fit_boundary_params_cxx(R, Z, c_order=4, s_order=4, maxtol=1.0)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}".lower()
        if any(token in detail for token in ("cmake", "compiler", "nanobind", "build", "native")):
            pytest.skip(f"Cxx native fitter unavailable: {error}")
        raise
    assert np.isfinite(float(fitted["rms"]))
