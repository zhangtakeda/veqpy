from __future__ import annotations

import numpy as np
import pytest
from _helpers import assert_finite

from benchmarks._common import (
    CASE_REFERENCE_GFILES,
    GEQDSK_ROUTE_PROFILE_SIGNATURE,
    RouteBenchmarkSpec,
    geqdsk_kernel_case,
    solve_numba_case,
)
from veqpy.kernels.boundary_fit import fit_boundary_params
from veqpy.model import Geqdsk

PHYSICS_CASES = ("solovev", "chease", "efit")


@pytest.mark.slow
@pytest.mark.parametrize("case_key", PHYSICS_CASES)
def test_geqdsk_boundary_fit_is_small_against_reference_points(case_key: str) -> None:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key])
    fit = fit_boundary_params(
        np.asarray(geqdsk.boundary[:, 0], dtype=np.float64),
        np.asarray(geqdsk.boundary[:, 1], dtype=np.float64),
        c_order=10,
        s_order=10,
        maxtol=1.0,
    )

    assert float(fit["rms"]) < {"solovev": 2.0e-3, "chease": 1.0e-2, "efit": 5.0e-3}[case_key]
    assert (
        float(fit["max_curve_error"])
        < {
            "solovev": 3.0e-2,
            "chease": 4.0e-2,
            "efit": 3.0e-2,
        }[case_key]
    )
    assert np.isfinite(float(geqdsk.Bt0))


@pytest.mark.slow
@pytest.mark.parametrize("case_key", PHYSICS_CASES)
def test_numba_geqdsk_low_order_solution_has_physical_diagnostics(case_key: str) -> None:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key])
    case = geqdsk_kernel_case(
        case_key,
        "Low",
        route_spec=RouteBenchmarkSpec("PF", "psin", "uniform", "Ip"),
        signature=GEQDSK_ROUTE_PROFILE_SIGNATURE,
        method="powell",
        max_residual=1.0e-6,
        max_evaluations=2000,
        initial="cold",
        norm="fast",
    )

    result, kernel = solve_numba_case(case)
    try:
        equilibrium = kernel.build_equilibrium()
    finally:
        kernel.close()

    assert result.success is True
    assert_finite(result.raw_norm, name=f"{case_key} raw_norm")
    assert result.raw_norm <= 1.0e-5
    assert equilibrium.Ip == pytest.approx(abs(float(geqdsk.Ip)), rel=5.0e-3, abs=1.0)
    assert_finite(equilibrium.beta_t, name=f"{case_key} beta_t")
    assert 0.0 <= float(equilibrium.beta_t) <= 1.0

    for name in ("rho", "psin", "Psi", "q", "FFn_psin", "Pn_psin", "jtor"):
        values = np.asarray(getattr(equilibrium, name), dtype=np.float64)
        assert values.size > 0
        assert np.all(np.isfinite(values)), f"{case_key} {name} contains non-finite values"

    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    assert np.all(np.diff(rho) > 0.0)
    assert np.all(np.diff(psin) >= -1.0e-10)
