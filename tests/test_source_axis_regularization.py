from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from veqpy.kernels.abi.source_semantics import (
    SOURCE_REGULARITY_ANCHOR_COUNT,
    _regularize_axis_even,
    _regularize_axis_linear,
    _regularize_axis_quadratic,
)


def _regularized(
    profile: np.ndarray,
    rho: np.ndarray,
    n_fix: int,
    kind: str,
) -> np.ndarray:
    out = np.array(profile, dtype=np.float64, copy=True)
    if kind == "linear":
        _regularize_axis_linear(out, rho, n_fix)
    elif kind == "quadratic":
        _regularize_axis_quadratic(out, rho, n_fix)
    else:
        _regularize_axis_even(out, rho, n_fix)
    return out


def test_axis_regularization_recovers_leading_parity_expansions() -> None:
    rho = np.linspace(0.0, 0.2, 21, dtype=np.float64)
    rho2 = rho * rho
    n_fix = 5
    profiles = {
        "linear": rho * (4.0 + 30.0 * rho2),
        "quadratic": rho2 * (4.0 + 30.0 * rho2),
        "even": 4.0 + 30.0 * rho2,
    }

    for kind, truth in profiles.items():
        corrupted = truth.copy()
        corrupted[:n_fix] = 99.0
        repaired = _regularized(corrupted, rho, n_fix, kind)
        assert_allclose(repaired[:n_fix], truth[:n_fix], rtol=2.0e-14, atol=2.0e-14)
        assert_allclose(repaired[n_fix:], corrupted[n_fix:], rtol=0.0, atol=0.0)


def test_axis_regularization_uses_a_fixed_four_anchor_stencil() -> None:
    assert SOURCE_REGULARITY_ANCHOR_COUNT == 4
    rho = np.linspace(0.0, 0.2, 21, dtype=np.float64)
    truth = 4.0 + 30.0 * rho * rho
    n_fix = 5
    corrupted = truth.copy()
    corrupted[:n_fix] = 99.0
    corrupted[n_fix + SOURCE_REGULARITY_ANCHOR_COUNT] = 1.0e12

    repaired = _regularized(corrupted, rho, n_fix, "even")

    assert_allclose(repaired[:n_fix], truth[:n_fix], rtol=2.0e-14, atol=2.0e-14)


def test_four_anchor_fit_damps_individual_anchor_noise() -> None:
    rho = np.linspace(0.0, 0.2, 21, dtype=np.float64)
    rho2 = rho * rho
    truth = 4.0 + 30.0 * rho2
    n_fix = 5
    noisy = truth.copy()
    noisy[n_fix : n_fix + 4] += np.array([0.5, -0.5, 0.0, 0.0])

    repaired = _regularized(noisy, rho, n_fix, "even")
    x0 = rho2[n_fix]
    x1 = rho2[n_fix + 1]
    legacy_gradient = (noisy[n_fix + 1] - noisy[n_fix]) / (x1 - x0)
    legacy_axis = noisy[n_fix] - legacy_gradient * x0

    assert abs(repaired[0] - truth[0]) < 0.2 * abs(legacy_axis - truth[0])
