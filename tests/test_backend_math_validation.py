from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.kernels.numba_kernel import jit_math
from veqpy.kernels.numba_kernel.profile_stage import update_profile


def test_numba_jit_math_dense_products_match_numpy() -> None:
    lhs = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float64)
    rhs = np.array([[2.0, 0.0], [-1.0, 3.0], [0.25, -2.0]], dtype=np.float64)
    vec = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    matmul = np.full((2, 2), np.nan, dtype=np.float64)
    matvec = np.full(2, np.nan, dtype=np.float64)
    indexed = np.zeros(4, dtype=np.float64)
    indices = np.array([3, 1], dtype=np.intp)

    jit_math.matmul_into(matmul, lhs, rhs)
    jit_math.matvec_into(matvec, lhs, vec)
    jit_math.indexed_matvec_into(indexed, indices, lhs, vec)

    assert_allclose(matmul, lhs @ rhs)
    assert_allclose(matvec, lhs @ vec)
    assert_allclose(indexed[indices], lhs @ vec)
    assert_allclose(indexed[[0, 2]], 0.0)


def test_numba_jit_math_reductions_match_numpy() -> None:
    values = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 4)
    other = np.linspace(0.25, 1.75, 12, dtype=np.float64).reshape(3, 4)
    row_weights = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    col_weights = np.array([1.0, 0.25, 0.5, 2.0], dtype=np.float64)

    row = np.empty(3, dtype=np.float64)
    col = np.empty(4, dtype=np.float64)

    jit_math.rowwise_sum_into(row, values)
    assert_allclose(row, np.sum(values, axis=1))
    jit_math.rowwise_weighted_sum_into(row, values, col_weights)
    assert_allclose(row, values @ col_weights)
    jit_math.rowwise_dot_into(row, values, other)
    assert_allclose(row, np.sum(values * other, axis=1))

    jit_math.colwise_sum_into(col, values)
    assert_allclose(col, np.sum(values, axis=0))
    jit_math.colwise_weighted_sum_into(col, values, row_weights)
    assert_allclose(col, row_weights @ values)
    jit_math.colwise_dot_into(col, values, other)
    assert_allclose(col, np.sum(values * other, axis=0))

    flat = values.ravel()
    flat_other = other.ravel()
    weights = np.linspace(1.0, 2.0, flat.size, dtype=np.float64)
    denominator = np.linspace(2.0, 3.0, flat.size, dtype=np.float64)
    assert jit_math.dot(flat, flat_other) == pytest.approx(float(np.dot(flat, flat_other)))
    assert jit_math.weighted_dot(flat, flat_other, weights) == pytest.approx(
        float(np.sum(weights * flat * flat_other))
    )
    assert jit_math.weighted_ratio_dot(flat, flat_other, denominator, weights) == pytest.approx(
        float(np.sum(weights * flat * flat_other / denominator))
    )


def test_numba_jit_math_elementwise_ops_match_numpy() -> None:
    lhs = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    rhs = np.array([0.5, 4.0, -1.5, 2.0], dtype=np.float64)
    denominator = np.array([2.0, 2.5, 3.0, 4.0], dtype=np.float64)
    out = np.empty_like(lhs)

    jit_math.copy_into(out, lhs)
    assert_allclose(out, lhs)
    jit_math.product_into(out, lhs, rhs)
    assert_allclose(out, lhs * rhs)
    jit_math.scale_into(out, lhs, 2.5)
    assert_allclose(out, 2.5 * lhs)
    jit_math.scaled_product_into(out, lhs, rhs, -0.25)
    assert_allclose(out, -0.25 * lhs * rhs)
    jit_math.scaled_ratio_into(out, lhs, denominator, 3.0)
    assert_allclose(out, 3.0 * lhs / denominator)
    jit_math.scaled_product_ratio_into(out, lhs, rhs, denominator, 1.25)
    assert_allclose(out, 1.25 * lhs * rhs / denominator)
    jit_math.maximum_floor_into(out, lhs, -1.0)
    assert_allclose(out, np.maximum(lhs, -1.0))


def _rp_fields(r: np.ndarray, power: int) -> np.ndarray:
    fields = np.empty((3, r.size), dtype=np.float64)
    fields[0] = r**power
    fields[1] = 0.0 if power == 0 else power * r ** (power - 1)
    fields[2] = 0.0 if power < 2 else power * (power - 1) * r ** (power - 2)
    return fields


def _constant_envelope(r: np.ndarray) -> np.ndarray:
    fields = np.zeros((3, r.size), dtype=np.float64)
    fields[0] = 1.0
    return fields


def test_numba_profile_stage_passive_profile_matches_analytic_derivatives() -> None:
    r = np.linspace(0.0, 1.0, 12, dtype=np.float64)
    rp_fields = _rp_fields(r, power=3)
    env_fields = _constant_envelope(r)
    T = np.empty((0, r.size), dtype=np.float64)
    out = np.empty((3, r.size), dtype=np.float64)

    update_profile(out, T, T, T, rp_fields, env_fields, 1.75, None, 1.0)

    assert_allclose(out[0], 1.75 * r**3)
    assert_allclose(out[1], 1.75 * 3.0 * r**2)
    assert_allclose(out[2], 1.75 * 6.0 * r)


def test_numba_profile_stage_active_polynomial_profile_matches_product_rule() -> None:
    r = np.linspace(0.0, 1.0, 12, dtype=np.float64)
    rp_fields = _rp_fields(r, power=2)
    env_fields = _constant_envelope(r)
    T = np.vstack((np.ones_like(r), r, r**2))
    T_r = np.vstack((np.zeros_like(r), np.ones_like(r), 2.0 * r))
    T_rr = np.vstack((np.zeros_like(r), np.zeros_like(r), np.full_like(r, 2.0)))
    coeff = np.array([0.25, -0.5, 2.0], dtype=np.float64)
    out = np.empty((3, r.size), dtype=np.float64)

    update_profile(out, T, T_r, T_rr, rp_fields, env_fields, 0.75, coeff, 1.0)

    amp = 0.75 + coeff[0] + coeff[1] * r + coeff[2] * r**2
    amp_r = coeff[1] + 2.0 * coeff[2] * r
    amp_rr = np.full_like(r, 2.0 * coeff[2])
    assert_allclose(out[0], r**2 * amp)
    assert_allclose(out[1], 2.0 * r * amp + r**2 * amp_r)
    assert_allclose(out[2], 2.0 * amp + 4.0 * r * amp_r + r**2 * amp_rr)

