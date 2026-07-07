"""
Module: veqpy.kernels.numba_kernel.jit_math

Role:
- Provide small Numba kernels for reusable dense float64 math operations.
- All `*_into` functions write through their first `out` argument and return None.
- Kernels do not allocate memory internally.

Public API:
- matmul_into
- matvec_into
- indexed_matvec_into
- dot
- weighted_dot
- weighted_ratio_dot
- rowwise_sum_into
- rowwise_weighted_sum_into
- rowwise_dot_into
- colwise_sum_into
- colwise_weighted_sum_into
- colwise_dot_into
- copy_into
- product_into
- scale_into
- scaled_product_into
- scaled_ratio_into
- scaled_product_ratio_into
- maximum_floor_into
"""

from __future__ import annotations

from collections.abc import Callable

import numba as nb
import numpy as np

# -----------------------------------------------------------------------------
# Public implementation
# -----------------------------------------------------------------------------


scalar = nb.float64
array = nb.types.Array(nb.float64, 1, "C")
matrix = nb.types.Array(nb.float64, 2, "C")
indices = nb.types.Array(nb.intp, 1, "C")
# Readonly signatures let Numba optimize callers that pass shared workspace
# inputs while still requiring explicit writable ``out`` arrays for results.
const_array = nb.types.Array(nb.float64, 1, "C", readonly=True)
const_matrix = nb.types.Array(nb.float64, 2, "C", readonly=True)
const_indices = nb.types.Array(nb.intp, 1, "C", readonly=True)


def fast_kernel(signature) -> Callable:
    """Decorator for small fast Numba kernels."""
    # inline="always" keeps these helpers cheap inside larger hot kernels; the
    # functions are intentionally tiny and allocation-free.
    return nb.njit(signature, cache=True, nogil=True, fastmath=True, inline="always")


@fast_kernel(nb.void(matrix, const_matrix, const_matrix))
def matmul_into(out: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> None:
    """Compute `out = lhs @ rhs` for dense 2D float64 arrays."""
    n_rows = lhs.shape[0]
    n_inner = lhs.shape[1]
    n_cols = rhs.shape[1]

    for i in range(n_rows):
        # Clear row before accumulation so callers can reuse scratch matrices
        # without paying for a separate fill.
        for j in range(n_cols):
            out[i, j] = 0.0

        for k in range(n_inner):
            lhs_ik = lhs[i, k]
            for j in range(n_cols):
                out[i, j] += lhs_ik * rhs[k, j]


@fast_kernel(nb.void(array, const_matrix, const_array))
def matvec_into(out: np.ndarray, mat: np.ndarray, vec: np.ndarray) -> None:
    """Compute `out = mat @ vec` for a dense 2D matrix and 1D vector."""
    n_rows = mat.shape[0]
    n_cols = mat.shape[1]

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += mat[i, j] * vec[j]
        out[i] = total


@fast_kernel(nb.void(array, const_indices, const_matrix, const_array))
def indexed_matvec_into(
    out: np.ndarray,
    out_indices: np.ndarray,
    mat: np.ndarray,
    vec: np.ndarray,
) -> None:
    """Compute `out[out_indices[i]] = sum_j mat[i, j] * vec[j]`."""
    n_rows = out_indices.shape[0]
    n_cols = vec.shape[0]

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += mat[i, j] * vec[j]
        out[out_indices[i]] = total


@fast_kernel(scalar(const_array, const_array))
def dot(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return `sum_i lhs[i] * rhs[i]`."""
    total = 0.0

    for i in range(lhs.shape[0]):
        total += lhs[i] * rhs[i]

    return total


@fast_kernel(scalar(const_array, const_array, const_array))
def weighted_dot(
    lhs: np.ndarray,
    rhs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return `sum_i weights[i] * lhs[i] * rhs[i]`."""
    total = 0.0

    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i]

    return total


@fast_kernel(scalar(const_array, const_array, const_array, const_array))
def weighted_ratio_dot(
    lhs: np.ndarray,
    rhs: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return `sum_i weights[i] * lhs[i] * rhs[i] / denominator[i]`."""
    total = 0.0

    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i] / denominator[i]

    return total


@fast_kernel(nb.void(array, const_matrix))
def rowwise_sum_into(out: np.ndarray, values: np.ndarray) -> None:
    """Compute `out[i] = sum_j values[i, j]`."""
    n_rows = values.shape[0]
    n_cols = values.shape[1]

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += values[i, j]
        out[i] = total


@fast_kernel(nb.void(array, const_matrix, const_array))
def rowwise_weighted_sum_into(
    out: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Compute `out[i] = sum_j values[i, j] * weights[j]`."""
    n_rows = values.shape[0]
    n_cols = values.shape[1]

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += values[i, j] * weights[j]
        out[i] = total


@fast_kernel(nb.void(array, const_matrix, const_matrix))
def rowwise_dot_into(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Compute `out[i] = sum_j lhs[i, j] * rhs[i, j]`."""
    n_rows = lhs.shape[0]
    n_cols = lhs.shape[1]

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += lhs[i, j] * rhs[i, j]
        out[i] = total


@fast_kernel(nb.void(array, const_matrix))
def colwise_sum_into(out: np.ndarray, values: np.ndarray) -> None:
    """Compute `out[j] = sum_i values[i, j]`."""
    n_rows = values.shape[0]
    n_cols = values.shape[1]

    for j in range(n_cols):
        # Column reductions are written in two passes so ``out`` never carries
        # data from a previous workspace use.
        out[j] = 0.0

    for i in range(n_rows):
        for j in range(n_cols):
            out[j] += values[i, j]


@fast_kernel(nb.void(array, const_matrix, const_array))
def colwise_weighted_sum_into(
    out: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Compute `out[j] = sum_i weights[i] * values[i, j]`."""
    n_rows = values.shape[0]
    n_cols = values.shape[1]

    for j in range(n_cols):
        out[j] = 0.0

    for i in range(n_rows):
        weight_i = weights[i]
        for j in range(n_cols):
            out[j] += weight_i * values[i, j]


@fast_kernel(nb.void(array, const_matrix, const_matrix))
def colwise_dot_into(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Compute `out[j] = sum_i lhs[i, j] * rhs[i, j]`."""
    n_rows = lhs.shape[0]
    n_cols = lhs.shape[1]

    for j in range(n_cols):
        out[j] = 0.0

    for i in range(n_rows):
        for j in range(n_cols):
            out[j] += lhs[i, j] * rhs[i, j]


@fast_kernel(nb.void(array, const_array))
def copy_into(out: np.ndarray, src: np.ndarray) -> None:
    """Compute `out[i] = src[i]`."""
    for i in range(out.shape[0]):
        out[i] = src[i]


@fast_kernel(nb.void(array, const_array, const_array))
def product_into(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Compute `out[i] = lhs[i] * rhs[i]`."""
    for i in range(out.shape[0]):
        out[i] = lhs[i] * rhs[i]


@fast_kernel(nb.void(array, const_array, scalar))
def scale_into(
    out: np.ndarray,
    src: np.ndarray,
    scale: float,
) -> None:
    """Compute `out[i] = scale * src[i]`."""
    for i in range(out.shape[0]):
        out[i] = scale * src[i]


@fast_kernel(nb.void(array, const_array, const_array, scalar))
def scaled_product_into(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    scale: float,
) -> None:
    """Compute `out[i] = scale * lhs[i] * rhs[i]`."""
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i]


@fast_kernel(nb.void(array, const_array, const_array, scalar))
def scaled_ratio_into(
    out: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    scale: float,
) -> None:
    """Compute `out[i] = scale * numerator[i] / denominator[i]`."""
    for i in range(out.shape[0]):
        out[i] = scale * numerator[i] / denominator[i]


@fast_kernel(nb.void(array, const_array, const_array, const_array, scalar))
def scaled_product_ratio_into(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    denominator: np.ndarray,
    scale: float,
) -> None:
    """Compute `out[i] = scale * lhs[i] * rhs[i] / denominator[i]`."""
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i] / denominator[i]


@fast_kernel(nb.void(array, const_array, scalar))
def maximum_floor_into(
    out: np.ndarray,
    src: np.ndarray,
    floor_value: float,
) -> None:
    """Compute `out[i] = max(src[i], floor_value)`."""
    # Used for current-like profiles where the cumulative primitive must remain
    # positive enough for later divisions; it is not a generic clipping policy.
    for i in range(src.shape[0]):
        value = src[i]
        out[i] = value if value > floor_value else floor_value
