"""
Module: veqpy.numerics.interpolate

Role:
- Build pure one-dimensional interpolation weights and matrices.
- Normalize source-interpolation scheme names used by Kernel source setup.

Public API:
- barycentric_log_weights
- interpolation_matrix
- normalize_source_interpolation_kind
- source_interpolation_kind_is_barycentric
- build_uniform_source_interpolation_coefficients
- build_uniform_source_interpolation_matrix
- build_explicit_source_interpolation_coefficients

Design notes:
- Interpolation matrices use source nodes as columns and evaluation nodes as
  rows.  Multiplying a matrix by source samples evaluates the remapped samples at
  the target nodes.
- This module only remaps normalized one-dimensional parameters.  It does not
  define the physical meaning of a source route, choose an ``Ip``/``beta``
  constraint, or repair profile ownership; those decisions live in Kernel
  source planning and backend source kernels.
- Uniform-source helpers support the canonical source-remapping choices:
  ``barycentric`` (default local barycentric stencil), ``not-a-knot`` spline,
  ``linear``, ``quadratic``, and ``cubic``.  Names are normalized only by
  stripping whitespace, lowercasing, and treating underscores as hyphens.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from veqpy.base import Registry

DEFAULT_LOCAL_BARYCENTRIC_STENCIL = 8
SOURCE_INTERP_KIND_ENV = "VEQPY_SOURCE_INTERP_KIND"
SOURCE_INTERP_BARYCENTRIC = "barycentric"
SOURCE_INTERP_NOT_A_KNOT = "not-a-knot"
SOURCE_INTERP_LINEAR = "linear"
SOURCE_INTERP_QUADRATIC = "quadratic"
SOURCE_INTERP_CUBIC = "cubic"
SOURCE_INTERP_DEFAULT = SOURCE_INTERP_BARYCENTRIC

_SOURCE_INTERP_KINDS = frozenset(
    {
        SOURCE_INTERP_BARYCENTRIC,
        SOURCE_INTERP_NOT_A_KNOT,
        SOURCE_INTERP_LINEAR,
        SOURCE_INTERP_QUADRATIC,
        SOURCE_INTERP_CUBIC,
    }
)


CoefficientBuilder = Callable[[np.ndarray], np.ndarray]
MatrixBuilder = Callable[[np.ndarray, int], np.ndarray]

uniform_source_interpolation_generator: Registry[str, CoefficientBuilder] = Registry(str, Callable)
uniform_source_interpolation_matrix_generator: Registry[str, MatrixBuilder] = Registry(
    str, Callable
)
_BARYCENTRIC_INTERPOLATION_SCHEMES: set[str] = set()

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def barycentric_log_weights(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build signed logarithmic barycentric weights for distinct nodes."""

    nodes = np.asarray(nodes, dtype=np.float64)
    diff = nodes[:, None] - nodes[None, :]
    mask = ~np.eye(len(nodes), dtype=bool)
    if np.any(np.isclose(diff[mask], 0.0)):
        raise ValueError("nodes must be distinct")

    abs_diff = np.abs(diff)
    sign_diff = np.sign(diff)
    np.fill_diagonal(abs_diff, 1.0)
    np.fill_diagonal(sign_diff, 1.0)

    # Products of pairwise distances overflow quickly for spectral grids.  Keep
    # signed log-weights and exponentiate only local ratios at evaluation time.
    log_abs_prod = np.sum(np.log(abs_diff), axis=1)
    signs = np.prod(sign_diff, axis=1)
    return signs, -log_abs_prod


def interpolation_matrix(source_nodes: np.ndarray, evaluation_nodes: np.ndarray) -> np.ndarray:
    """Build the Lagrange interpolation matrix from source to evaluation nodes."""

    source_nodes = np.asarray(source_nodes, dtype=np.float64)
    evaluation_nodes = np.asarray(evaluation_nodes, dtype=np.float64)
    signs, log_weights = barycentric_log_weights(source_nodes)
    matrix = np.empty((evaluation_nodes.size, source_nodes.size), dtype=np.float64)

    for i, node in enumerate(evaluation_nodes):
        diff = node - source_nodes
        exact = np.flatnonzero(np.isclose(diff, 0.0))
        if exact.size:
            matrix[i].fill(0.0)
            matrix[i, exact[0]] = 1.0
            continue

        log_terms = log_weights - np.log(np.abs(diff))
        scale = np.max(log_terms)
        # Rescale the barycentric numerator/denominator by the largest log term
        # so exact high-order interpolation remains finite.
        terms = signs * np.sign(diff) * np.exp(log_terms - scale)
        matrix[i] = terms / np.sum(terms)

    return matrix


def normalize_source_interpolation_kind(kind: str | None = None) -> str:
    """Normalize canonical uniform-source interpolation scheme names."""

    raw = os.environ.get(SOURCE_INTERP_KIND_ENV, SOURCE_INTERP_DEFAULT) if kind is None else kind
    key = str(raw).strip().lower().replace("_", "-")
    if key not in _SOURCE_INTERP_KINDS:
        supported = ", ".join(sorted(uniform_source_interpolation_generator.registry))
        raise ValueError(f"Unsupported {SOURCE_INTERP_KIND_ENV}={raw!r}; supported: {supported}")
    if key not in uniform_source_interpolation_generator:
        raise ValueError(f"Uniform source interpolation scheme {key!r} is not registered")
    return key


def source_interpolation_kind_is_barycentric(kind: str | None = None) -> bool:
    """Return whether a source interpolation kind uses barycentric runtime evaluation."""

    normalized = normalize_source_interpolation_kind(kind)
    return normalized in _BARYCENTRIC_INTERPOLATION_SCHEMES


def _build_uniform_local_polynomial_coefficients(values: np.ndarray, degree: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"Expected 1D source samples, got {samples.shape}")
    n = int(samples.shape[0])
    if n < 1:
        raise ValueError("source samples must be non-empty")

    interval_count = max(n - 1, 1)
    coeff = np.zeros((interval_count, 4), dtype=np.float64)
    if n == 1:
        coeff[0, 0] = samples[0]
        return coeff

    local_degree = min(max(int(degree), 1), 3, n - 1)
    # Local schemes degrade to the highest supported degree for the available
    # sample count instead of failing small GEQDSK/profile inputs.
    if local_degree == 1:
        for interval in range(interval_count):
            y0 = samples[interval]
            y1 = samples[interval + 1]
            coeff[interval, 0] = y0
            coeff[interval, 1] = y1 - y0
        return coeff

    if local_degree == 2:
        for interval in range(interval_count):
            if interval == 0:
                y0 = samples[0]
                y1 = samples[1]
                y2 = samples[2]
                coeff[interval, 0] = y0
                coeff[interval, 1] = -1.5 * y0 + 2.0 * y1 - 0.5 * y2
                coeff[interval, 2] = 0.5 * y0 - y1 + 0.5 * y2
            else:
                y0 = samples[interval - 1]
                y1 = samples[interval]
                y2 = samples[interval + 1]
                coeff[interval, 0] = y1
                coeff[interval, 1] = -0.5 * y0 + 0.5 * y2
                coeff[interval, 2] = 0.5 * y0 - y1 + 0.5 * y2
        return coeff

    for interval in range(interval_count):
        if interval == 0:
            # Boundary intervals use one-sided finite-difference polynomials;
            # interior intervals use centered local stencils below.
            y0 = samples[0]
            y1 = samples[1]
            y2 = samples[2]
            y3 = samples[3]
            coeff[interval, 0] = y0
            coeff[interval, 1] = (-11.0 * y0 + 18.0 * y1 - 9.0 * y2 + 2.0 * y3) / 6.0
            coeff[interval, 2] = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
            coeff[interval, 3] = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0
        elif interval == interval_count - 1:
            y0 = samples[interval - 2]
            y1 = samples[interval - 1]
            y2 = samples[interval]
            y3 = samples[interval + 1]
            coeff[interval, 0] = y2
            coeff[interval, 1] = (y0 - 6.0 * y1 + 3.0 * y2 + 2.0 * y3) / 6.0
            coeff[interval, 2] = 0.5 * y1 - y2 + 0.5 * y3
            coeff[interval, 3] = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0
        else:
            y0 = samples[interval - 1]
            y1 = samples[interval]
            y2 = samples[interval + 1]
            y3 = samples[interval + 2]
            coeff[interval, 0] = y1
            coeff[interval, 1] = (-2.0 * y0 - 3.0 * y1 + 6.0 * y2 - y3) / 6.0
            coeff[interval, 2] = 0.5 * y0 - y1 + 0.5 * y2
            coeff[interval, 3] = (-y0 + 3.0 * y1 - 3.0 * y2 + y3) / 6.0
    return coeff


def _build_uniform_not_a_knot_spline_coefficients(values: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"Expected 1D source samples, got {samples.shape}")
    n = int(samples.shape[0])
    if n < 1:
        raise ValueError("source samples must be non-empty")
    if n < 4:
        # Not-a-knot needs enough intervals to express endpoint constraints; for
        # small inputs it is equivalent to the supported local polynomial fit.
        return _build_uniform_local_polynomial_coefficients(samples, min(n - 1, 3))

    # Solve for second derivatives with not-a-knot endpoint constraints; the
    # coefficients are stored in normalized interval coordinates t in [0, 1].
    h = 1.0 / (n - 1.0)
    h2 = h * h
    matrix = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[0, 1] = -2.0
    matrix[0, 2] = 1.0
    matrix[-1, -3] = 1.0
    matrix[-1, -2] = -2.0
    matrix[-1, -1] = 1.0
    for row in range(1, n - 1):
        matrix[row, row - 1] = 1.0
        matrix[row, row] = 4.0
        matrix[row, row + 1] = 1.0
        rhs[row] = 6.0 * (samples[row + 1] - 2.0 * samples[row] + samples[row - 1]) / h2

    second = np.linalg.solve(matrix, rhs)
    coeff = np.empty((n - 1, 4), dtype=np.float64)
    for interval in range(n - 1):
        left_second = second[interval]
        right_second = second[interval + 1]
        coeff[interval, 0] = samples[interval]
        coeff[interval, 1] = (
            samples[interval + 1]
            - samples[interval]
            - h2 * (2.0 * left_second + right_second) / 6.0
        )
        coeff[interval, 2] = 0.5 * h2 * left_second
        coeff[interval, 3] = h2 * (right_second - left_second) / 6.0
    return coeff


def _evaluate_uniform_coefficients(coeff: np.ndarray, query: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    evaluation = np.asarray(query, dtype=np.float64)
    out = np.empty(evaluation.shape[0], dtype=np.float64)
    interval_count = coeff.shape[0]
    if interval_count == 1:
        for i, q_raw in enumerate(evaluation):
            # Clamp source queries to the tabulated physical interval.  Source
            # remapping should not extrapolate beyond [0, 1].
            q = min(max(float(q_raw), 0.0), 1.0)
            out[i] = ((coeff[0, 3] * q + coeff[0, 2]) * q + coeff[0, 1]) * q + coeff[0, 0]
        return out

    denom_scale = float(interval_count)
    last_interval = interval_count - 1
    for i, q_raw in enumerate(evaluation):
        q = min(max(float(q_raw), 0.0), 1.0)
        position = q * denom_scale
        interval = int(position)
        if interval > last_interval:
            # q==1 lands on the right edge of the last interval, not a new one.
            interval = last_interval
            t = 1.0
        else:
            t = position - interval
        out[i] = (
            (coeff[interval, 3] * t + coeff[interval, 2]) * t + coeff[interval, 1]
        ) * t + coeff[interval, 0]
    return out


def uniform_barycentric_weights(source_sample_count: int) -> np.ndarray:
    """Build barycentric weights for a uniform local stencil."""

    count = int(source_sample_count)
    if count < 1:
        raise ValueError(f"source_sample_count must be positive, got {source_sample_count!r}")
    weights = np.empty(count, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, count):
        weights[j] = -weights[j - 1] * (count - j) / j
    return weights


def _local_uniform_stencil_start(q: float, source_sample_count: int, stencil_size: int) -> int:
    if stencil_size >= source_sample_count:
        return 0
    pos = q * (source_sample_count - 1.0)
    center = int(pos)
    if pos > center:
        center += 1
    start = center - stencil_size // 2
    if start < 0:
        return 0
    max_start = source_sample_count - stencil_size
    if start > max_start:
        return max_start
    # Prefer a centered local stencil, then clamp near endpoints so the stencil
    # size stays fixed and barycentric weights can be reused.
    return start


def _build_uniform_barycentric_matrix(
    query: np.ndarray,
    source_sample_count: int,
    stencil_size: int,
) -> np.ndarray:
    evaluation = np.asarray(query, dtype=np.float64)
    matrix = np.empty((evaluation.shape[0], source_sample_count), dtype=np.float64)
    if source_sample_count == 1:
        matrix[:, 0] = 1.0
        return matrix

    weights = uniform_barycentric_weights(stencil_size)
    for i, q_raw in enumerate(evaluation):
        q = min(max(float(q_raw), 0.0), 1.0)
        matrix[i].fill(0.0)
        start = _local_uniform_stencil_start(q, source_sample_count, stencil_size)
        hit = False
        for local_j in range(stencil_size):
            j = start + local_j
            diff = q - j / (source_sample_count - 1.0)
            if abs(diff) <= 1.0e-14:
                matrix[i, j] = 1.0
                hit = True
                break
        if hit:
            continue

        denominator = 0.0
        for local_j in range(stencil_size):
            j = start + local_j
            denominator += weights[local_j] / (q - j / (source_sample_count - 1.0))
        for local_j in range(stencil_size):
            j = start + local_j
            term = weights[local_j] / (q - j / (source_sample_count - 1.0))
            matrix[i, j] = term / denominator
    return matrix


def build_uniform_source_interpolation_coefficients(
    values: np.ndarray, *, kind: str | None = None
) -> np.ndarray:
    """Precompute coefficients for registered coefficient-based uniform schemes."""

    normalized = normalize_source_interpolation_kind(kind)
    return uniform_source_interpolation_generator[normalized](values)


def build_uniform_source_interpolation_matrix(
    query: np.ndarray,
    source_sample_count: int,
    *,
    kind: str | None = None,
) -> np.ndarray:
    """Build a uniform-source remap matrix using a registered interpolation scheme."""

    evaluation = np.asarray(query, dtype=np.float64)
    if evaluation.ndim != 1:
        raise ValueError(f"Expected 1D query nodes, got {evaluation.shape}")
    count = int(source_sample_count)
    if count < 1:
        raise ValueError(f"source_sample_count must be positive, got {source_sample_count!r}")

    normalized = normalize_source_interpolation_kind(kind)
    if normalized in uniform_source_interpolation_matrix_generator:
        return uniform_source_interpolation_matrix_generator[normalized](evaluation, count)

    matrix = np.empty((evaluation.shape[0], count), dtype=np.float64)
    basis = np.zeros(count, dtype=np.float64)
    for j in range(count):
        # Schemes without a direct matrix builder get one by applying the
        # coefficient evaluator to each source basis vector.
        basis[j] = 1.0
        matrix[:, j] = _evaluate_uniform_coefficients(
            uniform_source_interpolation_generator[normalized](basis),
            evaluation,
        )
        basis[j] = 0.0
    return matrix


def build_explicit_source_interpolation_coefficients(
    nodes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Build shape-preserving cubic coefficients on arbitrary source nodes.

    Rows store ascending powers of ``dx = query - nodes[interval]`` so the
    Numba hot path can evaluate them without SciPy objects or allocations.
    """

    from scipy.interpolate import PchipInterpolator

    axis = np.asarray(nodes, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or samples.ndim != 1 or axis.shape != samples.shape:
        raise ValueError(f"Expected matching 1D nodes/values, got {axis.shape} and {samples.shape}")
    if axis.size < 2:
        raise ValueError("explicit source interpolation requires at least two nodes")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("explicit source nodes must be strictly increasing")
    # scipy stores descending powers [cubic, quadratic, linear, constant].
    coefficients = PchipInterpolator(axis, samples, extrapolate=False).c[::-1].T
    return np.ascontiguousarray(coefficients, dtype=np.float64)


@uniform_source_interpolation_generator(SOURCE_INTERP_BARYCENTRIC)
def barycentric_source_interpolation_coefficients(values: np.ndarray) -> np.ndarray:
    """Build local barycentric-style coefficients for uniform source samples."""
    return _build_uniform_local_polynomial_coefficients(values, 3)


@uniform_source_interpolation_matrix_generator(SOURCE_INTERP_BARYCENTRIC)
def barycentric_source_interpolation_matrix(query: np.ndarray, count: int) -> np.ndarray:
    """Build the registered barycentric remap matrix for uniform source samples."""
    return _build_uniform_barycentric_matrix(
        query,
        count,
        min(count, DEFAULT_LOCAL_BARYCENTRIC_STENCIL),
    )


_BARYCENTRIC_INTERPOLATION_SCHEMES.add(SOURCE_INTERP_BARYCENTRIC)


@uniform_source_interpolation_generator(SOURCE_INTERP_NOT_A_KNOT)
def not_a_knot_source_interpolation_coefficients(values: np.ndarray) -> np.ndarray:
    """Build not-a-knot spline coefficients for uniform source samples."""
    return _build_uniform_not_a_knot_spline_coefficients(values)


@uniform_source_interpolation_generator(SOURCE_INTERP_LINEAR)
def linear_source_interpolation_coefficients(values: np.ndarray) -> np.ndarray:
    """Build linear local-polynomial coefficients for uniform source samples."""
    return _build_uniform_local_polynomial_coefficients(values, 1)


@uniform_source_interpolation_generator(SOURCE_INTERP_QUADRATIC)
def quadratic_source_interpolation_coefficients(values: np.ndarray) -> np.ndarray:
    """Build quadratic local-polynomial coefficients for uniform source samples."""
    return _build_uniform_local_polynomial_coefficients(values, 2)


@uniform_source_interpolation_generator(SOURCE_INTERP_CUBIC)
def cubic_source_interpolation_coefficients(values: np.ndarray) -> np.ndarray:
    """Build cubic local-polynomial coefficients for uniform source samples."""
    return _build_uniform_local_polynomial_coefficients(values, 3)
