"""
Module: math.calculus

Role:
- Build radial prefix-integration and differentiation matrices.
- Own the registry that selects base radial calculus schemes.

Public API:
- make_calculus
- apply_differentiation
- apply_accumulation

Design notes:
- ``make_calculus(nodes)`` returns ``(accumulator, differentiator)`` for fields
  sampled on a one-dimensional radial node set.  These matrices are reusable
  linear operators; they do not know whether a field represents geometry,
  source data, diagnostics, or a residual ingredient.
- The accumulator is a prefix integral from the magnetic-axis-side origin.  Its
  additive constant is fixed by the value-at-zero convention so integrating a
  derivative reconstructs a profile up to that boundary value.
- The default ``spectral`` scheme uses compact finite differences on uniform
  nodes and global barycentric/spectral matrices on nonuniform nodes.  Additional
  compact finite-difference schemes are registered as ``compact``/``cfd33``,
  ``cfd35``, and ``cfd55``.
- Source-route meaning, profile ownership, and solver policy are outside this
  module; callers decide which fields to differentiate or accumulate.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from veqpy.base import Registry
from veqpy.model.numerics.interpolate import barycentric_log_weights, interpolation_matrix

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

CalculusBuilder = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
calculus_generator: Registry[str, CalculusBuilder] = Registry(str, Callable)

DEFAULT_CALCULUS = "spectral"


def make_calculus(
    nodes: np.ndarray,
    *,
    scheme: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(accumulator, differentiator)`` for a calculus scheme."""

    if scheme is None:
        scheme = DEFAULT_CALCULUS

    scheme = scheme.lower()
    if scheme not in calculus_generator:
        available = ", ".join(sorted(calculus_generator.registry))
        raise ValueError(f"Unknown calculus scheme: {scheme}. Available schemes: {available}")
    return calculus_generator[scheme](nodes)


def apply_differentiation(
    out: np.ndarray,
    arr: np.ndarray,
    differentiator: np.ndarray,
) -> np.ndarray:
    """Apply a radial differentiation matrix to a 1D field."""
    np.matmul(differentiator, arr, out=out)
    return out


def apply_accumulation(
    out: np.ndarray,
    arr: np.ndarray,
    accumulator: np.ndarray,
) -> np.ndarray:
    """Apply a radial prefix-integration matrix to a 1D field."""
    np.matmul(accumulator, arr, out=out)
    return out


@calculus_generator("spectral")
def spectral_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build base integration/differentiation matrices for the spectral scheme."""

    _validate_nodes(nodes, min_size=4)
    if _has_uniform_spacing(nodes):
        # Uniform nodes use compact finite differences for differentiation; the
        # global polynomial spectral matrix is mainly for nonuniform quadrature
        # nodes such as Legendre/Radau/Lobatto.
        return (
            _uniform_accumulator(nodes.shape[0]),
            _cfd33_differentiator(nodes),
        )
    return (
        _spectral_accumulator(nodes),
        _spectral_differentiator(nodes),
    )


@calculus_generator("compact", "cfd33")
def compact_cfd33_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build dense CFD33 compact integration and differentiation matrices."""

    _validate_nodes(nodes, min_size=4)
    return (
        _cfd33_accumulator(nodes),
        _cfd33_differentiator(nodes),
    )


@calculus_generator("cfd35")
def compact_cfd35_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build dense CFD35 compact integration and differentiation matrices."""

    _validate_nodes(nodes, min_size=5)
    return _compact_calculus(nodes, implicit_width=3, explicit_width=5)


@calculus_generator("cfd55")
def compact_cfd55_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build dense CFD55 compact integration and differentiation matrices."""

    _validate_nodes(nodes, min_size=5)
    return _compact_calculus(nodes, implicit_width=5, explicit_width=5)


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _cfd33_matrices(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build non-uniform CFD33 matrices ``A`` and ``B`` for ``A @ u_r == B @ u``."""

    return _compact_matrices(
        nodes,
        implicit_width=3,
        explicit_width=3,
    )


def _compact_calculus(
    nodes: np.ndarray, *, implicit_width: int, explicit_width: int
) -> tuple[np.ndarray, np.ndarray]:
    a_matrix, b_matrix = _compact_matrices(
        nodes,
        implicit_width=implicit_width,
        explicit_width=explicit_width,
    )
    return (
        _compact_accumulator(nodes, a_matrix, b_matrix),
        # Pre-eliminate A once so runtime differentiation is a matrix-vector
        # product, matching the spectral scheme interface.
        np.linalg.solve(a_matrix, b_matrix),
    )


def _compact_matrices(
    nodes: np.ndarray, *, implicit_width: int, explicit_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact matrices ``A`` and ``B`` for ``A @ u_r == B @ u``."""

    if implicit_width % 2 != 1 or explicit_width % 2 != 1:
        raise ValueError("compact stencil widths must be odd")

    n = nodes.shape[0]
    a_matrix = np.zeros((n, n), dtype=np.float64)
    b_matrix = np.zeros((n, n), dtype=np.float64)
    implicit_radius = implicit_width // 2
    explicit_radius = explicit_width // 2
    boundary_width = min(n, implicit_width + explicit_width - 1)

    for row in range(n):
        implicit_start = row - implicit_radius
        implicit_stop = row + implicit_radius + 1
        explicit_start = row - explicit_radius
        explicit_stop = row + explicit_radius + 1

        if (
            implicit_start >= 0
            and implicit_stop <= n
            and explicit_start >= 0
            and explicit_stop <= n
        ):
            # Interior rows use compact moment-matched stencils.  Boundary rows
            # fall back to explicit finite differences to avoid one-sided
            # implicit systems with weak support.
            implicit_indices = np.arange(implicit_start, implicit_stop, dtype=np.int64)
            explicit_indices = np.arange(explicit_start, explicit_stop, dtype=np.int64)
            a_values, b_values = _compact_row_coefficients(
                nodes,
                row,
                implicit_indices,
                explicit_indices,
            )
            a_matrix[row, implicit_indices] = a_values
            b_matrix[row, explicit_indices] = b_values
            continue

        stencil_start = min(max(row - boundary_width // 2, 0), n - boundary_width)
        stencil_stop = stencil_start + boundary_width
        stencil_indices = np.arange(stencil_start, stencil_stop, dtype=np.int64)
        a_matrix[row, row] = 1.0
        b_matrix[row, stencil_indices] = _finite_difference_weights(
            nodes[stencil_indices],
            nodes[row],
            derivative_order=1,
        )

    return a_matrix, b_matrix


def _compact_row_coefficients(
    nodes: np.ndarray,
    row: int,
    implicit_indices: np.ndarray,
    explicit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one local compact row by polynomial moment matching."""

    implicit_offsets = nodes[implicit_indices] - nodes[row]
    explicit_offsets = nodes[explicit_indices] - nodes[row]
    center_matches = np.flatnonzero(implicit_indices == row)
    if center_matches.size != 1:
        raise ValueError("implicit compact stencil must contain the target row")

    center = int(center_matches[0])
    implicit_unknowns = [index for index in range(implicit_indices.shape[0]) if index != center]
    unknown_count = len(implicit_unknowns) + explicit_indices.shape[0]
    system = np.zeros((unknown_count, unknown_count), dtype=np.float64)
    rhs = np.zeros(unknown_count, dtype=np.float64)

    for degree in range(unknown_count):
        for column, index in enumerate(implicit_unknowns):
            system[degree, column] = (
                degree * implicit_offsets[index] ** (degree - 1) if degree > 0 else 0.0
            )
        explicit_offset = len(implicit_unknowns)
        system[degree, explicit_offset:] = -(explicit_offsets**degree)
        rhs[degree] = -1.0 if degree == 1 else 0.0

    solution = np.linalg.solve(system, rhs)
    a_values = np.zeros(implicit_indices.shape[0], dtype=np.float64)
    a_values[center] = 1.0
    for value, index in zip(solution[: len(implicit_unknowns)], implicit_unknowns, strict=True):
        a_values[index] = value
    return a_values, solution[len(implicit_unknowns) :]


def _cfd33_differentiator(nodes: np.ndarray) -> np.ndarray:
    """Build the dense pre-eliminated CFD33 derivative matrix."""

    a_matrix, b_matrix = _cfd33_matrices(nodes)
    return np.linalg.solve(a_matrix, b_matrix)


def _cfd33_accumulator(nodes: np.ndarray) -> np.ndarray:
    """Build the dense CFD33 variable-limit integration matrix with ``v(0) == 0``."""

    a_matrix, b_matrix = _cfd33_matrices(nodes)
    return _compact_accumulator(nodes, a_matrix, b_matrix)


def _compact_accumulator(
    nodes: np.ndarray, a_matrix: np.ndarray, b_matrix: np.ndarray
) -> np.ndarray:
    """Build a variable-limit integration matrix from compact derivative matrices."""

    system = b_matrix.copy()
    rhs_matrix = a_matrix.copy()

    constraint_row = int(np.argmin(np.abs(nodes)))
    # Differentiation matrices are singular up to an additive constant; replacing
    # one row with v(0)=0 turns the inverse into a prefix-integration operator.
    system[constraint_row, :] = interpolation_matrix(nodes, np.array([0.0], dtype=np.float64))[0]
    rhs_matrix[constraint_row, :] = 0.0
    return np.linalg.solve(system, rhs_matrix)


def _spectral_differentiator(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial collocation first-derivative matrix."""

    nodes = np.asarray(nodes, dtype=np.float64)
    signs, log_weights = barycentric_log_weights(nodes)
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    log_ratio = log_weights[None, :] - log_weights[:, None]
    mag_ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
    # The diagonal is filled from the zero-row-sum identity after off-diagonal
    # barycentric entries are scaled into a finite exponent range.
    matrix = signs[None, :] * signs[:, None] * mag_ratio / diff
    np.fill_diagonal(matrix, 0.0)
    matrix[np.diag_indices_from(matrix)] = -np.sum(matrix, axis=1)
    return matrix


def _spectral_accumulator(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial matrix for integrals from zero to each node."""

    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    xg = 2.0 * nodes - 1.0
    # Build the prefix integral in a Legendre basis on [-1, 1], then solve for
    # nodal weights that reproduce the antiderivative at every radial node.
    full_legendre = np.polynomial.legendre.legvander(xg, n)
    legendre = full_legendre[:, :n]

    antiderivative = np.zeros((n, n), dtype=np.float64)
    antiderivative[:, 0] = 0.5 * (xg + 1.0)
    degrees = np.arange(1, n, dtype=np.float64)
    antiderivative[:, 1:] = (
        0.5 * (full_legendre[:, 2 : n + 1] - full_legendre[:, : n - 1]) / (2.0 * degrees + 1.0)
    )

    lower_legendre = np.polynomial.legendre.legvander(np.array([-1.0]), n)[0]
    lower_antiderivative = np.zeros(n, dtype=np.float64)
    lower_antiderivative[1:] = (
        0.5 * (lower_legendre[2 : n + 1] - lower_legendre[: n - 1]) / (2.0 * degrees + 1.0)
    )

    rhs = (antiderivative - lower_antiderivative[None, :]).T
    cond = np.linalg.cond(legendre)
    if np.isfinite(cond) and cond <= 1.0 / np.sqrt(np.finfo(np.float64).eps):
        return np.linalg.solve(legendre.T, rhs).T
    # High-order or clustered nodes can make the Vandermonde solve borderline;
    # least squares gives a stable accumulator instead of throwing away the grid.
    return np.linalg.lstsq(legendre.T, rhs, rcond=None)[0].T


def _uniform_accumulator(n: int) -> np.ndarray:
    """Build the trapezoidal variable-limit integration matrix."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        # Row i integrates from node 0 to node i.  Rows remain prefix operators,
        # not cumulative sums over every previous call.
        matrix[i, 0] = 0.5 * h
        matrix[i, i] = 0.5 * h
        if i > 1:
            matrix[i, 1:i] = h
    return matrix


def _validate_nodes(
    nodes: np.ndarray,
    *,
    min_size: int = 4,
):
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("nodes must be a one-dimensional array")
    if nodes.shape[0] < min_size:
        raise ValueError(f"require at least {min_size} nodes")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("nodes must be strictly increasing")


def _finite_difference_weights(
    stencil_nodes: np.ndarray, target: float, *, derivative_order: int
) -> np.ndarray:
    offsets = np.asarray(stencil_nodes, dtype=np.float64) - float(target)
    n = offsets.shape[0]
    powers = offsets[None, :] ** np.arange(n, dtype=np.float64)[:, None]
    rhs = np.zeros(n, dtype=np.float64)
    rhs[derivative_order] = float(math.factorial(derivative_order))
    return np.linalg.solve(powers, rhs)


def _has_uniform_spacing(nodes: np.ndarray) -> bool:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.shape[0] < 2:
        return False
    spacing = np.diff(nodes)
    return bool(np.all(np.abs(spacing - spacing[0]) < 1.0e-6))
