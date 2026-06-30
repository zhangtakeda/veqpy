"""
Module: math.quadrature

Role:
- Build pure radial quadrature nodes and weights for the unit interval.

Public API:
- make_quadrature
- quadrature_generator

Design notes:
- Builders return ``(rho, weights)`` on ``[0, 1]`` with weights already scaled
  for integration over that interval.  ``Grid`` combines these radial weights
  with the uniform poloidal rule when integrating 2D fields.
- Available schemes are registered by name: ``legendre`` (default), ``lobatto``,
  ``radau``, ``chebyshev``, and ``uniform``.  The scheme choice controls node
  placement and radial integration accuracy; it does not select interpolation,
  differentiation, source-route semantics, or solver behavior.
- This module is pure Python setup code.  Runtime Numba kernels consume the
  arrays produced through ``Grid`` or workspace snapshots; they do not call the
  quadrature builders directly.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal

from veqpy.base.registry import Registry

# -----------------------------------------------------------------------------
# Public implementation
# -----------------------------------------------------------------------------

Builder = Callable[[int], tuple[np.ndarray, np.ndarray]]
quadrature_generator: Registry[str, Builder] = Registry(str, Callable)

DEFAULT_QUADRATURE = "legendre"


def make_quadrature(
    n: int,
    *,
    scheme: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build quadrature ``(nodes, weights)`` for a quadrature scheme."""

    if scheme is None:
        scheme = DEFAULT_QUADRATURE

    scheme = scheme.lower()
    if scheme not in quadrature_generator:
        available = ", ".join(sorted(quadrature_generator.registry))
        raise ValueError(f"Unknown quadrature scheme: {scheme}. Available schemes: {available}")

    return quadrature_generator[scheme](n)


@quadrature_generator("chebyshev")
def chebyshev_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Chebyshev-distributed nodes and interpolatory weights on ``[0, 1]``."""

    k = np.arange(1, n + 1, dtype=np.float64)
    nodes = 0.5 * (np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))[::-1] + 1.0)
    weights = _interpolatory_quadrature_weights(nodes)
    return (
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


@quadrature_generator("legendre")
def legendre_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Gauss-Legendre nodes and weights on ``[0, 1]``."""

    nodes, weights = _golub_welsch_legendre(n)
    return (
        np.asarray(0.5 * (nodes + 1.0), dtype=np.float64),
        np.asarray(0.5 * weights, dtype=np.float64),
    )


@quadrature_generator("lobatto")
def lobatto_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Gauss-Lobatto nodes and weights on ``[0, 1]``."""

    nodes, weights = _golub_welsch_lobatto(n)
    return (
        np.asarray(0.5 * (nodes + 1.0), dtype=np.float64),
        np.asarray(0.5 * weights, dtype=np.float64),
    )


@quadrature_generator("radau")
def radau_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Gauss-Radau nodes and weights on ``[0, 1]``."""

    nodes, weights = _golub_welsch_radau(n)
    return (
        np.asarray(0.5 * (nodes + 1.0), dtype=np.float64),
        np.asarray(0.5 * weights, dtype=np.float64),
    )


@quadrature_generator("uniform")
def uniform_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build trapezoidal uniform nodes and weights on ``[0, 1]``."""
    rho = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    weights = np.full(n, h, dtype=np.float64)
    weights[0] = 0.5 * h
    weights[-1] = 0.5 * h
    return rho, weights


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _interpolatory_quadrature_weights(nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    vander = np.polynomial.legendre.legvander(2.0 * nodes - 1.0, n - 1)
    legendre_integrals = np.zeros(n, dtype=np.float64)
    legendre_integrals[0] = 2.0
    return 0.5 * np.linalg.solve(vander.T, legendre_integrals)


def _legendre_jacobi_coeffs(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag = np.zeros(n, dtype=np.float64)
    k = np.arange(1, n, dtype=np.float64)
    offdiag = k / np.sqrt(4.0 * k * k - 1.0)
    return diag, offdiag


def _golub_welsch_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag, offdiag = _legendre_jacobi_coeffs(n)
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights


def _golub_welsch_radau(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag, offdiag = _legendre_jacobi_coeffs(n)
    diag[-1] = n / (2.0 * n - 1.0)
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights


def _golub_welsch_lobatto(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag = np.zeros(n, dtype=np.float64)
    offdiag = np.empty(n - 1, dtype=np.float64)
    k = np.arange(1, n - 1, dtype=np.float64)
    offdiag[:-1] = k / np.sqrt(4.0 * k * k - 1.0)
    offdiag[-1] = np.sqrt((n - 1.0) / (2.0 * n - 3.0))
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights
