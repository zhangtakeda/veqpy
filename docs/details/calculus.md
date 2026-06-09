# Calculus

The `calculus` module builds radial differentiation and integration matrices. Given nodes $0\le x_0<\cdots<x_{n-1}\le1$, it returns two linear operators:

$$
u_\rho \approx D u,\qquad
\int_0^\rho f(s)\,ds \approx G f .
$$

These matrices are owned by `Grid` and reused by geometry, source profiles, diagnostics, and residual projection. This page describes the meaning of the available schemes; implementation details live in `veqpy/math/calculus.py`.

## Scheme

| Scheme | Alias | Meaning |
| ------ | ----- | ------- |
| Spectral Difference | `spectral` | Dense differentiation/integration matrices induced by global Lagrange interpolation |
| CFD33 | `compact`, `cfd33` | 3-point implicit / 3-point explicit compact stencil |
| CFD35 | `cfd35` | 3-point implicit / 5-point explicit compact stencil |
| CFD55 | `cfd55` | 5-point implicit / 5-point explicit compact stencil |

Spectral Difference treats the radial profile as a global polynomial approximation. The resulting matrices are usually dense, but the construction is direct and has a clear error structure for smooth low- to moderate-order cases.

Compact Finite Difference first builds a local stencil relation

$$
A u_\rho = B u,
$$

and then derives $D$ and $G$ from that relation. The local stencil makes the scheme closer to finite-difference intuition and supports one-sided windows near boundaries. The eliminated matrices may still be dense, but their discrete meaning is induced by the local compact format.

## Integration Constant

The integration matrix uses a zero integration-constant constraint:

$$
\int_0^0 f(s)\,ds = 0 .
$$

If the grid does not explicitly contain $0$, the implementation expresses this condition through interpolation. As a result, `G` always represents cumulative integration from the magnetic-axis-side origin to the current radial node, rather than an antiderivative defined only up to an arbitrary constant.

## Boundary

`calculus` only generates reusable linear operators from nodes and a scheme. It does not interpret the physical meaning of the nodes and does not decide how the residual is projected. Higher layers depend on the matrices exposed by `Grid`, so the radial discretization scheme can change without changing the public semantics of the operator, solver, or `Equilibrium`.
