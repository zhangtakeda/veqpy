# Quadrature

The `quadrature` module builds radial quadrature nodes and weights on $[0,1]$:

$$
\int_0^1 f(x)\,dx \approx \sum_i w_i f(x_i).
$$

These nodes enter `Grid` and are used for radial integration, flux-surface averages, and residual projection. This page describes the scheme semantics; implementation details live in `veqpy/math/quadrature.py`.

## Scheme

| Scheme | Node feature | Endpoint behavior | Typical meaning |
| ------ | ------------ | ----------------- | --------------- |
| Legendre | Gauss-Legendre interior nodes | Excludes endpoints | High-order integration accuracy for smooth integrands |
| Radau | Gauss-Radau nodes | Includes $x=1$ | Keeps the boundary endpoint without keeping the magnetic-axis-side endpoint |
| Lobatto | Gauss-Lobatto nodes | Includes $x=0,1$ | Explicitly represents both magnetic-axis-side and boundary-side endpoints |
| Chebyshev | Chebyshev-distributed interior nodes | Excludes endpoints | Natural fit with Chebyshev-like profile representation |
| Uniform | Uniform trapezoidal nodes | Includes $x=0,1$ | Useful for debugging, comparison, and external uniform data |

All weights are normalized for the unit interval, so constants satisfy

$$
\sum_i w_i = 1.
$$

If nodes are first constructed on $[-1,1]$, the implementation maps them to $[0,1]$ before exposing them to higher layers.

## Choosing a Scheme

The quadrature scheme is not only a performance parameter; it affects radial averaging in residual projection and snapshot diagnostics. Legendre/Radau/Lobatto are Gauss-type rules suited to integration accuracy; Chebyshev matches spectral-profile usage more naturally; Uniform is most useful for testing and comparison with external uniformly sampled data.

## Boundary

`quadrature` only generates nodes and weights. It does not define profile interpolation, build differentiation matrices, or interpret source routes. These arrays are aggregated by `Grid` before entering model and operator logic. Keeping this boundary lets integration rules evolve independently as enumerable schemes.
