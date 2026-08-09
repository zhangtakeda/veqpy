# VEQPy 1.3.1 — Correct radial endpoints and faster equilibrium state

VEQPy 1.3.1 strengthens the numerical meaning of radial endpoints, improves
magnetic-axis source regularization, and reduces equilibrium-state overhead.
It also expands the equilibrium model with IMAS-compatible flux-surface
geometry coefficients.

This release does not intentionally remove or rename any public API introduced
in VEQPy 1.3.0.

## Highlights

### Physical radial endpoints

- Open Gauss nodes are no longer treated as the magnetic axis or LCFS.
- `Grid.axis_eval()`, `Grid.edge_eval()`, and `Grid.full_integral()` now provide
  explicit physical-endpoint evaluation and full-domain integration.
- Plasma-current constraints, normalized-flux construction, source
  reconstruction, and the PQ route now consistently use physical endpoints.
- GEQDSK and resampling paths explicitly include the axis and LCFS when the
  destination representation requires closed endpoints.
- Numba and Cxx source semantics remain aligned after the endpoint correction.

### Magnetic-axis source regularization

- Source reconstruction near the magnetic axis now uses a fixed-cost,
  four-anchor least-squares regularization.
- The regularization preserves the natural axis limit without relying on the
  first open-grid sample as if it were the axis.
- Dedicated tests cover representative PF, PP, PI, PJ1, PJ2, and PQ source
  routes.

### Equilibrium and IMAS geometry

- `Equilibrium` now exposes the IMAS flux-surface geometry coefficients
  `gm1`–`gm9`.
- Signed `F` is preserved, and PJ2 parallel-current semantics are clarified.
- Toroidal-current reconstruction uses the conservative axis limit of the
  enclosed-current and surface terms.
- IMAS `rho_tor_norm` construction uses the complete toroidal-flux interval.
- Equilibrium plotting has been separated from the physical model, while the
  numerical kernels remain colocated with the model implementation.

### Runtime performance

- Reactive invalidation now propagates through precomputed dependency graphs,
  avoiding repeated graph discovery during root-state updates.
- Equilibrium diagnostics are materialized by Numba kernels with lazy reactive
  evaluation preserved.
- Serialization and import behavior remain compatible with the optimized
  reactive implementation.

## Compatibility and migration

- Python 3.12 or newer is required.
- No intentional public API migration is required from VEQPy 1.3.0.
- Numerical baselines on open Gauss grids may change because the first and last
  quadrature nodes are no longer interpreted as physical endpoints.
- Downstream code that needs an axis value, LCFS value, or full-domain integral
  should use `axis_eval()`, `edge_eval()`, or `full_integral()` instead of direct
  `[0]` or `[-1]` indexing.

## Validation

- 275 tests passed and 16 optional tests were skipped in the local release
  environment.
- Ruff static checks passed.
- The source distribution and wheel passed `twine check`.
- CI validates Python 3.12 and 3.13, both solver backends, package construction,
  and isolated wheel installation.

**Full changelog:** [v1.3.0...v1.3.1][full-changelog]

[full-changelog]: https://github.com/zhangtakeda/veqpy/compare/v1.3.0...v1.3.1
