# VEQPy 1.3.2 — Robust current closures and asymmetric GEQDSK export

VEQPy 1.3.2 strengthens source-route closure, makes the published package
usable without a native toolchain by default, preserves magnetic-axis flux
limits during resampling, and fixes GEQDSK export for vertically asymmetric
equilibria.

This release does not intentionally remove or rename any public API introduced
in VEQPy 1.3.x.

## Highlights

### Current-source semantics and closure

- The new PJ3 route accepts the IMAS `jtotal = <J.B>/B0` convention and converts
  it against the current geometry, `F`, and `gm1` inside each source evaluation.
- Shared source lowering preserves finite caller-provided `pprime` and route
  driver samples; route-local kernels regularize only closure-derived quantities.
- Geometric-rho PJ2/PJ3 can reconstruct enclosed current and `F²` without active
  `F` coefficients, using a cold-started, history-independent fixed-point closure.
- Strict PJ2/PJ3 closure now stops on a normalized defect of `1e-6` and fails
  after at most ten Picard iterations instead of relying on route-specific fixed
  sweep counts.
- The PQ route closes its quotient at the numerator, and rho-coordinate route
  closures retain their intended endpoint semantics.

### Default backend and package behavior

- `KernelRecipe()` and `recipe=None` now select the Numba backend by default.
- A normal PyPI installation therefore has a usable solver backend without
  requiring CMake, a C++ compiler, or separately provisioned native libraries.
- The Cxx backend remains available by explicit selection. Strict rho PJ2/PJ3
  with `F_count=0` remains intentionally unsupported there; positive `F_count`
  retains the optimized-F comparison path.

### Magnetic-axis and GEQDSK correctness

- Endpoint resampling reconstructs the unresolved magnetic-axis cell from
  `psin_r/rho` as a smooth function of `rho²`, preserving removable flux and
  safety-factor limits without filtering resolved source nodes.
- GEQDSK export now distinguishes the plasma-boundary vertical reference from
  the rectangular psi-grid `zmid` field.
- Vertically asymmetric equilibria retain their requested Z bounds and internal
  flux-surface registration across GEQDSK write/read round trips.
- The GEQDSK demo uses active vertical-shift and higher-order Fourier profiles
  suitable for asymmetric input boundaries.

## Compatibility and migration

- Python 3.12 or newer is required.
- No public symbols are intentionally removed or renamed.
- Code that relied on the implicit Cxx default should now request
  `KernelRecipe(backend="cxx")` explicitly.
- Explicit Numba and Cxx selections retain their existing meaning.
- GEQDSK files exported from asymmetric equilibria may differ in `zmid` and psi
  placement because the rectangular grid is now serialized consistently.

## Validation

- The complete local test suite passed with 352 tests and 16 optional skips.
- Ruff static checks passed.
- The source distribution and wheel passed `twine check`.
- The wheel passed isolated-install metadata, import, and default-backend smoke
  checks.
- CI validates Python 3.12 and 3.13, both solver backends, package construction,
  and isolated wheel installation.

**Full changelog:** [v1.3.1...v1.3.2][full-changelog]

[full-changelog]: https://github.com/zhangtakeda/veqpy/compare/v1.3.1...v1.3.2
