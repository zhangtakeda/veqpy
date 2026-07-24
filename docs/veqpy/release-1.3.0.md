# VEQPy 1.3.0 — Semantic source inputs and a specialized native solve core

VEQPy 1.3.0 introduces a clearer physical source contract, more flexible
initialization and equilibrium materialization, and a substantially optimized
Cxx backend.

This release contains intentional public API changes. Existing 1.2.x source
construction code must review the migration notes below before upgrading.

## Highlights

### Physical source inputs

- `KernelTopology` now uses
  `constraint="ip" | "beta" | "both" | "none"`.
- `KernelSource` accepts either absolute pressure samples with `p=...`, or
  derivative samples with `pprime=...` and optional LCFS pressure `p0=...`.
- Route drivers are explicit:
  - PF: `ffprime`
  - PP: `psi_r`
  - PI: `itor`
  - PJ1: `jtor`
  - PJ2: `jpara`
  - PQ: `q`
- Beta normalization acts on the complete pressure profile, including non-zero
  edge pressure.
- `Equilibrium` preserves `p0` during resampling, serialization, diagnostics,
  and GEQDSK export.

### Solver control and equilibrium output

- Added `x0=` support to `Kernel.solve()` and `veqpy.solve()`.
- Initial states may be packed arrays, previous `SolveResult` objects, or
  dictionaries of named active-profile coefficients.
- Added `Kernel.build_equilibrium(grid=...)` for direct materialization on a
  requested output grid.
- Added equilibrium geometry properties `h`, `v`, `kappa`, `Rc`, `epsilon`,
  and `ftrap`.
- Model and serialization classes are now available directly from `veqpy`.

### Native backend

- Specialized Powell hybrid and Levenberg–Marquardt paths for generated kernel
  shapes.
- Reduced residual traffic and unnecessary profile refreshes.
- Improved cache locality in fixed-size LU, LDLT, QR, SVD, and Householder
  operations.
- Added adaptive handoffs to audited CMINPACK and LAPACKE fallbacks for large
  systems.
- Bundled MINPACK-derived primitives, license files, and third-party notices.
- Removed the native `nlohmann-json` dependency.
- Fixed native builds on macOS where the former project `math.h` could shadow
  the system header.

Across the published GEQDSK benchmark matrix, Cxx solve times improve by
approximately 8–38%, averaging about 26%. Representative High configurations
improve by roughly 27–32%, with Cxx-to-Numba speedups reaching 11.1x.

### GEQDSK example

- Added a compact Solovev Numba example that reads a GEQDSK file, fits its
  LCFS, constructs PF sources, solves the equilibrium, writes a new GEQDSK,
  and plots magnetic-surface and source-profile comparisons.

### Reliability

- Added reference coverage for native nonlinear solvers and fixed-size linear
  algebra.
- Added Cxx/Numba source-route parity tests.
- Added absolute-pressure, edge-pressure, beta-normalization, fallback,
  packaging, and neoclassical-geometry tests.
- Preserved non-finite guards under fast-math builds.
- Fixed benchmark artifact-input and programmatic-caller behavior.

## Breaking changes

- Replace `ip_constraint` and `beta_constraint` with `constraint`.
- Replace `heat_profile` and `current_profile` with one pressure input and one
  route-specific driver.
- `KernelSource` is now keyword-only.
- Supplying missing, conflicting, or route-incompatible source profiles now
  fails during input validation.

Although this release remains in the 1.x series, these are intentional public
API migrations rather than backward-compatible additions.

## Migration

VEQPy 1.2.x code:

```python
topology = KernelTopology(
    ...,
    ip_constraint=True,
    beta_constraint=False,
)

source = KernelSource(
    heat_profile=pprime,
    current_profile=ffprime,
    Ip=3.0e6,
)
```

VEQPy 1.3.0 equivalent:

```python
topology = KernelTopology(
    ...,
    constraint="ip",
)

source = KernelSource(
    pprime=pprime,
    ffprime=ffprime,
    Ip=3.0e6,
)
```

For derivative pressure input, `p0` defaults to zero and can be specified
explicitly when the LCFS pressure is non-zero:

```python
source = KernelSource(
    pprime=pprime,
    p0=edge_pressure,
    ffprime=ffprime,
    Ip=plasma_current,
)
```

Absolute pressure can be supplied directly:

```python
source = KernelSource(
    p=pressure,
    q=safety_factor,
    beta=target_beta,
)
```

## Compatibility

- Python 3.12 or newer is required.
- The Numba backend does not require native compilation.
- The optional Cxx backend requires a C++20 toolchain, CMake, nanobind, GCEM,
  CMINPACK, LAPACKE/LAPACK, and OpenBLAS.

**Full changelog:** [v1.2.2...v1.3.0][full-changelog]

[full-changelog]: https://github.com/zhangtakeda/veqpy/compare/v1.2.2...v1.3.0
