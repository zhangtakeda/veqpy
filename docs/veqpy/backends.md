# Backends

`veqpy.Kernel` is the public runtime handle. Backend selection is a recipe
choice, not a user-visible class split:

```python
from veqpy import KernelRecipe

KernelRecipe(backend="numba")
KernelRecipe(backend="cxx")
```

The Numba backend is the direct Python runtime used for development, route
coverage, and pure-Python deployment. It owns packed layout metadata, source
runtime arrays, residual workspaces, finite-difference JVP/Jacobian calls, and
equilibrium snapshot assembly behind the common `Kernel` surface. It supports
`Kernel.variant(...)` for count-only active topology switches. Contained
variants reuse capacity-sized workspaces while refreshing active layout
metadata.

The Cxx backend is the native C++/nanobind runtime used for topology-specific
shared-library kernels and performance measurements. It uses the same public
Kernel dataclasses and method surface as the Numba backend. Native artifacts are
cached under `.veqpy-kernel-cache/` in the current working directory by default,
or under `VEQPY_KERNEL_CACHE` when that environment variable is set. The Cxx
backend does not currently support `Kernel.variant(...)` or capacity-style
`L_max` values larger than the active count requirement.

Source lowering is shared before backend dispatch. Both runtimes receive one
canonical pressure derivative plus its LCFS integration constant, one
route-specific driver, and the selected global constraints. The native Cxx
source runtime retains the common pressure multiplier, effective scaled LCFS
pressure, and final alpha factors from its latest residual evaluation so
backend-parity tests can inspect the physical state directly. This diagnostic
surface remains a low-level Cxx implementation detail rather than a package-root
API.

Benchmark entry points use backend names directly:

- `benchmarks/numba_routes.py`: Numba geometric-rho route matrix.
- `benchmarks/numba_variant_sweep.py`: Numba `Kernel.variant()` switch cost vs
  fresh Kernel construction benchmark.
- `benchmarks/cxx_routes.py`: Cxx geometric-rho route matrix compared with
  Numba. psin routes retain focused physics/API coverage but are excluded from
  route performance and pass-rate summaries.
- `benchmarks/cxx_geqdsk.py`: Cxx GEQDSK matrix compared with Numba.
- `benchmarks/cxx_continuation.py`: Cxx continuation-policy nfev benchmark.
- `benchmarks/cxx_boundary_fitters.py`: raw-boundary `qr`/`gnqr`/`least-square`
  fitter timing and diff matrix across Numba and Cxx.

The native backend needs a C++20 toolchain plus CMake, nanobind, GCEM,
CMINPACK, LAPACKE/LAPACK, and OpenBLAS. Its Powell hybrid and
Levenberg-Marquardt solvers compile the bundled MINPACK-derived primitives into
each native artifact; the fixed 8096-variable compile-time threshold retains
the standard CMINPACK drivers as a fallback. Normal Numba usage does not
require compiling native artifacts.
