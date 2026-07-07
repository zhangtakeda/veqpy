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
equilibrium snapshot assembly behind the common `Kernel` surface.

The Cxx backend is the native C++/nanobind runtime used for topology-specific
shared-library kernels and performance measurements. It uses the same public
Kernel dataclasses and method surface as the Numba backend. Native artifacts are
cached under `.veqpy-kernel-cache/` in the current working directory by default,
or under `VEQPY_KERNEL_CACHE` when that environment variable is set.

Benchmark entry points use backend names directly:

- `benchmarks/numba_routes.py`: Numba route matrix.
- `benchmarks/cxx_routes.py`: Cxx route matrix compared with Numba.
- `benchmarks/cxx_geqdsk_pareto.py`: Cxx GEQDSK matrix compared with Numba.
- `benchmarks/cxx_continuation.py`: Cxx continuation-policy nfev benchmark.

The native backend needs a C++20 toolchain plus CMake, nanobind, GCEM,
nlohmann-json, CMINPACK, LAPACKE/LAPACK, and OpenBLAS. Normal Numba usage does
not require compiling native artifacts.
