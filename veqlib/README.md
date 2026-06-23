# VEQlib

VEQlib is the experimental C++ kernel layer for VEQPy. The Python package
remains the owner of high-level model semantics, serialization, and test
orchestration. Setup-time numerical constants are being moved into C++20
`constexpr` generation so the runtime kernel can consume fixed-layout arrays
without Python-side numerical setup. This directory is for the gradually
introduced C++20 pieces:

- fixed-size numerical kernels compiled with clang and selectable floating-point
  modes over an aggressive Release profile (`-O3`, native CPU tuning, loop
  unrolling, vectorization, ThinLTO, and lld);
- GCEM-backed compile-time math for generated constants and fixed-size kernels;
- compile-time quadrature nodes, weights, spectral calculus, compact finite
  difference calculus, and fixed-size tensor/linalg helpers;
- route-specific residual implementations for the variational formulation;
- CMINPACK-backed nonlinear solves compatible with VEQPy's `hybr` / `lm`
  migration path;
- optional Enzyme-based automatic differentiation for C++ kernels;
- JSON-based smoke tests and small structured data exchange with Python.

The current boundary is intentionally narrow: VEQPy carries model meaning and
selects configurations, while VEQlib generates the fixed arrays needed by the
compiled kernels. Hot-path workspaces should remain fixed-layout runtime arrays;
setup data should be `constexpr` wherever the compile-time cost is acceptable.

## Current Target

The current production CMake target is the optional nanobind module
`veqlib_ext`. The previous executable-side validation, stage benchmark, and
parameter-scan CLI were removed after the PF-psin-uniform-Ip kernel was matched
against the VEQPy/Numba backend. Production comparisons now go through the
Python-facing `KernelSolver`, which exercises the same shared-library loading
path used by VEQPy.

The Python extension exposes a deliberately narrow, single-thread production
kernel surface:

```python
import veqlib_ext

solver = veqlib_ext.KernelSolver()
solver.metadata()
solver.set_case_json(payload_json)
solver.solve_direct()  # scalars plus read-only NumPy views
```

`KernelSolver` keeps the C++ context and workspace alive across calls. Solver
objects are mutable workspace owners and should not be shared across Python
threads; use one solver instance per thread.

## Default Topology

CMake generates `config.h` from cache variables and exposes them through
`config::Topology`. The generated topology is a composition boundary:
entry points may read `Topology::Nr`, `Topology::L_max`, and the
profile counts, then pass those values into ordinary templates. Generic headers
such as `grid.h` and `profiles.h` do not include the generated config.

The main topology variables are:

- `VEQ_NR`, `VEQ_NT` for radial and poloidal grid sizes;
- `VEQ_H_PROFILE_COUNT`, `VEQ_V_PROFILE_COUNT`, `VEQ_KAPPA_PROFILE_COUNT`,
  `VEQ_PSIN_PROFILE_COUNT`, and `VEQ_F_PROFILE_COUNT`;
- `VEQ_COS_PROFILE_COUNTS` and `VEQ_SIN_PROFILE_COUNTS` for Fourier-family
  coefficient counts;
- `VEQ_BOUNDARY_M_MAX` for the boundary/geometric Fourier ceiling. The default
  `AUTO` keeps the historical behavior by deriving it from active c/s counts;
  an explicit value lets boundary `ca`/`sa` orders exceed the optimized profile
  topology;
- `VEQ_PROFILE_KMAX_LIMIT` for the upper bound used when deriving `K_max`.

Configure-time validation requires `VEQ_NR >= 4`, `VEQ_NT >= 4`, derived
`L_max >= 1`, derived active `M >= 1`, boundary `M_max >= active M`, derived
`K_max >= 2`, and `VEQ_PROFILE_KMAX_LIMIT >= 2`.

## Static Profile Layout

VEQlib maps VEQPy setup-stage semantics to C++ compile-time data: profile family
order, profile ids, active/fixed/absent ownership, packed coefficient ordering,
and workspace extents are compile-time facts for each concrete instantiation.
Runtime code should carry numerical values only.

`profiles.h` starts this boundary with `profiles::ProfileShape`, which
mirrors VEQPy's profile order `h, v, k, c0, c, s, psin, F` and degree-first
packed coefficient layout. `profiles::ProfileSlot` distinguishes absent,
fixed, and optimized profiles so a profile can have runtime fields without
contributing coefficients to packed `x`.

## Runtime Profile ABI

`profiles::RuntimeProfiles<Shape, GridType>` is the storage boundary for later
source, geometry, and residual code. It owns fixed-size slabs for stable
profile-id fields, active c/s Fourier-family deltas, boundary-family bases, and
the cached boundary phase. The shape decides profile ids, family extents, active
profile order, and packed coefficient indices at compile time; runtime refresh
only moves numerical values from fixed parameters or packed `x` into those
slots. Boundary-only Fourier orders are computed once from setup/profile
parameters and cached outside the residual callback hot path, while optimized
orders are recomputed from packed `x` on each callback. Boundary amplitudes with
`abs(offset * scale) <= 1.0e-10` are pruned when building the cached boundary
base. The cached boundary setup records only the surviving c/s orders and clears
only the previously surviving rows on refresh, so zero/pruned high-order
capacity is not read by phase synthesis and does not force a full family-slab
clear.

Later stages should consume `RuntimeProfiles<Shape, GridType>&` and the
compile-time metadata on `Shape`. They should not recompute profile order,
family lengths, or packed coefficient layout from runtime state.

## Kernel Plan and Workspace Boundary

`operators.h` is the route-operator expansion surface: it currently carries the
single `operators::PfPsinUniformIpOperator` instantiation shell and is intended to
collect the benchmark-route operator family as those routes are expanded.
`operators::PfPsinUniformIpOperator` separates repeated callback state into a
small read-mostly `KernelPlan` and a mutable `KernelWorkspace`. The plan owns
setup-derived fixed profile rows and the precomputed `fix_rho` axis-count for
the concrete topology. The workspace owns the active/fixed profile slab,
geometry, source, and residual buffers used by each residual callback.

The operator is constructed from a setup object containing the fixed profile
parameters, `fix_rho`, and uniform source tables. Construction eagerly prepares
the static plan: fixed profile rows and the `fix_rho` axis-count are available
before the first callback, and the source tables are already seeded into the
workspace. Per-solve values (`a`, `R0`, `Z0`, `B0`, and `Ip`) are updated through
`set_solve_params()` without rebuilding the static plan, so parameter scans over
`Ip` avoid reloading fixed profiles or source inputs. A caller that changes
profile/source setup must reconstruct the operator or call `reprepare()` with a
new setup. Active rows remain callback-owned and are overwritten by
`refresh_active()`.

## Source Matvec Shape

The PF/psin/uniform/Ip Source path routes production dense radial matvecs
through `tensor_layout::RadialGridMatvecPlan<GridType>`. The plan keeps typed
`tensor_layout::MultiMatvecPlan<K, Rows, Cols>` storage for the K=1 accumulator
path and K=2 D/A path, while `tensor_kernels::matvec_into()` and
`tensor_kernels::multi_matvec_into()` select constexpr-safe scalar fallback or
the native SIMD backend. Row-dot probes remain available as stage-benchmark
baselines: `source_DA_psin` beside `source_DA_psin_packed`, and
`source_A_integrand_rowdot` beside the production `source_A_integrand`.

## Hot-path Surface Layout

Materialized geometry and residual surface slabs use physical
`[rho][field][theta]` storage while preserving logical accessors of the form
`surface_field(field, rho, theta)`. This keeps the theta sweep contiguous and
matches the producer pattern that writes multiple fields at each radial/theta
point. Kernel code should go through the logical accessor unless it is making a
measured layout-local optimization.

Geometry computes dynamic phase values and their radial/theta derivatives in a
separate theta pass before evaluating dynamic `sincos(tb)` and metric
quantities. In the default RELAXED kernel, that dynamic trig backend reduces
`tb` to the nearest `pi/2` quadrant and uses the validated `sin x^11` /
`cos x^10` Taylor truncation on `|r|<=pi/4`, followed by branchless
quadrant reconstruction. This is validated
against the Python reference and is part of the performance kernel contract; use
`STRICT`/`FMA` builds when establishing error budgets for future math backends.

## Grid and Calculus

The public grid families currently exposed from `grid.h` are:

```cpp
grid::Chebyshev
grid::Legendre
grid::Lobatto
grid::Radau
```

Each family provides compile-time unit-interval quadrature arrays:

```cpp
constexpr auto nodes = grid::Legendre::nodes<32>;
constexpr auto weights = grid::Legendre::weights<32>;
```

The calculus policies are:

```cpp
grid::Spectral
grid::CFD33
grid::CFD35
grid::CFD55
```

They expose dense fixed-size matrices so runtime application remains a matrix
vector operation:

```cpp
constexpr auto d = grid::CFD35::differentiator<8, grid::Lobatto>;
constexpr auto a = grid::CFD35::accumulator<8, grid::Lobatto>;
```

CFD matrices follow VEQPy's compact finite-difference construction: interior
rows are generated by local polynomial moment matching; boundary rows use
explicit finite-difference weights; the implicit compact matrix is eliminated
once during setup. The implicit CFD matrix is stored as
`Matrix<double, Bandwidth, N>` during generation, but the exported accumulator
and differentiator remain dense `Matrix<double, N, N>` values.

## Linear Algebra

Dense linear policies share the normal fixed-matrix interface:

```cpp
auto x = linalg::solve<linalg::Doolittle>(A, b);
auto context = linalg::factorize<linalg::Cholesky>(spd);
context.substitute_inplace<1>(rhs.data());
```

The Thomas policy uses the same `factorize` / `solve` / `solve_into` entry
points, but its matrix argument is band storage:

```cpp
Matrix<double, 3, N> tridiagonal_band;
Matrix<double, N, 1> rhs;
auto x = linalg::solve<linalg::Thomas>(tridiagonal_band, rhs);
```

For `Matrix<double, Bandwidth, N>`, the center row stores the main diagonal and
row `Bandwidth / 2 + i - j` stores entry `(i, j)`. Thomas contexts own only the
band factorization and `substitute_inplace`; solve orchestration stays in the
free `linalg` functions.

## Nonlinear Solver Notes

VEQPy's production-facing VEQlib path is the nanobind `KernelSolver`. Runtime
case setup is supplied by JSON payloads from Python; topology, layout, and route
kernel structure remain compile-time C++ metadata. Solver choices exposed to
Python are intentionally limited to the production-supported runtime method
codes, currently Powell hybrid and Levenberg-Marquardt.

To compare VEQPy's Python solve latency against a direct VEQlib nanobind call,
build the Release module and run the Python comparison script:

```bash
cmake --preset clang-release
cmake --build --preset clang-release --target veqlib_ext
../.venv/bin/python benchmark_pf_psin_uniform_compare.py \
  --module-dir build/release \
  --repeat 30 \
  --warmup 5 \
  --no-write
```

This benchmark times `Solver.solve()` and `KernelSolver.solve_direct()` from
Python with `time.perf_counter_ns()`. The direct method returns scalar solver
metadata plus read-only NumPy views for `x`, raw residual, scaled residual, and
`alpha`; the report also records the C++ internal solve time so the interface
overhead can be estimated as Python outer time minus C++ inner time.

For optimization gates, run the four-case artifact benchmark from the repository
root. It covers the 18-parameter PF case plus the solovev, chease, and efit
GEQDSK cases through the Python `Topology` / nanobind-artifact path:

```bash
taskset -c 0 .venv/bin/python veqlib/benchmark_4case_compare.py \
  --repeat 11 \
  --warmup 3 \
  --output /tmp/veqlib_4case_compare.json
```

Executable-side C++ validation/stage diagnostics are intentionally retired. New
performance evidence should use the production nanobind/shared-library path or a
Python-level lifecycle benchmark.

## Build Presets

Use CMake presets from this directory:

```bash
cd ~/veqpy/veqlib
cmake --preset clang-release
cmake --build --preset clang-release --target veqlib_ext
ctest --test-dir build/release -R veqlib_python_binding --output-on-failure
```

The Python binding smoke test imports `veqlib_ext`, constructs `KernelSolver`, and checks the read-only NumPy result views.

Available presets:

| Preset                 | Purpose                             |
| ---------------------- | ----------------------------------- |
| `clang-debug`          | Debug build without Enzyme          |
| `clang-release`        | Aggressive Release without Enzyme   |
| `clang-release-strict` | Release with strict FP contract     |
| `clang-release-fma`    | Release with strict math plus FMA   |
| `clang-enzyme-release` | Aggressive Release with ClangEnzyme |

The Enzyme preset currently records the local plugin path. On another machine,
either edit `VEQLIB_ENZYME_PLUGIN` in `CMakePresets.json` or pass it at configure
time:

```bash
cmake --preset clang-enzyme-release \
  -DVEQLIB_ENZYME_PLUGIN=/path/to/ClangEnzyme-18.so
```

`ENZYME_PLUGIN=/path/to/ClangEnzyme-18.so` is also accepted as an environment
variable.

GCEM is discovered through `VEQLIB_GCEM_ROOT`, which defaults to
`$HOME/opt/gcem-install`. On another machine, pass a different prefix at
configure time:

```bash
cmake --preset clang-enzyme-release \
  -DVEQLIB_GCEM_ROOT=/path/to/gcem-install
```

## Editor Setup

VS Code is configured to treat `veqlib` as the CMake source directory and to use
the `clang-enzyme-release` preset. The preset exports
`compile_commands.json`, which clangd uses for correct C++20 parsing and system
include discovery.

Useful local checks:

```bash
clangd --compile-commands-dir=~/veqpy/veqlib/build/enzyme-release \
  --check=~/veqpy/veqlib/main.cpp

clang-format --version
```

`clang-format` is not a build dependency. clangd can expose editor formatting,
but that formatting path follows clang-format style rules, so keeping the tool
installed makes formatting behavior explicit and reproducible. The local VS Code
settings disable clangd inlay hints, which removes gray inline previews such as
aggregate initializer element labels.

## Dependency Installation Notes

On Ubuntu 24.04, the validated system packages were installed with:

```bash
sudo apt install -y \
  build-essential \
  cmake \
  clang-18 \
  clangd-18 \
  clang-format-18 \
  lld-18 \
  libclang-18-dev \
  llvm-18-dev \
  nlohmann-json3-dev \
  libcminpack-dev \
  liblapacke-dev \
  libopenblas-dev
```

The Python extension target also needs nanobind in the Python environment used
by CMake. In this checkout, CMake prefers `../.venv/bin/python` when it exists:

```bash
.venv/bin/python -m pip install nanobind
```

The Enzyme plugin was built from source against LLVM/Clang 18. The critical
configure inputs are:

```bash
cmake -S . -B build-llvm18 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang \
  -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-18/build/utils/lit/lit.py
```

If Enzyme is rebuilt elsewhere, prefer `ClangEnzyme-<clang-major>.so` for this
project.

GCEM was installed as a header-only CMake package with:

```bash
mkdir -p ~/opt
git clone --branch v1.18.0 --depth 1 https://github.com/kthohr/gcem.git ~/opt/gcem
cmake -S ~/opt/gcem -B ~/opt/gcem/build \
  -DCMAKE_INSTALL_PREFIX=~/opt/gcem-install
cmake --build ~/opt/gcem/build
cmake --install ~/opt/gcem/build
```
