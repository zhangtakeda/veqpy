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

The current CMake targets are `veqlib_main` and the optional nanobind module
`veqlib_ext`. `veqlib_main` is the canonical C++ validation and benchmark
executable; all executable-side optimization comparisons should run through
`main.cpp` subcommands instead of separate test binaries. `veqlib_ext` exposes
the same PF-psin-uniform-Ip benchmark path to Python so Python can measure the
latency of a direct VEQlib call without launching a subprocess. The default
`veqlib_main probe` mode is a dependency smoke test, not the final VEQ kernel:

- instantiates compile-time quadrature nodes/weights for Chebyshev, Legendre,
  Lobatto, and Radau rules on the unit interval;
- instantiates spectral and compact CFD33/CFD35/CFD55 radial calculus matrices;
- checks compact calculus against constant integration and linear
  differentiation identities;
- solves dense systems with Doolittle, Cholesky, Bunch-Kaufman, Householder, and
  Golub-Reinsch policies;
- solves a tridiagonal band system through the Thomas policy;
- solves `x^2 - 4 = 0` with CMINPACK `hybrd1`;
- solves a 2x2 dense linear system with LAPACKE `dgesv`;
- writes a nlohmann/json report;
- when Enzyme is enabled, differentiates `x * x` at `x = 3` and expects `6`.

The generated-topology validation suite is now `veqlib_main --mode
temp-validation`. It checks that generated `config::DefaultTopology` values can
instantiate one concrete `grid::Grid` and `profiles::Profiles` pair without
making those generic types depend on `config.h`.

The Python extension currently exposes a deliberately narrow, single-thread
surface:

```python
import veqlib_ext

veqlib_ext.validate_pf_psin_uniform_ip_json()
veqlib_ext.solve_pf_psin_uniform_ip_json(repeat=10, warmup=1)
veqlib_ext.stage_pf_psin_uniform_ip_json(stage="evaluate", repeat=10, inner=10000)

solver = veqlib_ext.PfPsinUniformIpSolver()
solver.warmup(5)
solver.solve_direct()  # scalars plus read-only NumPy views
```

The free functions return the same JSON payloads as the corresponding
`veqlib_main` modes. `PfPsinUniformIpSolver` keeps the C++ context alive across
calls and is the interface used for Python-perceived latency comparisons.

## Default Topology

CMake generates `config.h` from cache variables and exposes them through
`config::DefaultTopology`. The generated topology is a composition boundary:
entry points may read `DefaultTopology::Nr`, `DefaultTopology::L_max`, and the
profile counts, then pass those values into ordinary templates. Generic headers
such as `grid.h` and `profiles.h` do not include the generated config.

The main topology variables are:

- `VEQ_NR`, `VEQ_NT` for radial and poloidal grid sizes;
- `VEQ_H_PROFILE_COUNT`, `VEQ_V_PROFILE_COUNT`, `VEQ_KAPPA_PROFILE_COUNT`,
  `VEQ_PSIN_PROFILE_COUNT`, and `VEQ_F_PROFILE_COUNT`;
- `VEQ_COS_PROFILE_COUNTS` and `VEQ_SIN_PROFILE_COUNTS` for Fourier-family
  coefficient counts;
- `VEQ_PROFILE_KMAX_LIMIT` for the upper bound used when deriving `K_max`.

Configure-time validation requires `VEQ_NR >= 4`, `VEQ_NT >= 4`, derived
`L_max >= 1`, derived `M_max >= 1`, derived `K_max >= 2`, and
`VEQ_PROFILE_KMAX_LIMIT >= 2`.

## Static Profile Layout

VEQlib maps VEQPy setup-stage semantics to C++ compile-time data: profile family
order, profile ids, active/fixed/absent ownership, packed coefficient ordering,
and workspace extents are compile-time facts for each concrete instantiation.
Runtime code should carry numerical values only.

`profile_layout.h` starts this boundary with `profiles::ProfileShape`, which
mirrors VEQPy's profile order `h, v, k, c0, c, s, psin, F` and degree-first
packed coefficient layout. `profiles::ProfileSlot` distinguishes absent,
fixed, and optimized profiles so a profile can have runtime fields without
contributing coefficients to packed `x`.

## Runtime Profile ABI

`profiles::RuntimeProfiles<Shape, GridType>` is the storage boundary for later
source, geometry, and residual code. It owns fixed-size slabs for stable
profile-id fields and c/s Fourier-family fields. The shape decides profile ids,
family extents, active profile order, and packed coefficient indices at compile
time; runtime refresh only moves numerical values from fixed parameters or
packed `x` into those slots.

Later stages should consume `RuntimeProfiles<Shape, GridType>&` and the
compile-time metadata on `Shape`. They should not recompute profile order,
family lengths, or packed coefficient layout from runtime state.

## Kernel Plan and Workspace Boundary

`operator_pf::PfPsinUniformOperator` separates repeated callback state into a
small read-mostly `KernelPlan` and a mutable `KernelWorkspace`. The plan owns
setup-derived fixed profile rows and the precomputed `fix_rho` axis-count for
the concrete topology. The workspace owns the active/fixed profile slab,
geometry, source, and residual buffers used by each residual callback.

`evaluate()` lazily prepares the static plan on the first call and then refreshes
only active profiles from packed `x` before geometry/source/residual execution.
Runtime parameters are updated through `set_runtime_params()`, which invalidates
the static plan so the next `evaluate()` or explicit `refresh_static_plan()`
rebuilds fixed profile rows and the `fix_rho` axis-count. The method seeds only
fixed profile rows into the workspace; active rows remain callback-owned and are
overwritten by `refresh_active()`.

## Source Matvec Shape

The PF/psin/uniform/Ip Source path applies the grid differentiator and
accumulator to the same `psin_r` vector in both materialization and normalized
source update. Production code uses `SourceMatvecPlan<GridType, 4>` to pack those
two matrices by output blocks of four rows, so a runtime AVX2/FMA kernel can
broadcast each input value once and update four output rows for both matrices.
The same helper has a generic packed fallback for non-AVX builds and constexpr
validation. The row-dot dual matvec remains as a stage-benchmark probe
(`source_DA_psin`) beside the production block-4 probe
(`source_DA_psin_block4`).

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

VEQPy's production solve path is currently modeled by the CMINPACK `hybrd`
configuration used in `veqlib_main --mode pf-validation`. The `veqlib_main
--mode solve` benchmark keeps the same PF-psin-uniform-Ip case inline and can
compare several candidate solve paths:

```bash
./build/enzyme-release/veqlib_main --mode solve \
  --solver residual|enzyme|lm|newton|nk|nr|powell|sundials-nk|sundials-nr
```

Solver meanings are intentionally explicit:

- `residual` is CMINPACK `hybrd`, matching the VEQPy-style residual-only path.
- `enzyme` is CMINPACK `hybrj` with an Enzyme dense Jacobian.
- `lm` is CMINPACK Levenberg-Marquardt.
- `newton` is a hand-written full-step dense Newton method.
- `nr` is a hand-written dense Newton-Raphson method with backtracking.
- `nk` is a hand-written Newton-Krylov method using Enzyme JVPs and local GMRES.
- `powell` is the CMINPACK Powell hybrid path.
- `sundials-nr` and `sundials-nk` use SUNDIALS KINSOL with dense and SPGMR
  linear solvers respectively.

SciPy's `root(method="krylov")` should not be treated as a thin wrapper around
KINSOL or CMINPACK. It is SciPy's own Newton-Krylov facade built around
`KrylovJacobian`: by default the inner linear solve uses
`scipy.sparse.linalg.lgmres`, Jacobian-vector products are finite-difference
approximations of the residual, and the nonlinear step uses Armijo line search.
That makes it a useful behavioral reference, but not a library implementation
that VEQlib can call directly from C++.

The current PF-psin-uniform-Ip benchmark shows that dense Newton variants are
not automatically faster for the small fixed topology. Even when they converge
in fewer nonlinear iterations, repeated dense Jacobian construction dominates
runtime unless the Jacobian is generated much more cheaply than a batch of
residual evaluations. Powell/hybrd therefore remains the baseline until VEQlib
has a route-specific analytic or template-generated Jacobian path.

`veqlib_main --mode stage` is the lower-level timing path for the same inline
PF-psin-uniform-Ip case. It measures repeated hot-path stages without the
CMINPACK solve loop:

```bash
./build/release/veqlib_main --mode stage \
  --stage all \
  --repeat 30 \
  --warmup 5 \
  --inner 10000
```

Available stages are `profiles_fixed`, `profiles_active`, `profiles_all`,
`geometry_phase`, `geometry_phase_sincos`, `geometry_phase_split_sincos`,
`geometry_metric_no_store`, `geometry`, `source_materialize`,
`source_copy_regularize`, `source_D_psin`, `source_A_psin`,
`source_interpolate_pair`, `source_integrand`, `source_A_integrand`,
`source_normalize`, `source_D_normalized`, `source_alpha`, `source_update`,
`residual_update`, `residual_theta_reduce`, `residual_radial_project`,
`residual_pack`, `evaluate`, and `evaluate_ring`. Each reported sample is
nanoseconds per stage call after dividing by `--inner`. The `geometry_*`,
`source_*`, and `residual_*` fine-grained probe stages are benchmark-only
decompositions of the current route-specific hot path; they are meant to locate
the next hotspot bucket and should not be treated as separate production kernels
or as exact hardware-event attribution. `geometry - geometry_metric_no_store` is
only a surface-output proxy, not a hardware store counter. The residual
theta/radial split materializes moment rows so projection can be timed
separately; use it to compare candidate code shapes, not as a claim that the
current `residual_pack` has identical memory traffic. `evaluate_ring` cycles
through a deterministic synthetic solver-state ring controlled by `--ring-size`;
use it to compare warm repeated callbacks against state-varying callback
traffic, not as a real nonlinear-solver trajectory.

`veqlib_main --mode pf-benchmark` keeps the solver-level JSON compatible while
adding diagnostic callback timing under `final.callback_timing_ms`. The current
fields split scaled residual callbacks into `residual_total`,
`residual_kernel`, and `residual_scale`, keep post-solve diagnostic residual
time as `final_residual`, and reserve `jacobian_total`, `jvp_total`, and
`linear_solve` for Jacobian/JVP/linear-solver paths. These timings use
`std::chrono` instrumentation inside callbacks, so use them for attribution and
counter alignment rather than as the cleanest absolute solve-time baseline.

To compare VEQPy's Python solve latency against a direct VEQlib nanobind call,
build the Release module and run the Python comparison script:

```bash
cmake --preset clang-release
cmake --build --preset clang-release --target veqlib_ext
../.venv/bin/python benchmark_pf_psin_uniform_compare.py \
  --cxx-backend nanobind \
  --module-dir build/release \
  --repeat 30 \
  --warmup 5 \
  --no-write
```

This benchmark times `Solver.solve()` and
`PfPsinUniformIpSolver.solve_direct()` from Python with `time.perf_counter_ns()`. The direct method returns scalar solver metadata plus read-only NumPy views for `x`, raw residual, scaled residual, and `alpha`; the report also records the C++ internal solve time so the interface overhead can be estimated as Python outer time minus C++ inner time.
Use `--cxx-backend subprocess --cxx-exe build/release/veqlib_main` only when
you want the older executable-internal timing path.

For topology sweeps, use `stage_topology_matrix.py`. It creates isolated CMake
build directories under `build/topology-matrix/`, configures `DefaultTopology`
for each requested `Nr x Nt x Mmax`, runs `veqlib_main --mode stage`, and emits a
single JSON matrix report:

```bash
./stage_topology_matrix.py \
  --topology 32x16x1 \
  --topology 32x32x1 \
  --stage evaluate \
  --repeat 10 \
  --warmup 4 \
  --inner 5000
```

When no explicit `--topology` is supplied, `--matrix-preset` selects a built-in
set: `default`, `representative`, or `full`. The full preset is the current
`Nr in {16,32,64}` x `Nt in {8,16,24,32,64}` x `Mmax in {1,4,8}` sweep. Pin the
whole script when comparing timing across rows so the benchmark child processes
inherit the same CPU affinity:

```bash
taskset -c 2 ./stage_topology_matrix.py \
  --matrix-preset full \
  --stage geometry \
  --repeat 6 \
  --warmup 3 \
  --inner 4000 \
  --output /tmp/veqlib_full_matrix_geometry_pinned.json
```

## Dependency Versions

The versions below are the currently validated local toolchain versions.

| Component          | Role                                 | Version                                                    |
| ------------------ | ------------------------------------ | ---------------------------------------------------------- |
| C++ standard       | Source language                      | C++20                                                      |
| clang / clang++    | Required C++ compiler                | 18.1.3, Ubuntu package `clang-18 1:18.1.3-1ubuntu1`        |
| CMake              | Build system                         | 3.28.3, package `cmake 3.28.3-1build7`                     |
| lld                | Release linker for ThinLTO           | 18.1.3, package `lld-18 1:18.1.3-1ubuntu1`                 |
| nlohmann/json      | JSON I/O                             | 3.11.3, package `nlohmann-json3-dev 3.11.3-1`              |
| GCEM               | Compile-time math                    | source install `v1.18.0` at `/home/rhzhang/opt/gcem-install` |
| CMINPACK           | MINPACK-style nonlinear solvers      | package `libcminpack-dev 1.3.6-5build1`                    |
| SUNDIALS KINSOL    | Newton / Newton-Krylov experiments   | package `libsundials-dev 6.4.1+dfsg1-3build4`              |
| LAPACKE / LAPACK   | Dense linear algebra interface       | package `liblapacke-dev 3.12.0-3build1.1`                  |
| OpenBLAS           | BLAS backend                         | package `libopenblas-dev 0.3.26+ds-1ubuntu0.1`             |
| nanobind           | Python extension bridge              | Python package `nanobind 2.13.0` in `.venv`                |
| LLVM dev files     | Enzyme build dependency              | package `llvm-18-dev 1:18.1.3-1ubuntu1`                    |
| libclang dev files | ClangEnzyme build dependency         | package `libclang-18-dev 1:18.1.3-1ubuntu1`                |
| Enzyme             | clang plugin for autodiff            | source build `v0.0.268`, git commit `41b6c734`             |
| ClangEnzyme plugin | Direct clang++ plugin used by VEQlib | `/home/rhzhang/opt/Enzyme/enzyme/build-llvm18/Enzyme/ClangEnzyme-18.so` |
| clangd             | Optional editor language server      | 18.1.3, package `clangd-18 1:18.1.3-1ubuntu1`              |
| clang-format       | Optional formatter                   | 18.1.3, package `clang-format-18 1:18.1.3-1ubuntu1`        |

`LLVMEnzyme-18.so` is also built locally, but VEQlib's direct `clang++` workflow
uses `ClangEnzyme-18.so`. `LLVMEnzyme` is for lower-level LLVM IR / `opt`
pipelines and is not the default integration path here.

## Optimization Profile

Release presets enable VEQlib's aggressive kernel profile by default:

- `VEQLIB_ENABLE_NATIVE_OPTIMIZATIONS=ON`
- `VEQLIB_FP_MODE=RELAXED`
- `VEQLIB_ENABLE_THIN_LTO=ON`

`VEQLIB_FP_MODE` separates the numerical-contract baseline from the remaining
native CPU and loop optimizations:

- `STRICT`: `-fno-fast-math -ffp-contract=off`;
- `FMA`: `-fno-fast-math -ffp-contract=fast`;
- `RELAXED`: the historical benchmark profile with fast-math, reciprocal, and
  approximate-function flags.

The common Release target flags are:

```text
-O3
-march=native -mtune=native
-fstrict-aliasing -fomit-frame-pointer
-funroll-loops -fvectorize -fslp-vectorize
-ffunction-sections -fdata-sections
-flto=thin
link: -fuse-ld=lld -Wl,-O3 -Wl,--gc-sections
```

The default `RELAXED` FP mode adds:

```text
-ffast-math -ffp-contract=fast -funsafe-math-optimizations
-fno-math-errno -fno-trapping-math -fno-signed-zeros
-freciprocal-math -ffinite-math-only -fapprox-func
```

`RELAXED` is intended for generated/fixed-size numerical kernels after route
correctness is locked. It relaxes IEEE floating-point edge-case behavior,
including NaN/inf propagation, trapping, signed zero, errno, and operation
reassociation. Use `STRICT` and `FMA` builds to define correctness and error
budgets before comparing approximate or vector math kernels. The current
Geometry RELAXED backend already uses a validated reduced-Taylor approximation
for dynamic `sincos(tb)`; do not silently replace it with scalar libm or a
different vector backend without paired timing and Python/C++ error evidence.

VEQlib route kernels deliberately do not make solve-success decisions.
`math::is_finite()` is a bit-level NaN/inf helper for diagnostics and tests,
but timed/source/residual kernels should not branch on finiteness, magnitude,
or fallback sentinel values. Solver acceptance belongs in the outer solver or
validation layer that interprets the residual norm.

For source-correlated performance diagnostics, configure a separate analysis
build with `VEQLIB_ANALYSIS_BUILD=ON`. This keeps `-O3` but disables ThinLTO for
that build and emits Clang vectorization optimization remarks, stack-usage
files, frame-size warnings, and large-by-value-copy warnings:

```bash
cmake -S . -B build/analysis \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_ENZYME=OFF \
  -DVEQLIB_ENABLE_THIN_LTO=OFF \
  -DVEQLIB_ANALYSIS_BUILD=ON
cmake --build build/analysis --target veqlib_main
```

## Build Presets

Use CMake presets from this directory:

```bash
cd ~/veqpy/veqlib
cmake --preset clang-enzyme-release
cmake --build --preset clang-enzyme-release
./build/enzyme-release/veqlib_main --mode probe
```

Expected output includes:

```json
{
  "cfd": {
    "cfd33_identity_derivative": 0.9999999999999998,
    "cfd35_identity_derivative": 1.0000000000000004,
    "cfd55_identity_derivative": 1.0000000000000004
  },
  "cminpack": {
    "info": 1,
    "x": 2.0,
    "f": 0.0
  },
  "enzyme": {
    "square_derivative_at_3": 6.0
  },
  "gcem": {
    "sqrt_9": 3.0
  },
  "grid": {
    "Nr": 32,
    "weight_sum": 1.0000000000000002
  },
  "lapacke": {
    "info": 0,
    "solution": [2.0, 3.0]
  },
  "linalg": {
    "thomas": [1.0, 1.0, 0.9999999999999999]
  }
}
```

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
  libsundials-dev \
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
