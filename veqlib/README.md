# VEQlib

VEQlib is the experimental C++ kernel layer for VEQPy. The Python package
remains the owner of high-level model semantics, serialization, and test
orchestration. Setup-time numerical constants are being moved into C++20
`constexpr` generation so the runtime kernel can consume fixed-layout arrays
without Python-side numerical setup. This directory is for the gradually
introduced C++20 pieces:

- fixed-size numerical kernels compiled with clang and an aggressive Release
  profile (`-O3`, native CPU tuning, fast-math, loop unrolling,
  vectorization, ThinLTO, and lld);
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

The current CMake target is `veqlib_probe`. It is a dependency smoke test, not
the final VEQ kernel:

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

The generated-topology validation target is `veqlib_temp_validation`. It checks
that generated `config::DefaultTopology` values can instantiate one concrete
`grid::Grid` and `profiles::Profiles` pair without making those generic types
depend on `config.h`.

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
| LAPACKE / LAPACK   | Dense linear algebra interface       | package `liblapacke-dev 3.12.0-3build1.1`                  |
| OpenBLAS           | BLAS backend                         | package `libopenblas-dev 0.3.26+ds-1ubuntu0.1`             |
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
- `VEQLIB_ENABLE_THIN_LTO=ON`

The current target flags are:

```text
-O3
-march=native -mtune=native
-ffast-math -ffp-contract=fast -funsafe-math-optimizations
-fno-math-errno -fno-trapping-math -fno-signed-zeros
-freciprocal-math -ffinite-math-only -fapprox-func
-fstrict-aliasing -fomit-frame-pointer
-funroll-loops -fvectorize -fslp-vectorize
-ffunction-sections -fdata-sections
-flto=thin
link: -fuse-ld=lld -Wl,-O3 -Wl,--gc-sections
```

This profile is intended for generated/fixed-size numerical kernels. It relaxes
IEEE floating-point edge-case behavior, including NaN/inf propagation,
trapping, signed zero, errno, and operation reassociation. Keep Debug presets
conservative, and use Python-side regression tests to lock acceptable numerical
tolerances for each route before relying on these flags for production kernels.

## Build Presets

Use CMake presets from this directory:

```bash
cd ~/veqpy/veqlib
cmake --preset clang-enzyme-release
cmake --build --preset clang-enzyme-release
./build/enzyme-release/veqlib_probe
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
