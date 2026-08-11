<p>
  <img
    align="left"
    src="docs/assets/veqpy_banner.svg"
    alt="VEQPy logo"
  />
</p>

<br clear="left"><br>

[![arXiv][arxiv-badge]][veq-arxiv]
[![Python][python-badge]][python]
[![Package][package-badge]][pypi]
[![CI][ci-badge]][ci]
[![Tests][tests-badge]][tests]
[![License][license-badge]][license]

---

# VEQPy

**VEQPy** is the Python implementation of **VEQ** (Veloce EQuilibrium), a fast
parametric Grad--Shafranov solver for fixed-boundary, axisymmetric tokamak equilibria.
It is designed for repeated modeling calls that require low-latency access to
continuous fixed-boundary geometry. Unlike grid-map equilibrium solvers whose primary
unknowns are two-dimensional flux values, VEQPy solves for MXH-type flux-surface
harmonics together with shifted-Chebyshev radial profile/source coefficients. The
primary nonlinear system is the finite-dimensional projection of the Grad--Shafranov
residual onto this representation; its solution is a continuous equilibrium snapshot
that can be resampled, serialized, and diagnosed. Sampled local strong-form residuals
and optional collocation polish are used as diagnostics or post-processing on the same
representation; they do not redefine the primary solve.

VEQPy is suited to parameter scans, source preprocessing, control-oriented iteration,
transport coupling, and surrogate-model workflows. It retains richer two-dimensional
shaping and residual diagnostics than low-order shape models, while remaining lighter
and easier to reuse than full solver-native equilibrium or reconstruction pipelines.

## Feature Overview

- **Compact equilibrium representation**: fixed-boundary flux surfaces, shaping profiles,
  and source-related radial profiles are represented by coefficients, with a continuous
  `Equilibrium` snapshot produced after the solve.
- **Unified source route layer**: PF, PP, PI, PJ1, PJ2, PJ3, and PQ routes map pressure-gradient,
  toroidal-field, flux-gradient, current-related, or safety-factor information to one
  finite-dimensional residual assembly.
- **Explicit Kernel runtime boundary**:
  `KernelTopology + KernelBoundary + KernelSource -> Kernel -> SolveResult + Equilibrium`
  separates packed topology, runtime case inputs, nonlinear solve orchestration, and
  post-solve snapshots.
- **GEQDSK workflow support**: GEQDSK I/O, fixed-boundary fitting from GEQDSK boundaries,
  snapshot export, flux-surface comparison, and common diagnostics.
- **Formula-oriented model objects**: `Profile` stores serializable shape-profile
  roots and, when bound to a `Grid`, lazily materializes value and radial
  derivatives. `Grid` and `Equilibrium` use reactive derived properties to
  reconstruct geometry and physics diagnostics by formula.
- **Kernel API**: `veqpy.Kernel` is the backend-neutral runtime handle.
  It uses
  `KernelTopology + KernelRecipe + KernelBoundary + KernelSource + KernelConfig`
  types from `veqpy`, keeps raw runtime source profiles in `KernelSource`, and
  selects the `cxx` or `numba` backend through `KernelRecipe.backend`.

## Installation

VEQPy requires Python 3.12 or newer. For normal use, install the published package from
PyPI into a project-local virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install veqpy
```

The PyPI installation is ready to use with the default Numba backend. It does not
provision the native toolchain and libraries required by `backend="cxx"`, so
`pip install veqpy` alone does not provide a usable Cxx backend. Missing Cxx
components do not affect imports or solves unless the Cxx backend is selected
explicitly.

For development, install VEQPy from a source checkout in editable mode. The `dev` extra
installs the runtime dependencies together with `pytest`, `ruff`, `build`, `twine`,
`nanobind`, and other development helpers into the same environment.

```bash
git clone https://github.com/zhangtakeda/veqpy.git
cd veqpy
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

For a runtime-only install from a local source checkout, omit the `dev` extra:

```bash
.venv/bin/python -m pip install .
```

VEQPy is a single public package. `veqpy.model` owns model-layer objects,
`veqpy.kernels` owns the public Kernel wrapper, typed Kernel contract, and
private Numba/Cxx backends, and `veqpy.api` provides thin function-style
entrypoints. The native C++ backend is optional for normal
Python/Numba use and requires a local C++20 toolchain and native libraries such
as CMake 3.24+, `clang++`, nanobind, GCEM, CMINPACK, LAPACKE/LAPACK, and
OpenBLAS.

All commands below use `.venv` explicitly; activating the environment is optional.

## Demo

`demo.py` is the external-user starting point. It builds the smallest smooth
fixed-boundary PF(`psin`) Kernel case directly with the public `veqpy` API:
`KernelTopology`, `KernelRecipe`, `KernelConfig`, `KernelBoundary`, and
`KernelSource`.

```bash
.venv/bin/python demo.py
```

It writes `demo_init.png`, `demo_result.png`, and `demo_equilibrium.json` in the
current directory. This branch keeps the public package, demo, benchmark helpers,
and current Kernel architecture aligned.

`demo_geqdsk.py` demonstrates the GEQDSK workflow with the bundled Solovev case.
It reads the reference LCFS and source profiles, solves them with the Numba
backend, writes `data/solovev-veqpy.geqdsk`, and creates a magnetic-surface and
source-profile comparison figure. Both outputs are local generated artifacts and
are intentionally excluded from Git and release archives.

```bash
.venv/bin/python demo_geqdsk.py
```

## Development Checks

Core local checks mirror the push/PR CI workflow.

```bash
.venv/bin/python -m compileall -q veqpy tests benchmarks demo.py demo_geqdsk.py
.venv/bin/ruff check veqpy tests benchmarks demo.py demo_geqdsk.py
.venv/bin/python -m pytest
```

## Optional C++ Kernels

The Cxx backend is the native C++/nanobind kernel layer used for
topology-specific shared-library kernels and Cxx-vs-Numba benchmarks.

Representative Cxx-vs-Numba timing data from
`benchmarks/cxx_geqdsk.py` is summarized below. The three benchmark families
are GEQDSK-backed cases:

- `D-shaped`: `data/SOLOVEV.geqdsk`
- `H-mode`: `data/CHEASE.geqdsk`
- `X-point`: `data/EFIT.geqdsk`

`solution diff` is the maximum absolute Cxx-vs-Numba packed solution-vector
difference. Bold rows mark the representative High configuration for each GEQDSK
family.

| case(params)    |     Cxx (ms) |    Numba (ms) |     speedup | solution diff |
| --------------- | -----------: | ------------: | ----------: | ------------: |
| D-shaped(4)     |     0.172940 |      1.911798 |     11.055x |      1.17e-12 |
| D-shaped(5)     |     0.214899 |      2.136862 |      9.944x |      3.14e-12 |
| **D-shaped(9)** | **0.229824** |  **2.547320** | **11.084x** |  **8.04e-12** |
| D-shaped(75)    |     1.000413 |      7.288244 |      7.285x |      1.48e-10 |
|                 |              |               |             |               |
| H-mode(27)      |     0.707322 |      5.654250 |      7.994x |      3.12e-11 |
| H-mode(36)      |     0.844900 |      7.095212 |      8.398x |      2.75e-11 |
| **H-mode(60)**  | **1.705001** | **15.468430** |  **9.072x** |  **1.26e-08** |
| H-mode(130)     |     8.265414 |     43.981732 |      5.321x |      1.29e-08 |
|                 |              |               |             |               |
| X-point(19)     |     0.374766 |      3.517772 |      9.387x |      1.55e-11 |
| X-point(29)     |     0.549881 |      4.886525 |      8.887x |      3.92e-11 |
| **X-point(94)** | **1.975305** | **11.051075** |  **5.595x** |  **8.39e-11** |
| X-point(130)    |     4.267138 |     24.079790 |      5.643x |      2.99e-10 |

The package-level Kernel API is intentionally semantic: users construct
`KernelTopology` for the solve topology, including a source `constraint` of
`"ip"`, `"beta"`, `"both"`, or `"none"`, then pass it explicitly as
`Kernel(topology=topology)` or `build(topology=topology, ...)`.
`KernelBoundary`/`KernelSource` carry runtime cases, `KernelConfig` carries the
handle-level default solve policy, and `KernelRecipe` remains the shared backend
recipe type. `KernelSource` stores exactly one raw pressure representation:
either `p`, or `pprime` with an optional edge pressure `p0`. It also stores `Ip`
and `beta` plus exactly one route-specific driver: `ffprime` (PF), `psi_r` (PP),
`itor` (PI), `jtor` (PJ1), `jpara` (PJ2), `jtotal` (PJ3), or `q` (PQ). The selected driver must
match `KernelTopology.route`; the Kernel runtime derives `pprime`/`p0` from `p`
when needed and materializes route-dependent `mu0` scaling before calling
backend kernels. Sine-family Kernel
inputs are s1-started: `KernelTopology.s_counts=(n1, n2, ...)` and
`KernelBoundary.s_offsets=(s1, s2, ...)`; backend runtime lowering adds the
structural s0=0 slot. `KernelRecipe` defaults to `backend="numba"` for the direct
Numba runtime; `backend="cxx"` explicitly selects the native backend. Both backends
use the same public `Kernel` type and method surface, including residuals, solves,
finite-difference JVP/Jacobian calls, and `build_equilibrium()`.
`build(topology=..., recipe=None, config=None)` creates a reusable `Kernel` and
caches that default policy on the handle; `Kernel.solve(...)` can use it as-is,
replace it with a one-off `config=...`, or override individual fields such as
`method=...` for one call. An explicit `x0=` packed array, active-profile
coefficient dictionary, or previous `SolveResult` overrides warm continuation
and the configured cold initial-state policy.

The current production boundary is narrow: route/topology planning covers the
benchmark matrix, while native execution is gated by the Cxx native-support
validation helper. The artifact cache key is
computed from the canonical topology, explicit artifact recipe, Python/toolchain
ABI, the native CMake define contract, and a digest of native implementation
inputs. Artifacts are cached under `.veqpy-kernel-cache/` in the current working
directory by default, or under `VEQPY_KERNEL_CACHE` when set.
Runtime boundary/source arrays, physical constraints, solver tolerances, and `x0`
belong to the per-case solve call.

The Cxx backend pins short native calls to one CPU by default to reduce scheduler
noise. Set `VEQPY_CXX_PIN_CPU=0` to disable scoped pinning, or
`VEQPY_CXX_PIN_CPU_ID=<cpu>` to request a specific allowed CPU. For high-volume
loops, prefer one outer pinning scope via the Kernel handle rather than relying
on per-call affinity changes.

Useful Kernel checks from the repository root:

```bash
.venv/bin/python -m compileall -q veqpy tests/test_kernel_contract_api.py
.venv/bin/ruff check veqpy tests/test_kernel_contract_api.py
.venv/bin/python -m pytest tests/test_kernel_contract_api.py
```

Retained benchmark result artifacts live under `benchmarks/results/`. Future
timing evidence should use the shared Kernel dataclasses directly through
`veqpy.Kernel`, selecting `backend="numba"` or `backend="cxx"` through
`KernelRecipe`.

## Implementation Documentation

User-facing architecture notes:

- [`model.md`][model-doc]: responsibilities, snapshot boundaries, and diagnostic
  interfaces for `Grid`, `Profile`, `Geqdsk`, and `Equilibrium`.
- [`architecture.md`][architecture-doc]: package layers, dependency direction,
  and public construction entry points.
- [`kernel.md`][kernel-doc]: Kernel runtime boundary, solve lifecycle, result
  semantics, and warm continuation.
- [`backends.md`][backends-doc]: Numba/Cxx backend responsibilities, cache
  behavior, and benchmark entry points.
- [`release-1.3.1.md`][release-1.3.1]: 1.3.1 numerical-correctness,
  equilibrium-model, and performance improvements.
- [`release-1.3.0.md`][release-1.3.0]: 1.3.0 highlights, breaking changes, and
  migration examples.

Low-level base/math design notes for `Reactive`, `Serial`, `Registry`, interpolation,
quadrature, and calculus now live in the corresponding source module headers.

## References

VEQPy is associated with the companion manuscript **[Zhang2026]**. Related VEQ-family and representation papers include:

- [**[Zhang2026]**: primary VEQ paper on the fixed-boundary Grad-Shafranov solver][veq-arxiv]

  > _Ruohan Zhang, Huasheng Xie, Yueyan Li, Weiqi Meng, Feng Wang, and Zhengxiong Wang,
  > "VEQ: a fast parametric Grad-Shafranov solver for fixed-boundary tokamak equilibria
  > with flexible source profiles", arXiv:2606.11821, 2026._

- [**[Xie2026]**: minimum-parameter fixed-boundary Grad-Shafranov representation][veq-min-parameters-arxiv]

  > _Huasheng Xie and Yueyan Li,
  > "What Is the Minimum Number of Parameters Required to Represent Solutions of the
  > Grad-Shafranov Equation?", arXiv:2601.02942, 2026._

- [**[Li2026]**: VEQ-R toroidal-rotation effects in spherical-torus equilibria][veqr-arxiv]

  > _Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang,
  > "Investigation of Toroidal Rotation Effects on Spherical Torus Equilibria using
  > the Fast Spectral Solver VEQ-R", arXiv:2602.11422, 2026._

---

<p>
<img align="left" src="docs/assets/veqpy_icon.svg" width="150" alt="veqpy logo">

<strong>License</strong>:<br>
<em>BSD 3-Clause License</em><br>

<strong>Maintainer</strong> (rhzhang):<br>
<em>Homepage</em> - <em>https://zhangtakeda.github.io</em><br>
<em>Email</em> - <em>rhzhang@mail.dlut.edu.cn</em><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>zhangtakeda@gmail.com</em><br>

</p>

[package-badge]: https://img.shields.io/badge/package-veqpy-blue.svg
[python-badge]: https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white
[ci-badge]: https://img.shields.io/github/actions/workflow/status/zhangtakeda/veqpy/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white
[license-badge]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[arxiv-badge]: https://img.shields.io/badge/arXiv-2606.11821-b31b1b.svg?logo=arxiv&logoColor=white
[tests-badge]: https://img.shields.io/badge/tests-pytest-blue.svg
[pypi]: https://pypi.org/project/veqpy/
[python]: https://www.python.org/
[ci]: https://github.com/zhangtakeda/veqpy/actions/workflows/ci.yml
[license]: LICENSE
[benchmarks]: benchmarks/
[tests]: tests/
[architecture-doc]: docs/veqpy/architecture.md
[model-doc]: docs/veqpy/model.md
[release-1.3.1]: docs/veqpy/release-1.3.1.md
[release-1.3.0]: docs/veqpy/release-1.3.0.md
[kernel-doc]: docs/veqpy/kernel.md
[backends-doc]: docs/veqpy/backends.md
[veq-arxiv]: https://arxiv.org/abs/2606.11821
[veq-min-parameters-arxiv]: https://arxiv.org/abs/2601.02942
[veqr-arxiv]: https://arxiv.org/abs/2602.11422
