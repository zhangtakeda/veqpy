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
- **Unified source route layer**: PF, PP, PI, PJ1, PJ2, and PQ routes map pressure-gradient,
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
as CMake 3.24+, `clang++`, nanobind, GCEM, nlohmann-json, CMINPACK,
LAPACKE/LAPACK, and OpenBLAS.

All commands below use `.venv` explicitly; activating the environment is optional.

## Example Workflows

Basic demo:

```bash
.venv/bin/python examples/minimal_equilibrium.py
```

This script builds a smooth fixed-boundary Kernel case, solves an equilibrium using PF(`psin`)
source input, writes an `Equilibrium` JSON snapshot, and generates a flux-surface figure.
By default, outputs go under `./outputs/minimal_equilibrium`; set
`VEQPY_OUTPUT_DIR` to choose another directory.

GEQDSK demo:

```bash
.venv/bin/python examples/geqdsk_workflow.py
```

This script reads a GEQDSK file, fits it as a VEQPy fixed boundary, solves a small
PF(`psin`) Kernel case with an `Ip` constraint using one-dimensional source profiles
from the GEQDSK file, and writes a comparison figure. By default, it reads
`./data/SOLOVEV.geqdsk` and writes outputs under
`./outputs/geqdsk_workflow`; set `VEQPY_GEQDSK` and `VEQPY_OUTPUT_DIR` to override
those paths. Manuscript-oriented reproduction scripts are available under
[`scripts/`][scripts]; they are heavier than the minimal examples and may write
paper/data artifacts rather than user-demo outputs.

## Development Checks

Core local checks mirror the push/PR CI workflow.

```bash
.venv/bin/python -m compileall -q veqpy tests examples benchmarks scripts
.venv/bin/ruff check veqpy tests examples benchmarks scripts
.venv/bin/python -m pytest
```

## Optional C++ Kernels

The Cxx backend is the native C++/nanobind kernel layer used for
topology-specific shared-library kernels and VEQPy-vs-native benchmarks.

Representative Cxx-vs-Numba timing data retained in
`benchmarks/results/veqlib_geqdsk.json` is summarized below. The three benchmark
families are GEQDSK-backed cases:

- `D-shaped`: `data/SOLOVEV.geqdsk`
- `H-mode`: `data/CHEASE.geqdsk`
- `X-point`: `data/EFIT.geqdsk`

Timings are median full nonlinear-solve wall times in milliseconds after warmup;
the Cxx timing excludes native build time. `solution diff` is the maximum absolute
Cxx-vs-Numba solution-vector difference for the benchmark case. Bold rows mark the
representative High configuration for each GEQDSK family.

| case(params)    |     Cxx (ms) |    Numba (ms) |     speedup | solution diff |
| --------------- | -----------: | ------------: | ----------: | ------------: |
| D-shaped(4)     |     0.077366 |      1.095349 |     14.158x |      1.17e-12 |
| D-shaped(5)     |     0.089723 |      1.202812 |     13.406x |      3.14e-12 |
| **D-shaped(9)** | **0.129603** |  **1.515981** | **11.697x** |  **8.04e-12** |
| D-shaped(75)    |     1.218136 |      6.718363 |      5.515x |      1.48e-10 |
| H-mode(27)      |     0.615500 |      4.967208 |      8.070x |      3.12e-11 |
| H-mode(36)      |     0.783802 |      6.142815 |      7.837x |      2.76e-11 |
| **H-mode(60)**  | **2.053660** | **13.310995** |  **6.482x** |  **1.46e-08** |
| H-mode(130)     |    12.440129 |     41.164805 |      3.309x |      6.11e-09 |
| X-point(19)     |     0.263450 |      2.660089 |     10.097x |      1.55e-11 |
| X-point(29)     |     0.443195 |      3.635614 |      8.203x |      3.92e-11 |
| **X-point(94)** | **2.502630** |  **9.804938** |  **3.918x** |  **8.34e-11** |
| X-point(130)    |     6.209815 |     21.124163 |      3.402x |      2.99e-10 |

The package-level Kernel API is intentionally semantic: users construct
`KernelTopology` for the solve topology, including `ip_constraint` and
`beta_constraint` boolean source constraints, then pass it explicitly as
`Kernel(topology=topology)` or `build(topology=topology, ...)`.
`KernelBoundary`/`KernelSource` carry runtime cases, `KernelConfig` carries the
handle-level default solve policy, and `KernelRecipe` remains the shared backend
recipe type. `KernelSource` stores raw user-facing `heat_profile`,
`current_profile`, `Ip`, and `beta` values; the Kernel runtime materializes
route-dependent `mu0` scaling before calling backend kernels. Sine-family Kernel
inputs are s1-started: `KernelTopology.s_counts=(n1, n2, ...)` and
`KernelBoundary.s_offsets=(s1, s2, ...)`; backend runtime lowering adds the
structural s0=0 slot. `KernelRecipe` selects `backend="numba"` for the direct
Numba runtime or `backend="cxx"` for the native backend. Both backends use the
same public `Kernel` type and method surface, including residuals, solves,
finite-difference JVP/Jacobian calls, and `build_equilibrium()`.
`build(topology=..., recipe=None, config=None)` creates a reusable `Kernel` and
caches that default policy on the handle; `Kernel.solve(...)` can use it as-is,
replace it with a one-off `config=...`, or override individual fields such as
`method=...` for one call.

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
noise. Set `VEQLIB_PIN_CPU=0` to disable scoped pinning, or
`VEQLIB_PIN_CPU_ID=<cpu>` to request a specific allowed CPU. For high-volume
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
  interfaces for `Grid`, `Profile`, `Boundary`, `Geqdsk`, and `Equilibrium`.
- [`operator.md`][operator-doc]: Kernel runtime boundary, source materialization,
  and packed runtime responsibilities.
- [`solver.md`][solver-doc]: Kernel solve lifecycle, result semantics, residual
  normalization, and warm continuation.

Low-level base/math design notes for `Reactive`, `Serial`, `Registry`, interpolation,
quadrature, and calculus now live in the corresponding source module headers.

## Paper and Reproducibility Resources

VEQPy is associated with the companion manuscript **"VEQ: a fast parametric
Grad--Shafranov solver for fixed-boundary tokamak equilibria with flexible source
inputs"**. The repository includes manuscript-oriented figure scripts under
[`scripts/`][scripts] and benchmark helpers under [`benchmarks/`][benchmarks]. Tagged
release artifacts can pin rendered figures, generated reference data, and dependency
metadata for archival reproduction.

Related VEQ-family and representation papers include:

- Ruohan Zhang, Huasheng Xie, Yueyan Li, Weiqi Meng, Feng Wang, and Zhengxiong Wang,
  "VEQ: a fast parametric Grad-Shafranov solver for fixed-boundary tokamak equilibria
  with flexible source profiles", arXiv:2606.11821, 2026.
  [https://arxiv.org/abs/2606.11821][veq-arxiv]
- Huasheng Xie and Yueyan Li, "What Is the Minimum Number of Parameters Required to
  Represent Solutions of the Grad-Shafranov Equation?", arXiv:2601.02942, 2026.
  [https://arxiv.org/abs/2601.02942][veq-min-parameters-arxiv]
- Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang, "Investigation of Toroidal
  Rotation Effects on Spherical Torus Equilibria using the Fast Spectral Solver VEQ-R",
  arXiv:2602.11422, 2026.
  [https://arxiv.org/abs/2602.11422][veqr-arxiv]

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
[tests]: tests/
[scripts]: scripts/
[benchmarks]: benchmarks/
[model-doc]: docs/veqpy/model.md
[operator-doc]: docs/veqpy/operator.md
[solver-doc]: docs/veqpy/solver.md
[veq-arxiv]: https://arxiv.org/abs/2606.11821
[veq-min-parameters-arxiv]: https://arxiv.org/abs/2601.02942
[veqr-arxiv]: https://arxiv.org/abs/2602.11422
