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
- **Explicit runtime boundary**: `Grid + Problem -> Operator -> Solver -> Equilibrium`
  separates packed coefficients, runtime workspaces, nonlinear solve orchestration, and
  post-solve snapshots. `Problem` is the public problem definition type.
- **GEQDSK workflow support**: GEQDSK I/O, fixed-boundary fitting from GEQDSK boundaries,
  snapshot export, flux-surface comparison, and common diagnostics.
- **Formula-oriented model objects**: `Problem` stores user-facing solve inputs and
  active profile lengths, while `Profile` is used for serializable shape-profile
  snapshots on `Equilibrium`. `Grid` and `Equilibrium` use reactive derived properties
  to lazily reconstruct geometry and physics diagnostics by formula.
- **Experimental VEQlib bridge**: `veqlib.facade` exposes a C++-aligned
  `KernelTopology + KernelRecipe + KernelBoundary + KernelSource + KernelConfig` API
  and builds/loads optional topology-specific VEQlib C++/nanobind kernels for the
  current MVP path.
  This is an optional acceleration path; the standard Python/Numba solver remains
  the default.

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

The optional VEQlib C++ kernel layer under `veqlib/` is not required for normal
Python/Numba use. Its `facade/` tree provides `veqlib.facade`, its `core/` tree
holds the C++/CMake implementation, and the top-level `benchmarks/` package holds
comparison scripts. It is intentionally separate from VEQPy internals. Building it requires a local
C++20 toolchain and native libraries such as CMake 3.24+, `clang++`, nanobind,
GCEM, nlohmann-json, CMINPACK, LAPACKE/LAPACK, and OpenBLAS.

All commands below use `.venv` explicitly; activating the environment is optional.

## Example Workflows

Basic demo:

```bash
.venv/bin/python examples/minimal_equilibrium.py
```

This script builds a smooth fixed-boundary problem, solves an equilibrium using PF(`psin`)
source input, writes an `Equilibrium` JSON snapshot, and generates a flux-surface figure.
By default, outputs go under `./outputs/minimal_equilibrium`; set
`VEQPY_OUTPUT_DIR` to choose another directory.

GEQDSK demo:

```bash
.venv/bin/python examples/geqdsk_workflow.py
```

This script reads an EFIT-style GEQDSK file, fits it as a VEQPy fixed boundary, solves
PF(`psin`) and PQ(`psin`) cases with an `Ip` constraint using one-dimensional source
profiles from the GEQDSK file, and writes a two-column VEQPy-vs-GEQDSK comparison
figure. By default, it reads `./data/EFIT.geqdsk` and writes outputs under
`./outputs/geqdsk_workflow`; set `VEQPY_GEQDSK` and `VEQPY_OUTPUT_DIR` to override
those paths. Manuscript-oriented reproduction scripts are available under
[`scripts/`][scripts]; they are heavier than the minimal examples and may write
paper/data artifacts rather than user-demo outputs.

## Development Checks

Core local checks mirror the push/PR CI workflow.

```bash
.venv/bin/python -m compileall -q veqpy tests examples veqlib/facade benchmarks scripts
.venv/bin/ruff check veqpy tests examples veqlib/facade benchmarks scripts
.venv/bin/python -m pytest
```

## Optional C++ Kernels

VEQlib is the experimental C++/nanobind kernel layer used for topology-specific
shared-library kernels and VEQPy-vs-native benchmarks.

Representative VEQlib-vs-Numba timing snapshot from `benchmarks.veqlib_geqdsk_pareto`
using the production nanobind/shared-library path is shown below. The three benchmark
families are GEQDSK-backed cases:

- `D-shaped`: `data/SOLOVEV.geqdsk`
- `H-mode`: `data/CHEASE.geqdsk`
- `X-point`: `data/EFIT.geqdsk`

Timings are median full nonlinear-solve wall times in milliseconds after warmup;
the VEQlib timing excludes native build time. `solution diff` is the maximum absolute
VEQlib-vs-Numba solution-vector difference for the benchmark case. Bold rows mark the
representative High configuration for each GEQDSK family.

| case(params)    |  VEQlib (ms) |    Numba (ms) |     speedup | solution diff |
| --------------- | -----------: | ------------: | ----------: | ------------: |
| D-shaped(4)     |     0.038843 |      0.988233 |     25.442x |      2.73e-09 |
| D-shaped(5)     |     0.044019 |      1.076983 |     24.466x |      9.20e-09 |
| **D-shaped(9)** | **0.066897** |  **1.384216** | **20.692x** |  **1.25e-08** |
| D-shaped(75)    |     0.778386 |      6.778595 |      8.709x |      1.47e-08 |
| H-mode(27)      |     0.220681 |      5.020843 |     22.752x |      1.89e-07 |
| H-mode(36)      |     0.294272 |      6.077891 |     20.654x |      3.87e-08 |
| **H-mode(60)**  | **0.551778** | **13.426837** | **24.334x** |  **3.46e-09** |
| H-mode(130)     |     2.537298 |     42.276592 |     16.662x |      1.36e-07 |
| X-point(19)     |     0.138631 |      2.667478 |     19.242x |      3.51e-09 |
| X-point(29)     |     0.243494 |      3.624370 |     14.885x |      4.51e-08 |
| **X-point(94)** | **1.315509** | **10.269229** |  **7.806x** |  **3.44e-08** |
| X-point(130)    |     2.530054 |     21.647612 |      8.556x |      1.24e-09 |

The package-level Python facade is intentionally semantic: users construct
`KernelTopology` for the physical/native topology, `KernelRecipe` for packed
layout and build options, `KernelBoundary`/`KernelSource` for runtime cases,
and `KernelConfig` for the handle-level default solve policy.
`build(..., recipe=None, config=None)` creates a reusable `Kernel` and caches
that default policy on the handle; `Kernel.solve(...)` can use it as-is, replace
it with a one-off `config=...`, or override individual fields such as
`method=...` for one call. `solve(...)` is the one-shot convenience path and
`clean(...)` cleans artifact cache entries. The lower-level artifact primitive
is named `compile(topology, recipe=...)` and returns `CompileResult`.
Option-code, CPU-affinity, and cache-root helpers live in their focused
submodules. Python drives CMake with explicit `-D...` definitions and loads the
resulting artifact through `veqlib.facade`.

The current production boundary is narrow: route/topology planning covers the
benchmark matrix, while native execution is gated by
`KernelTopology.validate_supported_for_veqlib_native()`. The artifact cache key is
computed from the canonical topology, explicit compile recipe, Python/toolchain
ABI, the native CMake define contract, and a digest of implementation inputs under
`veqlib/core` (`.h`, `.cpp`, `.in`, and `CMakeLists.txt`). Artifacts are cached
under `veqlib/artifact/` by default, or under `VEQLIB_KERNEL_CACHE` when set.
Runtime boundary/source arrays, physical constraints, solver tolerances, and `x0`
belong to the per-case solve call.

The facade pins short native calls to one CPU by default to reduce scheduler
noise. Set `VEQLIB_PIN_CPU=0` to disable scoped pinning, or
`VEQLIB_PIN_CPU_ID=<cpu>` to request a specific allowed CPU. For high-volume
loops, prefer one outer pinning scope via the facade handle rather than relying
on per-call affinity changes.

Useful VEQlib checks from the repository root:

```bash
.venv/bin/python -m compileall -q veqlib/facade tests/test_veqlib_facade_api.py
.venv/bin/ruff check veqlib/facade tests/test_veqlib_facade_api.py
.venv/bin/python -m pytest tests/test_veqlib_facade_api.py
```

Pure VEQPy/Numba benchmark entrypoints:

```bash
.venv/bin/python -m benchmarks.veqpy_routes \
  --repeat 100 \
  --warmup 5 \
  --output benchmarks/results/manual/veqpy_routes.json

.venv/bin/python -m benchmarks.veqpy_geqdsk_routes \
  --repeat 1 \
  --warmup 5 \
  --output benchmarks/results/manual/veqpy_geqdsk_routes.json
```

`benchmarks.veqpy_geqdsk_routes` defaults to the Solovev GEQDSK case; pass
`--geqdsk /path/to/case.geqdsk` to run another input. The `--repeat 1` example
is a quick smoke run; use larger repeat counts for timing evidence.

Route/topology comparison benchmark:

```bash
.venv/bin/python -m benchmarks.veqlib_routes \
  --build fastmath \
  --repeat 100 \
  --warmup 5 \
  --output benchmarks/results/manual/routes.json
```

GEQDSK configuration benchmark:

```bash
.venv/bin/python -m benchmarks.veqlib_geqdsk_pareto \
  --build fastmath \
  --repeat 100 \
  --warmup 5 \
  --output benchmarks/results/manual/geqdsk_configs.json
```

Continuation-policy effective-nfev benchmark:

```bash
.venv/bin/python -m benchmarks.veqlib_continuation
```

Executable-side C++ diagnostics are secondary. Retained performance evidence
should use the production nanobind/shared-library path through `veqlib.facade`,
paired with VEQPy correctness comparison.

## Implementation Documentation

User-facing architecture notes:

- [`model.md`][model-doc]: responsibilities, snapshot boundaries, and diagnostic
  interfaces for `Grid`, `Profile`, `Boundary`, `Geqdsk`, and `Equilibrium`.
- [`operator.md`][operator-doc]: source routes, packed state, stage pipeline,
  and runtime/snapshot separation.
- [`solver.md`][solver-doc]: nonlinear solve lifecycle, fallback behavior,
  residual normalization, and collocation polish.

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
