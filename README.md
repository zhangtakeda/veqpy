# VEQPy

VEQPy 2.x is the FusionPRIME fixed-boundary, axisymmetric equilibrium Module.
It consumes standalone boundary, source, and target dictionaries, solves
through its required Numba path or an optional Cxx path, and can materialize a
new frozen base `Equilibrium`.

## Public API

Users configure a prepared Module with ordinary mappings. The four named
Kernel ABI records (`KernelTopology`, `KernelInput`, `KernelConfig`, and
`KernelOutput`) are private implementation data and are not exported from
`veqpy`.

```python
import veqpy
from veqpy.demo_case import make_demo_inputs

topology = {
    "Nr": 8,
    "Nt": 12,
    "route": "PF",
    "coordinate": "psin",
    "constraint": "ip",
    "h_count": 3,
    "v_count": 3,
    "kappa_count": 3,
    "psin_count": 6,
    "F_count": 0,
    "c_counts": (3, 3, 3),
    "s_counts": (3, 3),
    "quadrature": "legendre",
    "calculus": "spectral",
}

module = veqpy.build(topology=topology, backend="numba")
try:
    boundary, source, targets = make_demo_inputs()
    record = module.solve(
        boundary=boundary,
        source=source,
        targets=targets,
        verbose=False,
    )
finally:
    module.close()
```

The topology keeps the numerical discretization fields `quadrature`,
`calculus`, `L_max`, `M_max`, and `K_max`. Source nodes are always explicit
runtime data. The Adapter owns a source buffer with
capacity epochs 256, 512, and 1024; a larger source is rejected before solve,
and growing the buffer does not rebuild the topology or backend artifact.

`boundary` has exactly `a`, `R0`, `Z0`, `B0`, `kappa_lcfs`, `c_lcfs`, and
`s_lcfs`. `source` has exactly one coordinate key selected by topology (`r`,
`rho`, or `psin`), one pressure representation (`P` or the matching
`P_r`/`P_rho`/`P_psin`), and the selected route driver. A derivative pressure
also requires `P0`; a full `P` forbids it. `targets` has exactly the values
selected by `constraint`: `Ip`, `beta`, both, or neither. Coordinate, pressure,
and driver arrays share one shape and grid; the Adapter never remaps one source
profile onto another. In particular, a `rho` source never requires `rho_r`.

Build defaults are `materialize=True`, `verbose=True`, `report=False`, and
`report_dir=None`. Each `run()`/`solve()` call can override materialization,
verbosity, reporting, report directory, and (for `solve()`) a partial solver
mapping. `verbose=True` prints a compact Rich KernelOutput diagnostic;
`report=True` writes a complete KT/KC/KI/KO JSON snapshot through
`veqpy.io.write_report`. The default report directory is `Path.cwd()/report`
and the filename is `veqpy-YYYYMMDD-HHMMSS-ffffff.json`; an exact timestamp
collision intentionally overwrites the previous file.

## Installation and CLI

Python 3.12+ is required. In the FusionPRIME workspace:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e "."
.venv/bin/python -m veqpy --version
.venv/bin/python -m veqpy --check
.venv/bin/python -m veqpy --demo numba
.venv/bin/python -m veqpy --demo cxx-strict
.venv/bin/python -m veqpy --links
```

`python -m veqpy` is a thin forwarder to `veqpy.cli`. The CLI only exercises
the public three-dictionary path.

GEQDSK is a pure file payload owned by `fusionprime-base`:
`fusionprime_base.io.geqdsk` provides `Geqdsk`, `load_geqdsk`, and
`save_geqdsk`. VEQPy does not convert GEQDSK files to or from Equilibrium and
does not include a plotting layer.

## Backends

The supported backend strings are:

- `numba`: required strict floating-point path;
- `cxx`: normalized to the Release relaxed artifact;
- `cxx-strict`: independent Release strict artifact;
- `cxx-relaxed`: independent Release relaxed artifact;
- `cxx-enzyme`: independent Release relaxed primal plus Enzyme residual derivatives.

`cxx-enzyme` preserves its own backend token, report metadata, cache namespace,
and artifact identity. Its public Kernel residual JVP/Jacobian and nonlinear
solver derivatives both use Enzyme; plugin or version failure during a new
build is fatal and never falls back to finite differences or `cxx-relaxed`.
These are residual derivatives with respect to the packed unknowns, not the
complete implicit Module solve-map derivative.

`artifact_dir`, `cpu_affinity`, and `rebuild` are build-only Module options.
Numba contains no fast-math decorators. Cxx relaxed may use fast-math where
safe and keeps standard-library fallbacks for special functions that require
strict evaluation.

After the first successful build, VEQPy publishes a build-tool-free runtime
pointer and loads the selected extension through `fusionprime_base.native`.
A normal cached `prepare()` therefore does not probe Clang/CMake, hash sources
or binaries, update artifact metadata, or repeat preparation before the first
solve. A missing or malformed selected artifact is reported directly instead
of triggering a silent rebuild. In an editable checkout, native source changes
must be made explicit with `rebuild=True`; released package versions and native
runtime schemas invalidate their runtime pointers independently.

On macOS arm64, native preparation uses the matching Homebrew LLVM/Clang 22
and Enzyme toolchain:

```bash
brew install enzyme
$(brew --prefix llvm@22)/bin/clang++ --version
```

The reference versions are LLVM `22.1.8` and Enzyme `0.0.290`. ClangEnzyme is
an LLVM compiler plugin, so its LLVM major must match the compiler major. It is
a system build dependency rather than a Python package dependency. All VEQPy
Cxx environments use LLVM major 22; Ubuntu 24.04 CI installs `clang-22` and
`lld-22` from the official apt.llvm.org repository. Enzyme builds on every
platform require a ClangEnzyme plugin and headers built for the same LLVM 22
major; there is no platform-specific fallback or alternate compiler contract.
For this reference toolchain the production dense Jacobian uses scalar forward
columns. Enzyme's vector-width intrinsic remains outside the supported build
contract because `0.0.290` crashes LLVM 22 type analysis on the full residual.
The plugin binary hash, Enzyme header hash, paths, version, compiler version,
and generated flags participate in new artifact identity. Cached artifacts
still load through the build-tool-free pointer without probing that toolchain.

## Development gates

The migration intentionally removes the old VEQPy pytest suite and pytest
configuration. Structural evidence is collected with:

```bash
.venv/bin/python -m compileall -q veqpy
.venv/bin/ruff check veqpy benchmarks demo.py demo_geqdsk.py
.venv/bin/python demo.py
.venv/bin/python demo_geqdsk.py
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

The maintained benchmark entry points are `benchmarks/cxx_geqdsk.py` for
three-file backend qualification and `benchmarks/kernel_routes.py` for the
fixed `4 Kernel x 7 route x 3 coordinate` matrix. Formal timings exclude
artifact builds, imports, and Numba JIT preparation. This development branch
is not declared release-ready solely because these structural gates pass.

## Repository boundary

This repository contains only the VEQPy migration. MCDPy, VTSPy, and the
top-level FusionPRIME implementation are outside this task's scope.
