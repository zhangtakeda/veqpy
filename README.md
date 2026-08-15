# VEQPy

VEQPy 2.x is the FusionPRIME fixed-boundary, axisymmetric equilibrium Module.
It solves a finite-dimensional Grad–Shafranov representation with a required
Numba backend and an optional Cxx backend, then materializes a new frozen
`fusionprime-base.Equilibrium`.

## Public architecture

The public high-level path is:

```text
frozen Plasma -> VEQAdapter -> four-buffer Kernel -> base Equilibrium -> VEQRecord
```

The only named data types in the low-level Kernel ABI are:

1. `KernelTopology`: immutable structure, route/capacity, and compiled shapes;
2. `KernelInput`: preallocated numeric case buffers;
3. `KernelConfig`: numeric solver policy codes and tolerances;
4. `KernelOutput`: identity-stable diagnostics and materialization buffers.

`VEQ` is a base `@module` with the contract
`run(*, plasma: Plasma, materialize=True) -> VEQRecord`. A successful Record
contains a new frozen base `Equilibrium`; the input Plasma is never modified.

Dynamic source coordinates use `KernelTopology.source_capacity` at prepare time
and `KernelInput.source_count` at run time. The active prefix is validated and
the unused suffix is deterministically zeroed, so buffer shape and identity do
not change between cases.

## Installation

Python 3.12+ is required. For a checkout in the FusionPRIME workspace:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,plot]"
```

The workspace development environment is `/Users/zhangtakeda/代码/FusionPRIME/.venv`
when working in the five-repository checkout. `fusionprime-base` is a runtime
dependency and owns the physical State, Plasma, Module, Record, and JVP
contracts.

## Demos and CLI

```bash
.venv/bin/python demo.py
.venv/bin/python demo_geqdsk.py
.venv/bin/python -m veqpy --version
.venv/bin/python -m veqpy --check
.venv/bin/python -m veqpy --demo numba
.venv/bin/python -m veqpy --demo cxx
.venv/bin/python -m veqpy --links
```

`demo.py` writes two optional Matplotlib figures and a base Equilibrium JSON
snapshot. `demo_geqdsk.py` reads the tracked `data/SOLOVEV.geqdsk`, solves it
through `VEQ`, exports a GEQDSK payload with a closed LCFS, and writes a local
comparison figure.

## Backends and source capability

Numba is the required default. Cxx is parity-tested for the shared `r`/
`psin` and fixed-node intersection. Cxx currently rejects explicit source
nodes and `rho` closure with an explicit capability error; it never silently
falls back to Numba. See [`docs/veqpy/backends.md`](docs/veqpy/backends.md).

## Development gates

```bash
.venv/bin/python -m compileall -q veqpy tests benchmarks demo.py demo_geqdsk.py
.venv/bin/ruff check veqpy tests benchmarks demo.py demo_geqdsk.py
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

The external wheel smoke installs the built wheel into a fresh environment,
imports the four ABI types, runs the Numba CLI check, and performs a minimal
GEQDSK round-trip. The benchmark entry point is
`benchmarks/benchmark_v2.py`.

## Repository boundary

This repository contains only the VEQPy migration. MCDPy, VTSPy, and the
top-level FusionPRIME implementation are outside this task's scope.
