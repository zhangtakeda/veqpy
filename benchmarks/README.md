# Benchmarks

This directory contains VEQPy Kernel benchmark entry points and result artifacts.
Current scripts use the Kernel API directly: Numba rows call
`veqpy.Kernel` with `backend="numba"`, and Cxx rows call the same
`veqpy.Kernel` surface with `backend="cxx"`.

The route and GEQDSK tables below report median `SolveResult.elapsed_ms`.
Each benchmark row reuses one Kernel handle across warmup and timed repeats;
the solver policy remains `continuation=cold`, so accepted solutions are not
reused between repeats. The JSON outputs also keep outer `wall_timing` samples
around each solve call. Cxx artifact build and first native workspace setup are
excluded from the reported medians by warmup; build metadata and sample timings
are recorded in the JSON outputs.

`cxx_continuation.py` is the explicit continuation-policy benchmark. Its
table reports effective function evaluations rather than solve-time medians.
`numba_variant_sweep.py` is also separate from the solve-time tables: it reports
a single-pass `Kernel.variant()` switch cost against fresh Numba `Kernel`
construction for the same active GEQDSK topology.
`numba_pareto.py` is a lightweight GEQDSK screening entry point for the
Numba-only `Kernel.pareto()` evaluator. It builds a deterministic full/partial
GEQDSK signature queue, then records counts, time, complexity, and R-only
shape-error frontier samples.

## Scripts

- `numba_routes.py`: Numba backend synthetic route matrix through `veqpy.Kernel`.
- `numba_variant_sweep.py`: Numba `Kernel.variant()` construction-cost comparison.
- `numba_pareto.py`: Numba `Kernel.pareto()` reduced-candidate screening benchmark.
- `cxx_routes.py`: Cxx backend route matrix compared with the Numba backend.
- `cxx_geqdsk.py`: GEQDSK Cxx backend matrix compared with the Numba backend.
- `cxx_continuation.py`: Cxx backend continuation-policy benchmark.
- `cxx_boundary_fitters.py`: boundary R/Z scatter-to-coefficient fitter method/backend matrix.
- `_common.py`: shared Kernel-case construction, timing, route specs, and JSON helpers.

## Reproduce

```bash
.venv/bin/python benchmarks/numba_routes.py --quiet-progress
.venv/bin/python benchmarks/numba_variant_sweep.py --quiet-progress
.venv/bin/python benchmarks/numba_pareto.py --sweep-mode partial --no-write
.venv/bin/python benchmarks/cxx_routes.py --quiet-progress
.venv/bin/python benchmarks/cxx_geqdsk.py --quiet-progress
.venv/bin/python benchmarks/cxx_continuation.py --quiet-progress
.venv/bin/python benchmarks/cxx_boundary_fitters.py --quiet-progress
```
