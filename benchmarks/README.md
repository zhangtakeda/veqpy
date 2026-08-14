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

- `numba_routes.py`: Numba backend geometric-r synthetic route matrix through
  `veqpy.Kernel`; `--pj2-f-count 0` (the default) selects strict r PJ2/PJ3,
  while a positive value retains the optimized-F comparison. psin routes remain
  focused physics/API tests rather than performance qualification rows.
- `numba_variant_sweep.py`: Numba `Kernel.variant()` construction-cost comparison.
- `numba_pareto.py`: Numba `Kernel.pareto()` reduced-candidate screening benchmark.
- `rho_fixed_point_experiment.py`: local `sqrt(Phi_N)` coordinate fixed-point
  convergence-domain scan across every route and legal source constraint.
- `psin_internal_closure_experiment.py`: rejected source-local `psin` ownership
  experiment; distinguishes fixed-equilibrium map convergence from complete
  nonlinear-solve qualification.
- `explicit_source_benchmark.py`: same-count uniform versus retained arbitrary
  nodes for an edge-localized source feature, measured against a dense native
  `rho` reference.
- `explicit_derivative_experiment.py`: rollback qualification for deriving
  the coordinate-matched pressure derivative and PI `itor_r` from the same
  retained r/explicit PCHIP as their primitive values.
- `pp_explicit_derivative_experiment.py`: PP r/explicit comparison of the
  retained-PCHIP plus analytic-axis derivative against the legacy remap-then-D
  path under all four legal source constraints.
- `native_source_performance.py`: all-route r/psin/rho uniform versus
  runtime-explicit hot-residual matrix after Numba source-loop fusion.
- `pj23_rho_joint_experiment.py`: production-form, single-layer PJ2/PJ3
  `(sqrt(Phi_N), u, C)` convergence-domain scan.
- `rho_route_benchmark.py`: warmed same-node r versus `sqrt(Phi_N)` Numba
  hot-residual and complete-solve comparison.
- `rho_geqdsk_benchmark.py`: warmed official GEQDSK PF/psin versus native
  PF/`sqrt(Phi_N)` complete-solve comparison with qualification gating.
- `cxx_routes.py`: Cxx backend geometric-r route matrix compared with the
  Numba backend.
- `cxx_geqdsk.py`: GEQDSK Cxx backend matrix compared with the Numba backend.
- `cxx_continuation.py`: Cxx backend continuation-policy benchmark.
- `cxx_boundary_fitters.py`: boundary R/Z scatter-to-coefficient fitter method/backend matrix.
- `_common.py`: shared Kernel-case construction, timing, route specs, and JSON helpers.

## Reproduce

```bash
.venv/bin/python benchmarks/numba_routes.py --quiet-progress
.venv/bin/python benchmarks/numba_routes.py --case PJ2 --case PJ3 --pj2-f-count 6 --no-write
.venv/bin/python benchmarks/numba_variant_sweep.py --quiet-progress
.venv/bin/python benchmarks/numba_pareto.py --sweep-mode partial --no-write
.venv/bin/python benchmarks/rho_fixed_point_experiment.py \
  --output benchmarks/results/rho_fixed_point_scan.json
.venv/bin/python benchmarks/psin_internal_closure_experiment.py \
  --output benchmarks/results/psin_internal_closure_scan.json
.venv/bin/python benchmarks/explicit_source_benchmark.py \
  --output benchmarks/results/explicit_source_benchmark.json
.venv/bin/python benchmarks/explicit_derivative_experiment.py \
  --output benchmarks/results/explicit_derivative_experiment.json
.venv/bin/python benchmarks/pp_explicit_derivative_experiment.py \
  --output benchmarks/results/pp_explicit_derivative_experiment.json
.venv/bin/python benchmarks/native_source_performance.py \
  --output benchmarks/results/native_source_performance.json
.venv/bin/python benchmarks/rho_route_benchmark.py \
  --output benchmarks/results/rho_route_benchmark.json
.venv/bin/python benchmarks/pj23_rho_joint_experiment.py \
  --output benchmarks/results/pj23_rho_joint_experiment.json
.venv/bin/python benchmarks/rho_geqdsk_benchmark.py \
  --output benchmarks/results/rho_geqdsk_benchmark.json
.venv/bin/python benchmarks/cxx_routes.py --quiet-progress
.venv/bin/python benchmarks/cxx_geqdsk.py --quiet-progress
.venv/bin/python benchmarks/cxx_continuation.py --quiet-progress
.venv/bin/python benchmarks/cxx_boundary_fitters.py --quiet-progress
```
