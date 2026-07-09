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
`numba_pareto.py` is a lightweight smoke entry point for the Numba-only
`Kernel.pareto()` topology-reduction interface; it records counts, time,
complexity, and R-only shape-error frontier samples.

## Scripts

- `numba_routes.py`: Numba backend synthetic route matrix through `veqpy.Kernel`.
- `numba_variant_sweep.py`: Numba `Kernel.variant()` construction-cost comparison.
- `numba_pareto.py`: Numba `Kernel.pareto()` topology-reduction smoke benchmark.
- `cxx_routes.py`: Cxx backend route matrix compared with the Numba backend.
- `cxx_geqdsk_pareto.py`: GEQDSK Cxx backend matrix compared with the Numba backend.
- `cxx_continuation.py`: Cxx backend continuation-policy benchmark.
- `cxx_boundary_fitters.py`: boundary R/Z scatter-to-coefficient fitter method/backend matrix.
- `_common.py`: shared Kernel-case construction, timing, route specs, and JSON helpers.

## Reproduce

```bash
.venv/bin/python benchmarks/numba_routes.py --quiet-progress
.venv/bin/python benchmarks/numba_variant_sweep.py --quiet-progress
.venv/bin/python benchmarks/numba_pareto.py --no-write
.venv/bin/python benchmarks/cxx_routes.py --quiet-progress
.venv/bin/python benchmarks/cxx_geqdsk_pareto.py --quiet-progress
.venv/bin/python benchmarks/cxx_continuation.py --quiet-progress
.venv/bin/python benchmarks/cxx_boundary_fitters.py --quiet-progress
```

## Environment

- CPU: `Intel(R) Core(TM) i5-14600KF`
- Python: `python 3.12.3 [GCC 13.3.0]`
- CMake: `cmake version 3.28.3`
- C++: `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- Thread env: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

## Results

### Numba Synthetic Route Matrix

- scope: `ip-uniform`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.
- timing source: `SolveResult.elapsed_ms`; default initial policy: `cold`.

| case                | status | x   | Numba ms | nfev | residual | shape    | psi_r RMS | FF_psi RMS |
| ------------------- | ------ | --- | -------- | ---- | -------- | -------- | --------- | ---------- |
| PF_rho_uniform_Ip   | passed | 12  | 1.363524 | 29   | 4.06e-10 | 1.66e-04 | 8.53e-05  | 2.51e-03   |
| PF_psin_uniform_Ip  | passed | 18  | 1.742151 | 40   | 9.81e-09 | 2.95e-03 | 1.43e-03  | 3.84e-03   |
| PP_rho_uniform_Ip   | passed | 12  | 1.541065 | 39   | 3.83e-11 | 2.05e-04 | 4.94e-05  | 4.21e-03   |
| PP_psin_uniform_Ip  | passed | 18  | 1.886485 | 52   | 5.56e-10 | 6.23e-03 | 2.61e-03  | 2.40e-02   |
| PI_rho_uniform_Ip   | passed | 12  | 1.315531 | 28   | 7.50e-08 | 1.52e-04 | 3.86e-05  | 1.38e-03   |
| PI_psin_uniform_Ip  | passed | 18  | 2.089362 | 55   | 2.87e-12 | 1.91e-03 | 6.64e-04  | 9.37e-03   |
| PJ1_rho_uniform_Ip  | passed | 12  | 1.329319 | 29   | 1.11e-09 | 1.68e-04 | 4.40e-05  | 3.39e-04   |
| PJ1_psin_uniform_Ip | passed | 18  | 1.686264 | 40   | 8.15e-09 | 3.13e-03 | 1.52e-03  | 9.67e-03   |
| PJ2_rho_uniform_Ip  | passed | 18  | 2.298635 | 82   | 7.99e-09 | 1.26e-04 | 4.30e-05  | 8.81e-04   |
| PJ2_psin_uniform_Ip | passed | 18  | 2.810067 | 66   | 2.97e-09 | 3.50e-03 | 1.80e-03  | 1.18e-02   |
| PQ_rho_uniform_Ip   | passed | 12  | 1.570927 | 29   | 6.60e-08 | 1.35e-04 | 2.81e-05  | 1.34e-03   |
| PQ_psin_uniform_Ip  | passed | 18  | 2.066588 | 42   | 2.63e-09 | 5.00e-03 | 2.56e-03  | 2.18e-02   |

### Cxx Synthetic Route Matrix

- scope: `ip-uniform`; build: `fastmath`; layout: `degree`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.
- timing source: `SolveResult.elapsed_ms`; Cxx policy: `initial=cold`, `continue=cold`, `norm=fast`.
- `diff` is `x_max_abs` between Cxx and Numba packed solutions.

| case                | status | x   | Cxx ms   | Numba ms | speedup | diff     |
| ------------------- | ------ | --- | -------- | -------- | ------- | -------- |
| PF_rho_uniform_Ip   | passed | 12  | 0.195199 | 1.349844 | 6.915x  | 2.47e-11 |
| PF_psin_uniform_Ip  | passed | 18  | 0.262638 | 1.724206 | 6.565x  | 2.05e-11 |
| PP_rho_uniform_Ip   | passed | 12  | 0.214994 | 1.532311 | 7.127x  | 2.71e-11 |
| PP_psin_uniform_Ip  | passed | 18  | 0.325016 | 1.927748 | 5.931x  | 2.08e-11 |
| PI_rho_uniform_Ip   | passed | 12  | 0.193432 | 1.398809 | 7.232x  | 2.90e-11 |
| PI_psin_uniform_Ip  | passed | 18  | 0.332388 | 2.002928 | 6.026x  | 4.71e-11 |
| PJ1_rho_uniform_Ip  | passed | 12  | 0.192612 | 1.342179 | 6.968x  | 2.47e-11 |
| PJ1_psin_uniform_Ip | passed | 18  | 0.262352 | 1.699648 | 6.479x  | 2.09e-11 |
| PJ2_rho_uniform_Ip  | passed | 17  | 0.410881 | 2.235011 | 5.440x  | 2.08e-11 |
| PJ2_psin_uniform_Ip | passed | 17  | 0.715743 | 3.497273 | 4.886x  | 1.90e-11 |
| PQ_rho_uniform_Ip   | passed | 12  | 0.272039 | 1.638940 | 6.025x  | 2.17e-11 |
| PQ_psin_uniform_Ip  | passed | 18  | 0.395121 | 2.053765 | 5.198x  | 2.01e-11 |

### Cxx GEQDSK Low/Medium/High/Ref

- build: `fastmath`; warmup: `5`; repeat: `100`; validation atol: `1e-06`.
- timing source: `SolveResult.elapsed_ms`; Cxx policy: `initial=cold`, `continue=cold`, `norm=fast`.
- `diff` is `x_max_abs` between Cxx and Numba packed solutions.

| case    | status | config | x   | Cxx ms    | Numba ms  | speedup | diff     |
| ------- | ------ | ------ | --- | --------- | --------- | ------- | -------- |
| solovev | passed | Low    | 4   | 0.184037  | 1.863800  | 10.127x | 1.17e-12 |
| solovev | passed | Medium | 5   | 0.190765  | 2.003676  | 10.503x | 3.14e-12 |
| solovev | passed | High   | 9   | 0.226169  | 2.322533  | 10.269x | 8.04e-12 |
| solovev | passed | Ref    | 75  | 1.408666  | 7.443985  | 5.284x  | 1.48e-10 |
| chease  | passed | Low    | 27  | 0.783436  | 5.688168  | 7.261x  | 3.12e-11 |
| chease  | passed | Medium | 36  | 0.947667  | 6.877491  | 7.257x  | 2.75e-11 |
| chease  | passed | High   | 60  | 2.232203  | 13.545717 | 6.068x  | 4.45e-08 |
| chease  | passed | Ref    | 130 | 12.777493 | 41.331305 | 3.235x  | 2.55e-09 |
| efit    | passed | Low    | 19  | 0.394639  | 3.667276  | 9.293x  | 1.55e-11 |
| efit    | passed | Medium | 29  | 0.588495  | 4.429315  | 7.527x  | 3.92e-11 |
| efit    | passed | High   | 94  | 2.789353  | 10.951346 | 3.926x  | 8.34e-11 |
| efit    | passed | Ref    | 130 | 6.321108  | 22.125056 | 3.500x  | 2.99e-10 |

### Cxx Continuation Effective Nfev

- cases: `chease, efit, solovev`; configs: `Ref`; updates: `ip, boundary, source, mixed`; points: `11`; warmup: `1`; repeat: `5`.
- Policy columns are mean `effective_nfev`; `vs cold` is the evaluation-count reduction of the best policy relative to cold.

| experiment  | case    | cold | warm-fixed | warm-predict | warm-chord | best         | vs cold |
| ----------- | ------- | ---- | ---------- | ------------ | ---------- | ------------ | ------- |
| C1 Ip       | solovev | 1085 | 109        | 109          | 109        | warm-fixed   | 9.95x   |
| C1 Ip       | chease  | 6248 | 578        | 578          | 578        | warm-fixed   | 10.81x  |
| C1 Ip       | efit    | 3168 | 298        | 298          | 298        | warm-fixed   | 10.63x  |
| C2 boundary | solovev | 1089 | 899        | 197          | 194        | warm-chord   | 5.61x   |
| C2 boundary | chease  | 7295 | 2197       | 982          | 1109       | warm-predict | 7.43x   |
| C2 boundary | efit    | 3147 | 1643       | 437          | 433        | warm-chord   | 7.27x   |
| C3 source   | solovev | 1089 | 899        | 197          | 194        | warm-chord   | 5.61x   |
| C3 source   | chease  | 6230 | 1920       | 714          | 710        | warm-chord   | 8.77x   |
| C3 source   | efit    | 3013 | 1632       | 435          | 432        | warm-chord   | 6.97x   |
| C4 mixed    | solovev | 1088 | 899        | 197          | 194        | warm-chord   | 5.61x   |
| C4 mixed    | chease  | 6501 | 1934       | 719          | 714        | warm-chord   | 9.11x   |
| C4 mixed    | efit    | 3132 | 1642       | 436          | 432        | warm-chord   | 7.25x   |

### Boundary QR Fitter Comparison

- cases: `solovev, chease, efit`; order: `10/10`; warmup: `5`; repeat: `100`.
- timing source: wall time around one scatter-to-coefficient fit call.
- `coeff diff` is the max absolute parameter/coefficient difference versus the NumPy QR baseline.
- The Cxx fitter is a standalone native module cached under `.veqpy-kernel-cache/fastmath/_boundary_fit`; its identity is independent of Kernel topology, source route, and solver config.

| case    | backend | points | median ms | fit rms  | curve    | coeff diff |
| ------- | ------- | ------ | --------- | -------- | -------- | ---------- |
| solovev | numpy   | 512    | 7.839196  | 5.37e-04 | 1.02e-02 | 0.00e+00   |
| solovev | numba   | 512    | 0.783640  | 5.37e-04 | 1.02e-02 | 1.11e-16   |
| solovev | cxx     | 512    | 0.190371  | 5.37e-04 | 1.02e-02 | 2.59e-12   |
| chease  | numpy   | 300    | 2.640611  | 3.54e-03 | 1.27e-02 | 0.00e+00   |
| chease  | numba   | 300    | 0.324443  | 3.54e-03 | 1.27e-02 | 1.39e-16   |
| chease  | cxx     | 300    | 0.099039  | 3.54e-03 | 1.27e-02 | 3.70e-12   |
| efit    | numpy   | 106    | 0.726909  | 1.36e-03 | 9.81e-03 | 0.00e+00   |
| efit    | numba   | 106    | 0.075393  | 1.36e-03 | 9.81e-03 | 2.22e-16   |
| efit    | cxx     | 106    | 0.030602  | 1.36e-03 | 9.81e-03 | 3.43e-12   |

## Notes

- The default route and GEQDSK tables are cold-repeat Kernel benchmarks.
- Warm-start behavior is isolated in the continuation benchmark.
- Cxx tables exclude first-run C++/nanobind build cost; inspect JSON `artifact.*` fields for build timing.
- This WSL2 run records solve-time medians only; it does not claim PMU, cache, IPC, or Roofline evidence.
- Regenerate this file after changing Kernel API benchmark entry points, solver
  policy, route topology, profile layout, compiler flags, or hardware.
