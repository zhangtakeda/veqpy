# Benchmarks

This directory contains VEQPy Kernel benchmark entry points and result artifacts.
Current scripts use the Kernel API directly: Python/Numba rows call
`veqpy.Kernel` with `backend="numba"`, and native C++ rows call the same
`veqpy.Kernel` surface with `backend="cxx"`.

The route and GEQDSK tables below report median `SolveResult.elapsed_ms`.
Each measured repeat uses a fresh Kernel handle for the case, so accepted
solutions are not reused between repeats. The JSON outputs also keep outer
`wall_timing` samples, which include per-repeat Kernel construction and close.
Cxx artifact build time is not included in `Cxx ms`; build metadata and
sample timings are recorded in the JSON outputs.

`veqlib_continuation.py` is the explicit continuation-policy benchmark. Its
table reports effective function evaluations rather than solve-time medians.

## Scripts

- `veqpy_routes.py`: VEQPy/Numba synthetic route matrix through `veqpy.Kernel`.
- `veqlib_routes.py`: Cxx backend route matrix compared with VEQPy/Numba Kernel.
- `veqlib_geqdsk_pareto.py`: GEQDSK Cxx backend vs VEQPy/Numba Kernel matrix.
- `veqlib_continuation.py`: Cxx backend continuation-policy benchmark.
- `_common.py`: shared Kernel-case construction, timing, route specs, and JSON helpers.

## Reproduce

```bash
.venv/bin/python benchmarks/veqpy_routes.py --quiet-progress
.venv/bin/python benchmarks/veqlib_routes.py --quiet-progress
.venv/bin/python benchmarks/veqlib_geqdsk_pareto.py --quiet-progress
.venv/bin/python benchmarks/veqlib_continuation.py --quiet-progress
```

## Environment

- CPU: `Intel(R) Core(TM) i5-14600KF`
- Python: `python 3.12.3 [GCC 13.3.0]`
- CMake: `cmake version 3.28.3`
- C++: `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- Thread env: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

## Results

### VEQPy synthetic route matrix

- scope: `ip-uniform`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.
- timing source: `SolveResult.elapsed_ms`; default initial policy: `cold`.

| case                | status | x   | Numba ms | nfev | residual | shape    | psi_r RMS | FF_psi RMS |
| ------------------- | ------ | --- | -------- | ---- | -------- | -------- | --------- | ---------- |
| PF_rho_uniform_Ip   | passed | 12  | 0.828178 | 29   | 4.06e-10 | 1.66e-04 | 8.53e-05  | 2.51e-03   |
| PF_psin_uniform_Ip  | passed | 18  | 1.117537 | 40   | 9.81e-09 | 2.95e-03 | 1.43e-03  | 3.84e-03   |
| PP_rho_uniform_Ip   | passed | 12  | 0.927659 | 39   | 3.83e-11 | 2.05e-04 | 4.94e-05  | 4.21e-03   |
| PP_psin_uniform_Ip  | passed | 18  | 1.303901 | 52   | 5.56e-10 | 6.23e-03 | 2.61e-03  | 2.40e-02   |
| PI_rho_uniform_Ip   | passed | 12  | 0.755546 | 28   | 7.50e-08 | 1.52e-04 | 3.86e-05  | 1.38e-03   |
| PI_psin_uniform_Ip  | passed | 18  | 1.364899 | 55   | 2.87e-12 | 1.91e-03 | 6.64e-04  | 9.37e-03   |
| PJ1_rho_uniform_Ip  | passed | 12  | 0.725092 | 29   | 1.11e-09 | 1.68e-04 | 4.40e-05  | 3.39e-04   |
| PJ1_psin_uniform_Ip | passed | 18  | 1.088234 | 40   | 8.15e-09 | 3.13e-03 | 1.52e-03  | 9.67e-03   |
| PJ2_rho_uniform_Ip  | passed | 18  | 1.661757 | 82   | 7.99e-09 | 1.26e-04 | 4.30e-05  | 8.81e-04   |
| PJ2_psin_uniform_Ip | passed | 18  | 2.043581 | 66   | 2.97e-09 | 3.50e-03 | 1.80e-03  | 1.18e-02   |
| PQ_rho_uniform_Ip   | passed | 12  | 0.933022 | 29   | 6.60e-08 | 1.35e-04 | 2.81e-05  | 1.34e-03   |
| PQ_psin_uniform_Ip  | passed | 18  | 1.409089 | 42   | 2.64e-09 | 5.00e-03 | 2.56e-03  | 2.18e-02   |

### VEQlib synthetic route matrix

- scope: `ip-uniform`; build: `fastmath`; layout: `degree`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.
- timing source: `SolveResult.elapsed_ms`; native policy: `initial=cold`, `continue=cold`, `norm=fast`.
- `diff` is `x_max_abs` between the VEQlib and VEQPy packed solutions.

| case                | status | x   | Cxx ms   | Numba ms | speedup | diff     |
| ------------------- | ------ | --- | -------- | -------- | ------- | -------- |
| PF_rho_uniform_Ip   | passed | 12  | 0.119175 | 0.758896 | 6.368x  | 2.47e-11 |
| PF_psin_uniform_Ip  | passed | 18  | 0.180541 | 1.020400 | 5.652x  | 2.05e-11 |
| PP_rho_uniform_Ip   | passed | 12  | 0.135385 | 0.843838 | 6.233x  | 2.71e-11 |
| PP_psin_uniform_Ip  | passed | 18  | 0.211416 | 1.258294 | 5.952x  | 2.08e-11 |
| PI_rho_uniform_Ip   | passed | 12  | 0.109826 | 0.698312 | 6.358x  | 2.90e-11 |
| PI_psin_uniform_Ip  | passed | 18  | 0.243136 | 1.367347 | 5.624x  | 4.71e-11 |
| PJ1_rho_uniform_Ip  | passed | 12  | 0.111225 | 0.769978 | 6.923x  | 2.47e-11 |
| PJ1_psin_uniform_Ip | passed | 18  | 0.190843 | 1.060848 | 5.559x  | 2.09e-11 |
| PJ2_rho_uniform_Ip  | passed | 17  | 0.314262 | 1.659362 | 5.280x  | 2.08e-11 |
| PJ2_psin_uniform_Ip | passed | 17  | 0.583613 | 2.596334 | 4.449x  | 1.90e-11 |
| PQ_rho_uniform_Ip   | passed | 12  | 0.188252 | 0.966788 | 5.136x  | 2.15e-11 |
| PQ_psin_uniform_Ip  | passed | 18  | 0.303669 | 1.437389 | 4.733x  | 2.07e-11 |

### VEQlib GEQDSK Low/Medium/High/Ref

- build: `fastmath`; warmup: `5`; repeat: `100`; validation atol: `1e-06`.
- timing source: `SolveResult.elapsed_ms`; native policy: `initial=cold`, `continue=cold`, `norm=fast`.
- `diff` is `x_max_abs` between the VEQlib and VEQPy packed solutions.

| case    | status | config | x   | Cxx ms    | Numba ms  | speedup | diff     |
| ------- | ------ | ------ | --- | --------- | --------- | ------- | -------- |
| solovev | passed | Low    | 4   | 0.077366  | 1.095349  | 14.158x | 1.17e-12 |
| solovev | passed | Medium | 5   | 0.089723  | 1.202812  | 13.406x | 3.14e-12 |
| solovev | passed | High   | 9   | 0.129603  | 1.515981  | 11.697x | 8.04e-12 |
| solovev | passed | Ref    | 75  | 1.218136  | 6.718363  | 5.515x  | 1.48e-10 |
| chease  | passed | Low    | 27  | 0.615500  | 4.967208  | 8.070x  | 3.12e-11 |
| chease  | passed | Medium | 36  | 0.783802  | 6.142815  | 7.837x  | 2.76e-11 |
| chease  | passed | High   | 60  | 2.053660  | 13.310995 | 6.482x  | 1.46e-08 |
| chease  | passed | Ref    | 130 | 12.440129 | 41.164805 | 3.309x  | 6.11e-09 |
| efit    | passed | Low    | 19  | 0.263450  | 2.660089  | 10.097x | 1.55e-11 |
| efit    | passed | Medium | 29  | 0.443195  | 3.635614  | 8.203x  | 3.92e-11 |
| efit    | passed | High   | 94  | 2.502630  | 9.804938  | 3.918x  | 8.34e-11 |
| efit    | passed | Ref    | 130 | 6.209815  | 21.124163 | 3.402x  | 2.99e-10 |

### VEQlib continuation effective nfev

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

## Notes

- The default route and GEQDSK tables are cold-repeat Kernel benchmarks.
- Warm-start behavior is isolated in the continuation benchmark.
- VEQlib tables exclude first-run C++/nanobind build cost; inspect JSON `artifact.*` fields for build timing.
- This WSL2 run records solve-time medians only; it does not claim PMU, cache, IPC, or Roofline evidence.
- Regenerate this file after changing Kernel API benchmark entry points, solver
  policy, route topology, profile layout, compiler flags, or hardware.
