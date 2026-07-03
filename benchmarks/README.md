# Benchmarks

This directory retains VEQPy/VEQlib benchmark result artifacts and environment
metadata captured before the Kernel API became the sole public runtime entry.
New benchmark entrypoints should be built directly on `veqpy.facade.Kernel` and,
when native C++ comparison is needed, `veqlib.facade`.

Runtime tables report median solve/runtime time only. VEQlib artifact build time is not included in `Cxx ms`; full build metadata and samples are in the JSON outputs.

## Environment

- CPU: `Intel(R) Core(TM) i5-14600KF`
- Python: `python 3.12.3 [GCC 13.3.0]`
- CMake: `cmake version 3.28.3`
- C++: `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- Thread env: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

## Results

### VEQPy synthetic route matrix

- scope: `ip-uniform`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.

| case                | status | x   | Numba ms | nfev | residual | shape    | psi_r RMS | FF_psi RMS |
| ------------------- | ------ | --- | -------- | ---- | -------- | -------- | --------- | ---------- |
| PF_rho_uniform_Ip   | passed | 12  | 0.686798 | 29   | 4.06e-10 | 1.66e-04 | 8.53e-05  | 2.51e-03   |
| PF_psin_uniform_Ip  | passed | 18  | 0.980646 | 40   | 9.81e-09 | 2.95e-03 | 1.43e-03  | 3.84e-03   |
| PP_rho_uniform_Ip   | passed | 12  | 0.808259 | 39   | 3.83e-11 | 2.05e-04 | 4.94e-05  | 4.21e-03   |
| PP_psin_uniform_Ip  | passed | 18  | 1.184529 | 52   | 5.56e-10 | 6.23e-03 | 2.61e-03  | 2.40e-02   |
| PI_rho_uniform_Ip   | passed | 12  | 0.630345 | 28   | 7.50e-08 | 1.52e-04 | 3.86e-05  | 1.38e-03   |
| PI_psin_uniform_Ip  | passed | 18  | 1.276317 | 55   | 2.87e-12 | 1.91e-03 | 6.64e-04  | 9.37e-03   |
| PJ1_rho_uniform_Ip  | passed | 12  | 0.650989 | 29   | 1.11e-09 | 1.68e-04 | 4.40e-05  | 3.39e-04   |
| PJ1_psin_uniform_Ip | passed | 18  | 0.987413 | 40   | 8.15e-09 | 3.13e-03 | 1.52e-03  | 9.67e-03   |
| PJ2_rho_uniform_Ip  | passed | 18  | 1.630445 | 82   | 7.99e-09 | 1.26e-04 | 4.30e-05  | 8.81e-04   |
| PJ2_psin_uniform_Ip | passed | 18  | 1.995853 | 66   | 2.76e-09 | 3.50e-03 | 1.80e-03  | 1.18e-02   |
| PQ_rho_uniform_Ip   | passed | 12  | 0.859657 | 29   | 6.60e-08 | 1.35e-04 | 2.81e-05  | 1.34e-03   |
| PQ_psin_uniform_Ip  | passed | 18  | 1.310679 | 42   | 2.63e-09 | 5.00e-03 | 2.56e-03  | 2.18e-02   |

### VEQPy GEQDSK route matrix

- scope: `ip-uniform`; GEQDSK: `EFIT.geqdsk`; warmup: `5`; repeat: `1`; summary: `9/12 passed`.
- Failed rows are tolerance failures; timing and diagnostics are still written to JSON: `PP_psin_uniform_Ip`, `PJ2_rho_uniform_Ip`, `PJ2_psin_uniform_Ip`.

| case                | status | x   | Numba ms  | nfev | residual | shape    | psi_r RMS | FF_psi RMS |
| ------------------- | ------ | --- | --------- | ---- | -------- | -------- | --------- | ---------- |
| PF_rho_uniform_Ip   | passed | 120 | 35.853889 | 4    | 8.82e-16 | 2.48e-03 | 5.49e-04  | 1.98e-02   |
| PF_psin_uniform_Ip  | passed | 130 | 42.631155 | 4    | 7.35e-16 | 4.05e-03 | 1.07e-03  | 1.16e-03   |
| PP_rho_uniform_Ip   | passed | 120 | 37.356181 | 4    | 6.36e-16 | 7.12e-03 | 6.00e-04  | 3.28e-03   |
| PP_psin_uniform_Ip  | failed | 130 | 58.602281 | 5    | 7.51e-16 | 1.16e-02 | 2.99e-03  | 5.42e-03   |
| PI_rho_uniform_Ip   | passed | 120 | 33.524309 | 4    | 7.31e-16 | 6.12e-03 | 8.45e-04  | 6.86e-03   |
| PI_psin_uniform_Ip  | passed | 130 | 40.176811 | 4    | 6.21e-16 | 2.84e-03 | 1.87e-03  | 5.52e-03   |
| PJ1_rho_uniform_Ip  | passed | 120 | 33.791879 | 4    | 6.48e-16 | 2.28e-03 | 5.18e-04  | 2.49e-03   |
| PJ1_psin_uniform_Ip | passed | 130 | 38.939923 | 4    | 4.48e-16 | 3.95e-03 | 1.59e-03  | 1.75e-03   |
| PJ2_rho_uniform_Ip  | failed | 125 | 43.598342 | 5    | 1.14e-14 | 4.57e-01 | 3.81e-02  | 3.29e-02   |
| PJ2_psin_uniform_Ip | failed | 125 | 66.583221 | 7    | 4.02e-11 | 2.29e-01 | 2.16e-02  | 2.07e-02   |
| PQ_rho_uniform_Ip   | passed | 120 | 40.578132 | 4    | 5.18e-14 | 2.90e-03 | 1.05e-03  | 4.18e-03   |
| PQ_psin_uniform_Ip  | passed | 130 | 53.065577 | 5    | 5.49e-14 | 8.92e-03 | 3.73e-03  | 6.73e-03   |

### VEQlib synthetic route matrix

- scope: `ip-uniform`; build: `fastmath`; layout: `degree`; warmup: `5`; repeat: `100`; summary: `12/12 passed`.
- `diff` is `x_max_abs` between the VEQlib and VEQPy packed solutions.

| case                | status | x   | Cxx ms   | Numba ms | speedup | diff     |
| ------------------- | ------ | --- | -------- | -------- | ------- | -------- |
| PF_rho_uniform_Ip   | passed | 12  | 0.052055 | 0.640157 | 12.298x | 1.23e-09 |
| PF_psin_uniform_Ip  | passed | 18  | 0.076291 | 1.028155 | 13.477x | 3.28e-08 |
| PP_rho_uniform_Ip   | passed | 12  | 0.046632 | 0.804723 | 17.257x | 1.54e-10 |
| PP_psin_uniform_Ip  | passed | 18  | 0.074705 | 1.157306 | 15.492x | 3.11e-09 |
| PI_rho_uniform_Ip   | passed | 12  | 0.051501 | 0.621950 | 12.077x | 1.91e-07 |
| PI_psin_uniform_Ip  | passed | 18  | 0.080234 | 1.243373 | 15.497x | 5.89e-11 |
| PJ1_rho_uniform_Ip  | passed | 12  | 0.048545 | 0.651485 | 13.420x | 2.03e-09 |
| PJ1_psin_uniform_Ip | passed | 18  | 0.076897 | 0.969706 | 12.611x | 3.41e-08 |
| PJ2_rho_uniform_Ip  | passed | 17  | 0.071526 | 1.515656 | 21.190x | 4.80e-08 |
| PJ2_psin_uniform_Ip | passed | 17  | 0.114249 | 2.507372 | 21.947x | 5.49e-08 |
| PQ_rho_uniform_Ip   | passed | 12  | 0.090696 | 0.856035 | 9.438x  | 1.98e-07 |
| PQ_psin_uniform_Ip  | passed | 18  | 0.136294 | 1.349934 | 9.905x  | 5.07e-09 |

### VEQlib GEQDSK Low/Medium/High/Ref

- build: `fastmath`; warmup: `5`; repeat: `100`; validation atol: `1e-06`.
- `diff` is `x_max_abs` between the VEQlib and VEQPy packed solutions.

| case    | status | config | x   | Cxx ms   | Numba ms  | speedup | diff     |
| ------- | ------ | ------ | --- | -------- | --------- | ------- | -------- |
| solovev | passed | Low    | 4   | 0.038843 | 0.988233  | 25.442x | 2.73e-09 |
| solovev | passed | Medium | 5   | 0.044019 | 1.076983  | 24.466x | 9.20e-09 |
| solovev | passed | High   | 9   | 0.066897 | 1.384216  | 20.692x | 1.25e-08 |
| solovev | passed | Ref    | 75  | 0.778386 | 6.778595  | 8.709x  | 1.47e-08 |
| chease  | passed | Low    | 27  | 0.220681 | 5.020843  | 22.752x | 1.89e-07 |
| chease  | passed | Medium | 36  | 0.294272 | 6.077891  | 20.654x | 3.87e-08 |
| chease  | passed | High   | 60  | 0.551778 | 13.426837 | 24.334x | 3.46e-09 |
| chease  | passed | Ref    | 130 | 2.537298 | 42.276592 | 16.662x | 1.36e-07 |
| efit    | passed | Low    | 19  | 0.138631 | 2.667478  | 19.242x | 3.51e-09 |
| efit    | passed | Medium | 29  | 0.243494 | 3.624370  | 14.885x | 4.51e-08 |
| efit    | passed | High   | 94  | 1.315509 | 10.269229 | 7.806x  | 3.44e-08 |
| efit    | passed | Ref    | 130 | 2.530054 | 21.647612 | 8.556x  | 1.24e-09 |

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

- VEQlib tables exclude first-run C++/nanobind build cost; inspect JSON `artifact.*` fields for build timing.
- This WSL2 run records script wall-time medians only; it does not claim PMU, cache, IPC, or Roofline evidence.
- Regenerate this file after adding Kernel API benchmark entrypoints or changing
  solver policy, route topology, profile layout, compiler flags, or hardware.
