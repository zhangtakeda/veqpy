# VEQlib experiment summary

## Environment / tool limits

- Host reported WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`.
- `/usr/lib/linux-tools/6.8.0-124-generic/perf` runs, but hardware events report `<not supported>`.
- LIKWID topology works, but `likwid-perfctr` reports unsupported processor / MSR access failure.
- Therefore first-round evidence uses stage wall-clock timing, Clang remarks/objdump, and Cachegrind deterministic references.

## Correctness baseline

- Debug CTest: 3/3 passed (`veqlib/experiments/baseline/ctest-debug-after-review-fixes.log`).
- Python/C++ PF-psin-uniform validation: passed=True, max_abs=6.762e-12.

## Stable release stage timing

| stage | median ns/call | avg ns/call | p95/median | CV |
| --- | ---: | ---: | ---: | ---: |
| `profiles_all` | 133.7 | 134.7 | 1.082 | 0.049 |
| `geometry` | 9330.3 | 9416.4 | 1.045 | 0.036 |
| `source_materialize` | 975.4 | 982.9 | 1.034 | 0.036 |
| `source_update` | 922.4 | 922.7 | 1.030 | 0.023 |
| `residual_update` | 1073.5 | 1072.1 | 1.028 | 0.018 |
| `residual_pack` | 144.6 | 145.1 | 1.053 | 0.031 |
| `evaluate` | 12917.5 | 12919.0 | 1.034 | 0.016 |

- Geometry share of full `evaluate`: 72.2% by median wall time.
- Very small stages such as `profiles_fixed` are below reliable timer granularity and are omitted from the decision table.
- Timing source files are `veqlib/experiments/perf/stage_release_*_after_review_fixes.json`.

## Cachegrind per-call references (inner=10000)

| stage | I refs | D refs | D1 misses | branches | branch mispredicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `geometry` | 162169.5 | 63068.9 | 260.7 | 6297.3 | 669.4 |
| `residual_update` | 24080.2 | 7674.6 | 871.7 | 907.0 | 45.5 |
| `evaluate` | 203750.3 | 87605.7 | 2296.6 | 7634.2 | 723.4 |

## Tooling evidence

- Release dynamic symbols include `sincos@GLIBC_2.2.5`; objdump has a direct `sincos@plt` call in the stage binary.
- Clang analysis build emitted vectorized residual loops at `residual.h:275` and `residual.h:292` with vectorization width 4.
- Clang analysis build emitted no final `-Wframe-larger-than` / `-Wlarge-by-value-copy` warnings after moving the benchmark operator off the stack.

## Decision

- First optimization target should be geometry (`sin/cos` path and 9-plane update), not solver dispatch.
- Residual materialization remains structurally suspicious: it is a smaller wall-time slice than geometry, but Cachegrind reports high D1 activity in `residual_update`; fusion should be tested after geometry baseline improvements or as a separate A/B.
- Direct PMU validation of the 4 KiB plane conflict is blocked on this WSL environment; use a native Linux boot or a perf-capable VM for final L1 replacement evidence.
