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

## Follow-up A/B: geometry surface layout

Change tested: geometry surface storage order changed from logical field planes
`[field][rho][theta]` to physical `[rho][field][theta]`. The public accessor
`surface_field(row, radial_node, theta_node)` preserves logical read/write
semantics; `theta` remains contiguous.

Paired local wall-clock timing (`repeat=30`, `warmup=8`, `inner=10000`,
3 rounds; compare same time window, not absolute ns):

| metric | baseline | candidate | candidate / baseline |
| --- | ---: | ---: | ---: |
| `geometry` median-of-medians ns/call | 9476.8 | 5457.7 | 0.576 |
| `evaluate` median-of-medians ns/call | 13149.6 | 8796.7 | 0.669 |
| `geometry / evaluate` share | 0.721 | 0.620 | - |

Interpretation: the layout hypothesis is strongly supported in this workstation
run. Because PMU counters are unavailable under WSL2, treat the exact cache-set
mechanism as still requiring hardware-counter confirmation, but keep the layout
change unless a native-PMU run contradicts the wall-clock A/B.

## Layout retest after commit

The committed layout change (`a5c4d3c`) was retested against the pre-layout
baseline commit (`7566f97`) from an independent worktree. Paired local
wall-clock timing used 5 rounds, `repeat=30`, `warmup=8`, `inner=10000`, and
`taskset -c 2` when available.

| metric | pre-layout baseline | committed layout | layout / baseline |
| --- | ---: | ---: | ---: |
| `geometry` median-of-medians ns/call | 9297.0 | 5378.6 | 0.579 |
| `evaluate` median-of-medians ns/call | 12879.2 | 8883.5 | 0.687 |
| `geometry / evaluate` share | 0.723 | 0.608 | - |

Decision: the layout change remains effective under a fresh paired run and
should stay as the new optimization baseline.

## Follow-up candidates tested and rejected

After the layout retest, several smaller candidates were tested against
`a5c4d3c`. They were reverted because they did not produce a stable end-to-end
`evaluate` improvement.

| candidate | key stage ratio | `evaluate` ratio | decision |
| --- | ---: | ---: | --- |
| Skip absent Fourier orders in `GeometryRuntime::update` | `geometry` 0.988 | 1.001 | reject; stage-only clue below decision threshold |
| Hoist repeated residual geometry loads | `residual_update` 0.974 | 0.996 | reject; end-to-end effect too small |
| Explicit glibc `sincos` path | `geometry` 2.386 | 1.835 | reject; severe regression |
| Residual surface physical layout `[rho][field][theta]` | `residual_update` 0.927 | 0.991 | reject; noisy end-to-end gain and `residual_pack` regression |
| Source `psin_r` regularization/pass reduction | `source_update` 0.954 | 1.000 | reject; no end-to-end gain |
| Geometry hot-loop pointer/index flattening | `geometry` 1.003 | 1.000 | reject; compiler already removes most accessor overhead |

Next optimization work should not repeat these micro-candidates. Prefer either a
larger geometry math-path experiment with a dedicated correctness comparator, or
native-Linux PMU validation before spending more time on cache-mechanism claims.

## Planning update after 3851f62 review

The post-review priority is adjusted as follows:

1. Treat the geometry layout change as accepted. PMU is useful to validate the
   cache/conflict mechanism, not to decide whether the wall-clock improvement is
   real.
2. Promote FP build semantics to P0. Split strict/FMA/relaxed modes before
   testing vector sincos or approximate math backends, and separate true
   finite-check semantics from magnitude-validity checks.
3. Fill the post-layout stage table before choosing the next absolute hot spot.
   Current formal retest only covers `geometry`, `evaluate`, and their share.
4. Extend the benchmark beyond the default resonance-prone topology
   (`Nr=32`, `Nt=16`, `Mmax=1`) and add a solver-state ring in addition to the
   same-x warm benchmark.
5. Decompose geometry into phase synthesis, dynamic sincos, metric arithmetic,
   and output traffic micro-stages. Fixed theta/harmonic trig tables are already
   setup-time data; the relevant target is dynamic `sin(tb)` / `cos(tb)`.
6. Move residual fusion earlier. The residual-layout A/B indicates conflicting
   producer/consumer locality, so the next structural direction is theta-moment
   fusion rather than another materialized surface layout.
7. Keep PMU/native-Linux work as a parallel mechanism-validation line while the
   main optimization loop continues with paired wall-clock, correctness checks,
   assembly, and Cachegrind.

## Phase 0 update: FP modes and validity semantics

Phase 0 is now implemented before further approximate/vector-math experiments:

- Added `VEQLIB_FP_MODE=STRICT|FMA|RELAXED`.
  - `STRICT`: `-fno-fast-math -ffp-contract=off`.
  - `FMA`: `-fno-fast-math -ffp-contract=fast`.
  - `RELAXED`: preserves the historical release benchmark flags, including
    `-ffast-math`, `-ffinite-math-only`, `-freciprocal-math`, and
    `-fapprox-func`.
- Added `clang-release-strict` and `clang-release-fma` presets. The default
  `clang-release` remains `RELAXED` so existing performance comparisons keep
  the same baseline semantics.
- Split validity helpers:
  - `math::is_finite()` is now a bit-level NaN/inf test that is still
    meaningful under `-ffinite-math-only`.
  - `math::is_valid_magnitude()` preserves the previous `max_double / 4`
    overflow margin and is used by source/operator hot-path acceptance checks.

Validation:

- Compile-command inspection confirmed the expected FP flags for `release`,
  `release-strict`, `release-fma`, and `debug`.
- Debug CTest: 3/3 passed.
- RELAXED release CTest: 3/3 passed.
- STRICT release CTest: 3/3 passed.
- FMA release CTest: 3/3 passed.
- RELAXED release Python/C++ PF-psin-uniform validation passed with
  `max_abs=6.7622192567728945e-12`.
- RELAXED release sanity benchmark (`taskset -c 2`, `repeat=15`, `warmup=5`,
  `inner=10000`) reported `geometry median=5554.3095 ns/call` and
  `evaluate median=9109.3373 ns/call`. This is not a paired optimization proof;
  it only confirms Phase 0 did not visibly damage the current layout baseline.

Next step: Phase 1 should generate a post-layout stage table across the new
FP modes and then add topology/state-ring coverage before any approximate
dynamic `sincos(tb)` backend is compared.
