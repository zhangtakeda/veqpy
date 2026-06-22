# VEQlib experiment summary

## Environment / tool limits

- Host reported WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`.
- `/usr/lib/linux-tools/6.8.0-124-generic/perf` runs, but hardware events report `<not supported>`.
- LIKWID topology works, but `likwid-perfctr` reports unsupported processor / MSR access failure.
- Therefore first-round evidence uses stage wall-clock timing, Clang remarks/objdump, and Cachegrind deterministic references.

## Correctness baseline

- Debug CTest: 3/3 passed (`veqlib/experiments/baseline/ctest-debug-after-review-fixes.log`).
- Python/C++ PF-psin-uniform validation: passed=True, max_abs=6.762e-12.

## 2026-06-22 P0 reproducible baseline

Baseline run directory: `veqlib/experiments/7e4229d-20260622-p0-baseline`.
The run uses commit `7e4229dcd044b43c4414cb793826c29a705bf98b`; both
`worktree.diff` and `index.diff` are empty in the saved manifest. Release and
debug CTest both passed 4/4, including the nanobind binding smoke test. The
RELAXED Python/C++ comparator passed with `max_abs=7.66e-10` under the existing
`1e-9` tolerance.

Pinned RELAXED timing used `taskset -c 2`, `repeat=30`, `warmup=5`, and
`ring-size=16`; stage timing used `inner=3000`.

| stage | median ns/call | p95 ns/call |
| --- | ---: | ---: |
| `profiles_all` | 123.3 | 156.6 |
| `geometry` | 1753.0 | 1916.7 |
| `source_materialize` | 797.0 | 863.9 |
| `source_update` | 781.6 | 945.0 |
| `residual_update` | 777.3 | 858.7 |
| `residual_pack` | 132.5 | 187.7 |
| `evaluate` | 4655.1 | 5168.6 |
| `evaluate_ring` | 4696.0 | 5345.8 |

Pinned residual-only C++ solve timing reported median `0.206 ms`, p95
`0.264 ms`, `nfev=38`, `callback_evaluations=38`, and
`raw_norm=9.81e-09`. The nanobind comparison reported Python median
`1.086 ms`, Python-perceived C++ median `0.219 ms`, direct inner C++ median
`0.215 ms`, and interface median overhead `0.0023 ms`; median speedup is
`4.97x` over the Python reference. Treat these numbers as the P0 baseline for
the next Source/Residual/Solver substage split. The current WSL2 environment
still has no usable hardware-PMU path, so PMU remains a later native-Linux
mechanism check rather than a P0/P1 gate.

## 2026-06-22 P0-B Source/Residual substage probes

Benchmark-only Source and Residual fine stages were added without changing the
production `evaluate()` sequence. Source wrappers expose individual current-path
operations such as psin copy/regularization, D/A fixed-size matvecs,
interpolation, integrand fill, normalization, and alpha update. Residual wrappers
split the materialized-surface pack path into moment-row theta reduction and
radial projection from those materialized moments. The residual split is a probe
shape, not an exact decomposition of current `residual_pack`, because it writes a
moment-row buffer so projection can be timed separately.

Validation after adding the probes:

- Release CTest: 4/4 passed.
- Debug CTest: 4/4 passed.
- RELAXED Python/C++ comparator: passed with `max_abs=7.66e-10`.

Pinned stage timing (`taskset -c 2`, `repeat=30`, `warmup=5`, `inner=3000`) was
saved under `veqlib/experiments/7e4229d-20260622-p0-substages`.

| probe | median ns/call | p95 ns/call |
| --- | ---: | ---: |
| `source_materialize` | 800.3 | 968.5 |
| `source_copy_regularize` | 9.7 | 9.9 |
| `source_D_psin` | 249.5 | 549.1 |
| `source_A_psin` | 243.6 | 330.6 |
| `source_interpolate_pair` | 212.4 | 239.7 |
| `source_integrand` | 4.0 | 4.4 |
| `source_A_integrand` | 240.6 | 258.3 |
| `source_normalize` | 39.5 | 42.5 |
| `source_D_normalized` | 242.2 | 318.4 |
| `source_alpha` | 5.8 | 5.9 |
| `source_update` | 831.6 | 1176.1 |
| `residual_update` | 794.6 | 884.8 |
| `residual_theta_reduce` | 359.2 | 493.1 |
| `residual_radial_project` | 35.1 | 37.6 |
| `residual_pack` | 132.5 | 163.0 |
| `evaluate` | 4578.5 | 4823.6 |
| `evaluate_ring` | 4812.8 | 5080.4 |

Immediate interpretation: Source cost is dominated by the repeated fixed-size
D/A matvec shape plus interpolation, not by scalar integrand or alpha arithmetic.
The residual materialized-moment probe shows that a future `MomentPlan` must
avoid merely adding a moment buffer around the current scalar reductions; the
next viable residual candidate should batch/merge moments with a vector-friendly
theta loop before claiming projection savings.

## 2026-06-22 P0-C KernelPlan / KernelWorkspace split

`PfPsinUniformOperator` now has an explicit read-mostly `KernelPlan` and mutable
`KernelWorkspace`. The plan owns fixed profile rows and the `fix_rho` axis-count
for the current runtime parameters; the workspace owns active/fixed profiles,
geometry, source, and residual buffers. `evaluate()` lazily prepares the plan on
first use and no longer rebuilds fixed profiles or recomputes the axis count on
each callback. `refresh_static_plan()` is the explicit boundary to call after
changing fixed runtime parameters on a reused operator.

Validation after the split:

- Release CTest: 4/4 passed.
- Debug CTest: 4/4 passed.
- RELAXED Python/C++ comparator: passed with `max_abs=7.66e-10`.

Pinned P0-C timing used `taskset -c 2`, `repeat=30`, `warmup=5`; stage timing
used `inner=1000` and was saved under
`veqlib/experiments/0d68ce3-20260622-p0-kernelplan`.

| stage / metric | P0-B median | P0-C median | ratio |
| --- | ---: | ---: | ---: |
| `profiles_all` | 124.0 ns | 130.6 ns | 1.053 |
| `source_materialize` | 800.3 ns | 747.3 ns | 0.934 |
| `source_update` | 831.6 ns | 798.0 ns | 0.960 |
| `residual_update` | 794.6 ns | 790.5 ns | 0.995 |
| `residual_pack` | 132.5 ns | 129.5 ns | 0.978 |
| `evaluate` | 4578.5 ns | 4367.2 ns | 0.954 |
| `evaluate_ring` | 4812.8 ns | 4199.9 ns | 0.873 |
| solve median | 0.206 ms | 0.188 ms | 0.908 |

Interpretation: the minimal plan/workspace split satisfies the P0-C retention
gate in this WSL2 run: correctness is unchanged, callback-level `evaluate` and
state-ring `evaluate_ring` do not regress, and the full residual-only solve
median improved. `profiles_all` now includes explicit static-plan refresh plus
fixed-row seeding and is not the callback gate; it remains close enough to keep
the boundary because static refresh is outside repeated solver callbacks.

## 2026-06-22 P0-D Solver callback timing split

Solver benchmark JSON now preserves the existing counters while adding
`final.callback_timing_ms`. The new fields separate scaled residual callback
time into total callback, raw residual kernel, and residual-scaling portions;
the post-solve diagnostic residual is reported separately as `final_residual`.
Jacobian, JVP, and linear-solver timing fields are present for non-residual-only
solver paths and remain zero for the residual-only CMINPACK `hybrd` baseline.

Validation after adding solver timing:

- Release CTest: 4/4 passed.
- Debug CTest: 4/4 passed.
- RELAXED Python/C++ comparator: passed with `max_abs=7.66e-10`.

Pinned residual-only solve timing artifact:
`veqlib/experiments/c5dccbe-20260622-p0-solver-timing`.

| field | value |
| --- | ---: |
| solve median | 0.209 ms |
| solve p95 | 0.249 ms |
| `nfev` / callbacks | 38 / 38 |
| residual callback total | 0.182 ms |
| residual raw-kernel total | 0.179 ms |
| residual scale total | 0.001 ms |
| final diagnostic residual | 0.004 ms |

Interpretation: for the current residual-only path, almost all callback time is
inside the route raw-residual kernel, not z/x scaling or residual normalization.
Because the timing probes use `std::chrono` inside callbacks, these artifacts
are diagnostic attribution evidence; use the P0-C uninstrumented solve/stage
samples for cleaner absolute performance comparisons.

## 2026-06-22 P1-A Source dual matvec candidate

Source now has a `dual_matvec_into()` helper for the two places where the same
`psin_r` vector was immediately multiplied by both the differentiator and
accumulator. The production Source path uses it in profile-owned psin
materialization and in normalized PF/IP source update. Existing single-output
`matvec_into()` and benchmark probes remain available; interpolation and
residual formulas are unchanged.

Validation after the candidate:

- Release CTest: 4/4 passed.
- Debug CTest: 4/4 passed.
- RELAXED Python/C++ comparator: passed with `max_abs=7.66e-10`.

Pinned candidate artifacts:
`veqlib/experiments/0601ecf-20260622-p1-source-dual-matvec`.

| stage / metric | P0-C median | P1-A median | ratio |
| --- | ---: | ---: | ---: |
| `source_materialize` | 747.3 ns | 650.9 ns | 0.871 |
| `source_D_psin` | 240.5 ns | 239.5 ns | 0.996 |
| `source_A_psin` | 238.3 ns | 238.4 ns | 1.001 |
| `source_interpolate_pair` | 210.2 ns | 194.1 ns | 0.923 |
| `source_update` | 798.0 ns | 652.0 ns | 0.817 |
| `source_D_normalized` | 242.1 ns | 234.4 ns | 0.968 |
| `residual_update` | 790.5 ns | 710.4 ns | 0.899 |
| `evaluate` | 4367.2 ns | 3969.0 ns | 0.909 |
| `evaluate_ring` | 4199.9 ns | 3988.0 ns | 0.950 |
| solve median | instrumented | 0.186 ms | success |

Decision: keep the dual D/A row-dot shape. It clears the source and end-to-end
retention gate without changing route semantics. The individual single-output
`source_D_*`/`source_A_*` probes remain near neutral because they still time the
old single-output helpers; the retained benefit is from avoiding two consecutive
passes over the same `psin_r` vector in the production Source sequence.

### P1-A representative topology check

A temporary worktree at baseline `0601ecf` was compared against candidate
`484c645` with the representative topology preset, `taskset -c 2`, `repeat=10`,
`warmup=4`, `inner=5000`, and `ring-size=16`. Artifacts are saved under
`veqlib/experiments/484c645-20260622-p1-source-dual-matvec-topology`.

| stage | geomean ratio | improved rows | worst row |
| --- | ---: | ---: | --- |
| `evaluate_ring` | 0.949 | 7 / 9 | `32x24x1` ratio 1.051 |
| `source_update` | 0.932 | 7 / 9 | `16x16x1` ratio 1.251 |

Decision: the representative matrix supports keeping the candidate. The
geomean is positive for both the direct source stage and state-ring full
evaluate path. The two source-stage regressions are small/low-cost topology
rows in absolute terms (`16x16x1` is sub-300 ns; `64x16x1` ratio 1.041), and
only one full `evaluate_ring` row exceeds 1.0 materially.

## 2026-06-22 P1-B Residual MomentPlan gate

Before writing production `MomentPlan` code, the existing benchmark-only
materialized moment probe was rerun as a lower-bound gate under the current
P1-A baseline. Focused timing used `taskset -c 2`, `repeat=40`, `warmup=8`, and
`inner=10000`; artifacts are saved under
`veqlib/experiments/f1fb208-20260622-p1-residual-momentplan-gate`.

| stage / probe | median ns | p95 ns |
| --- | ---: | ---: |
| `residual_theta_reduce` | 372.3 | 496.0 |
| `residual_radial_project` | 34.6 | 43.1 |
| theta + project combo | 406.9 | - |
| production `residual_pack` | 131.3 | 148.9 |
| `evaluate_ring` smoke | 4429.5 | 4620.4 |

Decision: reject a production moment-buffer `MomentPlan` rewrite in this phase.
The best available materialized-moment proxy is about `3.10x` slower than the
current vectorized `residual_pack`; implementing it would knowingly move away
from the requested end state. Residual work should resume only with a
vector-friendly/blocked moment design that does not replace vectorized rowwise
passes with scalar dependency chains.

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

## Follow-up candidates tested

After the layout retest, several smaller candidates were tested against
`a5c4d3c`. Most were reverted because they did not produce a stable end-to-end
`evaluate` improvement. The residual surface layout was later re-tested against
`7cf44f4` and retained because it aligns the materialized residual producer with
the now-standard physical `[rho][field][theta]` slab contract while keeping
end-to-end timing neutral-to-slightly-positive.

| candidate | key stage ratio | `evaluate` ratio | decision |
| --- | ---: | ---: | --- |
| Skip absent Fourier orders in `GeometryRuntime::update` | `geometry` 0.988 | 1.001 | reject; stage-only clue below decision threshold |
| Hoist repeated residual geometry loads | `residual_update` 0.974 | 0.996 | reject; end-to-end effect too small |
| Explicit glibc `sincos` path | `geometry` 2.386 | 1.835 | reject; severe regression |
| Reduced-Taylor dynamic `sincos(tb)` after split-trig | final full matrix: `geometry` 0.443 median, 45/45 improved | full matrix `evaluate` 0.609 median, 45/45 improved | keep; large RELAXED-only gain with PF Python/C++ max_abs≈5.578e-11 |
| Reduced-Taylor order trim to `sin x^11` / `cos x^10` | default paired: `geometry_phase_split_sincos` 0.928, `geometry` 0.946 | default `evaluate` 0.975, `evaluate_ring` 0.969; 45-topology evaluate median 0.985, 37/45 improved | keep; `x^9/x^8` failed comparator, `x^11/x^10` passed with max_abs≈7.66e-10 |
| Residual surface physical layout `[rho][field][theta]` | re-test: `residual_update` 0.931; `residual_pack` 1.004 | 0.994 | retain; semantic layout alignment, no significant end-to-end regression |
| Residual theta-moment fusion | active-only `residual_fused / (update+pack)` 1.090; naive 1.166 | active-only 1.007; naive 1.021 | reject; scalar moment accumulation lost to materialized update + vectorized rowwise pack |
| Geometry residual-ready descriptor compression | `residual_update` 0.809, but `geometry` 1.021 | `evaluate` 1.004; `evaluate_ring` 1.008 | reject; moved arithmetic into dominant geometry stage without end-to-end gain |
| Geometry absent Fourier order static skip for `harmonic_rows>2` | `32x16x4` geometry-only 0.925; `32x16x8` geometry-only 0.808 | default `32x16x1` neutral; high-Mmax all-stage evaluate about 0.95 / 0.89 | keep; default topology keeps original loop, high Mmax skips absent c-family orders at compile time |
| Source sign-normalization dot fusion | default paired `source_update` 0.977 | default `evaluate` 0.995 and `evaluate_ring` 0.984; 45-topology median 0.995 | keep; removes one independent source-update scan with no route/finite/sentinel branch |
| Geometry theta-loop vectorization pragmas | long paired `geometry` 0.998 | long paired `evaluate` 1.015 and `evaluate_ring` 1.002 | reject; explicit loop hints did not improve geometry and disturbed endpoint codegen |
| Residual pack unit-weight marker | default paired `residual_pack` 1.063 | `evaluate` 1.006 and `evaluate_ring` 1.007 | reject; compiler handles tiny unit-weight vectors better than marker indirection |
| Geometry surface row padding | early paired `geometry` 2.33--2.35 and `residual_update` 1.13 | `evaluate` 1.54--1.59 and `evaluate_ring` 1.56--1.59 | reject; padding destroys the compact row-locality that made `[rho][field][theta]` effective |
| Remove independent `Pn_psin` buffer and read `materialized_heat_input` instead | `source_update` 1.008; `residual_update` 1.003 | `evaluate` 0.993 but mixed-sign pairs; `evaluate_ring` 0.998 | reject; aliasing the duplicate value is semantically clean but not a stable performance win |
| Remove duplicate `source_psin_query/source_parameter_query` buffers | first pass: `source_materialize` 1.002; long rerun: `evaluate` 0.997 | long rerun `evaluate_ring` 1.004 | reject; direct root-psin interpolation did not survive state-ring timing |
| Source `psin_r` regularization/pass reduction | `source_update` 0.954 | 1.000 | reject; no end-to-end gain |
| Geometry hot-loop pointer/index flattening | `geometry` 1.003 | 1.000 | reject; compiler already removes most accessor overhead |

Next optimization work should not repeat the rejected micro-candidates. Residual
surface layout is now an accepted physical-layout cleanup, but simple
theta-moment fusion is also rejected for this topology; residual work should only
resume with a vector-friendly or blocked moment design, otherwise return to
geometry/source structural candidates.

## Planning update after 3851f62 review

The post-review priority is adjusted as follows:

1. Treat the geometry layout change as accepted. PMU is useful to validate the
   cache/conflict mechanism, not to decide whether the wall-clock improvement is
   real.
2. Promote FP build semantics to P0. Split strict/FMA/relaxed modes before
   testing vector sincos or approximate math backends, and keep solve-success
   decisions out of VEQlib route kernels.
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
- Replaced the old hot-path magnitude/validity policy with a narrow diagnostic
  helper: `math::is_finite()` is now a bit-level NaN/inf test that remains
  meaningful under `-ffinite-math-only`. There is no source/operator
  magnitude-validity guard in the route kernel path.

Validation:

- Compile-command inspection confirmed the expected FP flags for `release`,
  `release-strict`, `release-fma`, and `debug`.
- Debug CTest: 3/3 passed.
- RELAXED release CTest: 3/3 passed.
- STRICT release CTest: 3/3 passed.
- FMA release CTest: 3/3 passed.
- RELAXED release Python/C++ PF-psin-uniform/Ip validation passed with
  `max_abs=6.7622192567728945e-12` before Phase 1a and
  `max_abs=5.5774052043489064e-11` after the route-pure kernel split.
- RELAXED release sanity benchmark (`taskset -c 2`, `repeat=15`, `warmup=5`,
  `inner=10000`) reported `geometry median=5554.3095 ns/call` and
  `evaluate median=9109.3373 ns/call`. This is not a paired optimization proof;
  it only confirms Phase 0 did not visibly damage the current layout baseline.

Follow-up same-window FP A/B (`taskset -c 2`, three median-of-medians rounds,
`repeat=25`, `warmup=8`, `inner=10000`) shows that RELAXED must be the
performance baseline:

| FP mode | `geometry` ns/call | `evaluate` ns/call | `evaluate / STRICT` |
| --- | ---: | ---: | ---: |
| `STRICT` | 12223.9 | 16498.8 | 1.000 |
| `FMA` | 10731.0 | 15059.6 | 0.913 |
| `RELAXED` | 5391.8 | 8907.0 | 0.540 |

Decision: future performance A/B work should compare candidate vs baseline
inside RELAXED builds only. STRICT/FMA remain correctness and error-budget
references, not wall-clock baselines for hotspot decisions.

`math::is_finite()` in RELAXED is implemented as a raw double-bit exponent
mask check, not as `std::isfinite()`: exponent bits equal to all ones mean NaN
or inf. This is the only viable local finite probe under `-ffinite-math-only`,
but the RELAXED contract still means NaN/inf should not be used as ordinary
hot-path control flow.

Phase 1a outcome: the timed benchmark path is now a single route-specific
`PF/psin/uniform/Ip` kernel path, not a checked/unchecked dual facade.
`evaluate()`, source materialization, and source update return `void`; they do
not branch on magnitude-validity, finite checks, or fallback sentinel values.
Solve success is interpreted only by the outer solver/validation layer from the
computed residual norm. The generic free/beta source branches were removed from
this PF/psin/uniform/Ip path.

Residual surface layout re-test after the kernel-purity cleanup: physical storage
now uses `[rho][field][theta]` with the logical accessor preserved as
`surface_field(field, rho, theta)`. Paired timing against `7cf44f4` used 5 rounds,
`repeat=25`, `warmup=8`, `inner=10000`, `taskset -c 2`. Median ratios were
`residual_update=0.931`, `residual_pack=1.004`, and `evaluate=0.994`; the change
is retained as a small producer-locality and semantic-alignment cleanup.

Residual theta-moment fusion was tested next against `c307d2c`. A naive variant
that accumulated all possible moments regressed (`residual_fused / materialized
update+pack=1.166`, `evaluate=1.021`). An active-only variant was better but
still slower (`residual_fused / materialized update+pack=1.090`, `evaluate=1.007`).
Reject the simple fusion shape: it removes slab traffic but replaces vectorized
rowwise passes with scalar accumulation dependency chains. Any future fusion
should start with a vector-friendly or blocked moment design.

State-ring benchmark support was added to `veqlib_stage_benchmark` as an
`evaluate_ring` stage with `--ring-size`. It keeps the original same-x
`evaluate` stage unchanged and cycles a deterministic synthetic x-sequence only
for the ring stage. Release smoke timing with `repeat=10`, `warmup=4`,
`inner=5000`, `ring-size=16` observed same-x `evaluate=8115.7 ns/call` and
`evaluate_ring=8191.4 ns/call` (`ring/evaluate=1.009`). Treat this as an input
state-variation harness, not a real solver trajectory.

Topology-matrix infrastructure was added by making `veqlib_stage_benchmark` read
CMake-generated `config::DefaultTopology` instead of a source-level hard-coded
`32x16` topology. The standard debug/release presets now explicitly configure
the previous default benchmark shape (`Nr=32`, `Nt=16`, `Mmax=1`, `x_size=18`).
`stage_topology_matrix.py` can configure isolated build directories and collect
JSON reports for requested `Nr x Nt x Mmax` entries. Smoke coverage ran
`32x16x1`, `32x32x1`, and `32x16x4`; a representative production-stage matrix
then covered nine topologies with `repeat=12`, `warmup=4`, `inner=5000`:

| Topology | `geometry` ns | `evaluate` ns | Geometry share |
| --- | ---: | ---: | ---: |
| `16x16x1` | 2510.8 | 3666.6 | 0.685 |
| `32x16x1` | 5048.1 | 8278.8 | 0.610 |
| `64x16x1` | 10157.5 | 15073.7 | 0.674 |
| `32x8x1` | 2582.0 | 6004.0 | 0.430 |
| `32x24x1` | 7656.2 | 10899.3 | 0.702 |
| `32x32x1` | 10301.8 | 14593.4 | 0.706 |
| `32x64x1` | 20502.7 | 26664.9 | 0.769 |
| `32x16x4` | 5724.1 | 9280.5 | 0.617 |
| `32x16x8` | 7743.2 | 10935.6 | 0.708 |

The result generalizes the hotspot ordering: Geometry remains the largest
single production stage across the representative sweep, and its share grows as
`Nt` increases.

The full pinned `3 x 5 x 3` matrix was then completed with
`taskset -c 2`, `repeat=6`, `warmup=3`, and `inner=4000` for the production
`geometry` and `evaluate` stages. Raw JSON was kept under `/tmp` rather than
committed. Across all 45 topologies, median `geometry/evaluate` share was
0.699, with a range of 0.476--0.803. Aggregated by `Nt`:

| `Nt` | median Geometry share | median `evaluate` ns |
| ---: | ---: | ---: |
| 8 | 0.528 | 5694.4 |
| 16 | 0.626 | 8682.0 |
| 24 | 0.735 | 11522.1 |
| 32 | 0.725 | 15425.7 |
| 64 | 0.774 | 27965.3 |

Only `32x8x{1,4,8}` fell below 0.5 Geometry share. For normal and larger
`Nt`, Geometry remains the optimization priority; small-`Nt` cases are the main
reason to keep source/profile fixed-overhead cleanup as a parallel line.

A first source fixed-overhead cleanup removed the independent `Pn_psin` array and
used `materialized_heat_input` as the synonymous PF/psin/uniform/Ip value. It
passed release CTest and PF validation, and all paired stage sinks matched the
baseline, but timing did not justify keeping it: `source_update` median ratio was
1.008, `residual_update` 1.003, `evaluate` 0.993 with mixed-sign pairs, and
`evaluate_ring` 0.998. A second source cleanup removed duplicate
`source_psin_query/source_parameter_query` buffers and let interpolation read
the root `psin` row directly; the long rerun was `evaluate` 0.997 and
`evaluate_ring` 1.004. Both patches were reverted; future source cleanup should
aim at fixed-refresh policy or larger interpolation/regularization structure
rather than single alias buffers.

A smaller profile staticization probe was also rejected. It used `if constexpr`
helpers to skip generic accessor reads for absent default-topology `v` and `c0`
profile slots. Release build, PF validation, and release CTest passed, with zero
stage sink differences. Nine paired runs (`repeat=24`, `warmup=8`,
`inner=10000`, `ring-size=16`, `taskset -c 2`) measured `geometry` median
ratio≈1.005, `evaluate`≈0.995 with high noise, and `evaluate_ring`≈1.003. The
patch was reverted because it adds code-shape complexity without stable
end-to-end gain.

Geometry residual-ready descriptor compression was tested after `c56a5b6`: the
prototype replaced the 9 raw geometry surface fields with 7 residual-ready fields
(`qR`, `qZ`, `R2`, `JdivR`, `gttdivJR`, `dmetric`, `sin_tb`). Correctness passed
against Python (`max_abs≈5.578e-11`), but paired RELAXED timing rejected it:
`residual_update=0.809`, `geometry=1.021`, `evaluate=1.004`, and
`evaluate_ring=1.008`. Do not move residual recomputation into Geometry unless
the full `evaluate` path improves.

A later high-`Mmax` specialization is retained: Geometry now skips absent Fourier
orders with a compile-time fold when `harmonic_rows>2`, while preserving the old
small loop for the default `Mmax=1` path. The default `32x16x1` geometry timing
is effectively neutral (short all-stage median ratio≈0.998; longer geometry-only
median ratio≈1.006). For `32x16x4`, geometry-only paired median ratio≈0.925; for
`32x16x8`, geometry-only paired median ratio≈0.808. Stage `geometry` and
`evaluate` sink checks matched the baseline exactly for both high-Mmax topologies.

Phase 1a RELAXED validation and timing (three same-window runs; per-run
`taskset -c 2`, `repeat=25`, `warmup=8`, `inner=10000`; table uses the
median of per-run medians):

| Stage | ns/call | Share of `evaluate` |
| --- | ---: | ---: |
| `profiles_all` | 124.5 | 1.5% |
| `geometry` | 5148.0 | 60.4% |
| `source_materialize` | 895.6 | 10.5% |
| `source_update` | 948.0 | 11.1% |
| `residual_update` | 914.9 | 10.7% |
| `residual_pack` | 135.5 | 1.6% |
| `evaluate` | 8521.5 | 100% |

Compared with the previous RELAXED baseline (`evaluate=8907.0 ns/call`,
`geometry=5391.8 ns/call`), route purification plus guard removal improves
`evaluate` by about 4.3%. Geometry remains the dominant hotspot.

Geometry micro-stage probes were then added to `veqlib_stage_benchmark` as
benchmark-only cumulative stages. They duplicate the hot geometry formulas only
for timing decomposition; they are not production kernels and the final
`geometry - geometry_metric_no_store` delta is only a surface-output proxy, not a
PMU store counter. One RELAXED pinned run (`taskset -c 2`, `repeat=15`,
`warmup=5`, `inner=10000`) measured:

| Geometry probe | Median ns/call | Incremental bucket |
| --- | ---: | ---: |
| `geometry_phase` | 504.6 | Fourier phase synthesis |
| `geometry_phase_sincos` | 3752.3 | +3247.7 dynamic `sin/cos(tb)` |
| `geometry_metric_no_store` | 5039.8 | +1287.5 metric/radial arithmetic |
| `geometry` | 5064.5 | +24.7 surface-output proxy |

The dynamic `sin/cos(tb)` bucket is therefore the largest remaining geometry
component in this default topology, roughly 64% of the measured production
`geometry` stage. This supports prioritizing vector/approximate dynamic trig
backends over more surface-layout tweaks.

The first vector-math backend probe rejected the no-restructure path. The system
has glibc `libmvec`, and a standalone canonical loop compiled with
`-ffast-math -fveclib=libmvec` emits `_ZGVdN4v_sin` and `_ZGVdN4v_cos`, so the
backend itself is available. Rebuilding `veqlib_stage_benchmark` with
`-fveclib=libmvec` still left Geometry with scalar `sin@plt`, `cos@plt`, and
`sincos@plt` call sites and no `_ZGV*` symbols. Five paired runs
(`repeat=10`, `warmup=4`, `inner=5000`) measured median ratios of
`geometry_phase_sincos=1.018` and `evaluate=1.003`, so the flag-only candidate
is rejected. Adding a temporary `#pragma clang loop vectorize(enable)` to the
production theta loop also failed to produce vector calls under LTO and was
reverted. Any future vector trig attempt must reshape Geometry into a canonical
materialize-`tb` / vector-trig / metric-consume pipeline and pay for that extra
traffic explicitly.

The structural split-trig candidate was then tested and retained. A
benchmark-only `geometry_phase_split_sincos` probe first showed that separating
phase synthesis from a canonical `sin/cos(tb)` loop improved the trig bucket
even without changing the default build flags (`split/fused≈0.966`; with the
libmvec experiment build, `≈0.938`). The same split was then applied to
production `GeometryRuntime::update`: per radial node it now materializes phase
and derivative arrays, evaluates `sin/cos(tb)` in a dedicated theta loop, and
then consumes those arrays in the metric/surface pass. Nine paired RELAXED runs
against the pre-split baseline measured `geometry=0.897`, `evaluate=0.908`, and
`evaluate_ring=0.925`, with all sinks matching. The libmvec build still does not
justify changing default flags: final `libmvec/normal` was `geometry=1.021`,
`evaluate=0.988`, and `evaluate_ring=0.975`.

The split structure then enabled a much larger RELAXED-only dynamic trig change:
Geometry now uses a domain-specific reduced-Taylor `sincos(tb)` backend. It
reduces `tb` to the nearest `pi/2` quadrant (`|r|<=pi/4`), evaluates high-order
Taylor polynomials for `sin(r)` and `cos(r)`, and restores the quadrant with
branchless selects. A first branchy version already improved default Geometry
but emitted LTO "loop contains a switch statement" vectorization warnings; the
branchless quadrant mapping removed those warnings and is the retained form.
Release/debug CTest and PF Python/C++ comparison passed (`max_abs≈5.578e-11`,
worst field `final.x`). Five paired RELAXED default-topology runs measured
`geometry_phase_split_sincos≈0.253`, `geometry≈0.424`, `evaluate≈0.643`, and
`evaluate_ring≈0.635` against the pre-polynomial split baseline.

The retained polynomial was then trimmed from `sin x^15` / `cos x^14` to
`sin x^11` / `cos x^10`. Release CTest and PF Python/C++ comparison still pass,
with a tighter but accepted comparator margin (`max_abs≈7.66e-10`, worst field
`initial.geometry_V_r`). Nine paired default-topology runs measured
`geometry_phase_split_sincos≈0.928`, `geometry≈0.946`, `evaluate≈0.975`, and
`evaluate_ring≈0.969` relative to the higher-order polynomial baseline. The full
45-topology `evaluate` matrix measured 37/45 improved, median ratio≈0.985 and
range≈0.939--1.150; the apparent worst non-paired `64x8x1` regression was
retested with paired binaries and measured median ratio≈0.982, while
`32x16x8` retested at≈0.979. A lower `sin x^9` / `cos x^8` candidate failed the
1e-9 comparator (`max_abs≈1.65e-7`) and was rejected.

The post-adoption full topology matrix also supports keeping split-trig. Using
the same 45-entry full preset as the pre-split pinned matrix
(`Nr={16,32,64}`, `Nt={8,16,24,32,64}`, `Mmax={1,4,8}`, `taskset -c 2`,
`repeat=6`, `warmup=3`, `inner=4000`), every measured topology improved:
`geometry` median ratio was 0.906 (range 0.766--0.944, 45/45 improved) and
`evaluate` median ratio was 0.929 (range 0.874--0.978, 45/45 improved). By `Nt`,
the median `evaluate` ratios were 0.948, 0.929, 0.918, 0.925, and 0.945 for
`Nt=8,16,24,32,64`, respectively.

The final branchless reduced-Taylor topology matrix is stronger: against the
post-split baseline, all 45 measured topologies improved again. `geometry`
median ratio was 0.443 (range 0.379--0.537, 45/45 improved) and `evaluate`
median ratio was 0.609 (range 0.504--0.756, 45/45 improved). By `Nt`, median
`evaluate` ratios were 0.737, 0.662, 0.578, 0.583, and 0.541 for
`Nt=8,16,24,32,64`, respectively.

After split-trig, reduced-Taylor sincos, residual local-hoisting, and the
post-split metric-probe fix, the default-stage table was remeasured once with
`--stage all` (`taskset -c 2`, `repeat=15`, `warmup=5`, `inner=10000`,
`ring-size=16`). `evaluate` is now 4850.9 ns/call; `geometry` is 1911.9 ns/call
(39.4%), `source_materialize` 896.4 (18.5%), `source_update` 818.6 (16.9%),
`residual_update` 797.1 (16.4%), and `residual_pack` 133.4 (2.8%).
`evaluate_ring` was 4875.5 ns/call. The corrected `geometry_metric_no_store`
probe measured 1921.5 ns/call and still only acts as a surface-output proxy;
the old scalar-libm reference probe `geometry_phase_sincos` remains 3767.5
ns/call, while the retained split reduced-Taylor bucket
`geometry_phase_split_sincos` is 920.4 ns/call.

Phase 2 first geometry micro-results:

| Candidate | Paired result | Decision |
| --- | --- | --- |
| Reuse `inv_JR` for `JdivR = J*J/(J*R)` | 5 early runs: `geometry`≈0.992 and `evaluate`≈0.990 but with reverse pairs; post-polynomial 7-run retest: `geometry_metric_no_store`≈0.997, `geometry`≈1.018, `evaluate`≈1.006, `evaluate_ring`≈1.016 | reject; metric-only arithmetic clue did not survive production/evaluate timing |
| Hoist harmonic profile reads from theta loop to rho loop | 5 paired runs: median `geometry` ratio≈0.984, `evaluate` ratio≈0.993; all evaluate pairs improved | keep; small but stable low-risk gain |
| Explicit per-rho arithmetic hoist (`a*rho`, `a*h`, `k+rho*k_r`) | 9 paired runs: `geometry` median ratio≈1.000, `evaluate`≈1.008, `evaluate_ring`≈0.990; sinks matched | reject; compiler already handles this class of invariants and same-x evaluate regressed |

Residual local load hoisting was kept as a small efficiency cleanup rather than a
structural fusion substitute. The patch loads same-point Geometry fields and
`alpha1/alpha2` into locals inside `residual_update`, matching the retained
`[rho][field][theta]` physical layout. Release/debug CTest and PF Python/C++
comparison passed (`max_abs≈5.578e-11`). Nine paired RELAXED runs
(`taskset -c 2`, `repeat=24`, `warmup=8`, `inner=10000`, `ring-size=16`) measured
`residual_update` median ratio≈0.947, `evaluate`≈0.995, and
`evaluate_ring`≈0.995 with zero sink differences. Keep the change, but continue
to treat residual theta-moment fusion as the only likely larger residual
opportunity.

A source materialization exact-hit cleanup was then retained. The uniform-node
path in `local_barycentric_interpolate_pair()` now checks only the nearest
sample node (`round(q * (N-1))`) instead of scanning all 8 local stencil nodes
for an exact hit before falling back to the unchanged barycentric interpolation.
This removes fixed per-radial-node comparisons without adding any route,
finite-value, or solve-success branch. Release/debug CTest and the PF Python/C++
comparison passed (`max_abs≈5.578e-11`). Seven paired RELAXED default-topology
runs (`taskset -c 2`, `repeat=24`, `warmup=8`, `inner=10000`, `ring-size=16`)
measured `source_materialize` median ratio≈0.868, `source_update`≈0.995,
`evaluate`≈0.956, and `evaluate_ring`≈0.971. The full 45-topology `evaluate`
matrix showed 40/45 improved, median ratio≈0.976 and range≈0.876--1.027; the
benefit is largest at small `Nt` and mostly neutral at `Nt=64`.

After the source exact-hit cleanup, a pinned `--stage all` run measured
`evaluate` at 4756.6 ns/call. The current default-stage breakdown is:
`geometry` 1926.6 ns/call (40.5%), `source_materialize` 768.5 (16.2%),
`source_update` 813.6 (17.1%), `residual_update` 796.3 (16.7%), and
`residual_pack` 135.3 (2.8%). `evaluate_ring` was 4805.8 ns/call. This keeps
Geometry as the largest single bucket, but source update/materialization and
residual update are now close enough that only endpoint-improving changes should
be kept.

A follow-up root-row copy elimination candidate was rejected. The patch made
`update_psin_coordinate()` and the post-regularization `psin_r` normalization
read directly from `source_target_root_fields` for matvec/dot work instead of
first copying the root row into a `RadialVector`. Correctness passed release
CTest and the PF Python/C++ comparison, and seven paired default-topology runs
showed small apparent wins (`source_materialize`≈0.969, `source_update`≈0.990,
`evaluate`≈0.988, `evaluate_ring`≈0.994). The full 45-topology `evaluate` matrix
did not generalize: only 20/45 topologies improved, median ratio≈1.003 and
range≈0.967--1.077. The candidate was reverted; do not keep root-row copy
micro-tuning without stronger topology-wide or assembly/PMU evidence.

A delayed-stencil-start variant of the source exact-hit path was rejected. It
moved `local_uniform_stencil_start(q)` below the nearest-node exact-hit check so
exact hits would skip the stencil-start calculation. The change was semantically
equivalent and passed release CTest plus the PF comparator, but nine paired
default-topology runs measured `source_materialize`≈1.005, `source_update`≈0.999,
`evaluate`≈0.999, and `evaluate_ring`≈0.992. The code was reverted because the
retained nearest-node exact-hit optimization already captures the useful source
materialization win, while this branch reorder is only noise-level.


A residual `psin_over_J` arithmetic-fold candidate was rejected. The patch
computed `psin_r_i / J_ij` once and reused it for `psin_R` and `psin_Z`, reducing
the apparent pointwise multiplication count. Correctness passed release CTest
and the PF Python/C++ comparison (`max_abs≈7.66e-10`). Nine paired default runs
showed only a tiny residual-stage signal (`residual_update` median ratio≈0.995)
and noise-level endpoint movement (`evaluate`≈0.998, `evaluate_ring`≈0.994). The
full 45-topology evaluate matrix had a positive median (≈0.993, 30/45 improved)
but exposed unstable outliers; paired retests of the worst small-theta cases
measured `32x8x4` at≈1.007 and `64x8x4` at≈1.002. The candidate was reverted:
single-expression residual arithmetic folds are not robust enough to keep unless
they produce clear endpoint gains across topology or stronger assembly/PMU
evidence.


A residual `surface_G` row-sum cache candidate was rejected. The patch accumulated
one radial `G` sum during `update_compact()` and let the `block_psin/F` pack path
reuse that value instead of rescanning `surface_G`. Correctness passed release
CTest and the PF Python/C++ comparator (`max_abs≈7.66e-10`). Default paired
timing showed the expected stage trade: `residual_pack` became much faster
(median ratio≈0.836) while `residual_update` slowed (≈1.014); endpoint movement
was weak (`evaluate`≈0.992) and state-ring was neutral to slightly worse
(`evaluate_ring`≈1.001). The full 45-topology evaluate matrix was mixed: 31/45
improved, median ratio≈0.992, range≈0.859--1.054. The candidate was reverted
because, with only one active `G` consumer in the current PF/psin/uniform/Ip
shape, it mostly moves a vector-friendly rowwise sum into the pointwise update
dependency chain instead of removing robust endpoint work.


A `regularize_psin_r()` pass-fusion candidate was rejected. It combined the
axis-fix rewrite with the `1e-10` floor clamp so the axis segment would not be
written and then revisited by the full clamp pass. Correctness passed release
CTest and the PF Python/C++ comparator (`max_abs≈7.66e-10`). Default paired
timing did not support retention: `source_materialize` median ratio≈0.992,
`source_update`≈0.999, `evaluate`≈1.008, and `evaluate_ring`≈1.002. The patch was
reverted because the default `fix_rho` touches very few radial nodes, so the
minor write saving is outweighed by branch/code-shape noise.


A residual-pack static weight table candidate was rejected. It replaced the
runtime-returned `unit_weights()`, `rho_power<P>()`, and `theta_sin/cos<Order>()`
temporaries with class-scope `inline static constexpr` tables returned by
reference. Correctness passed release CTest and the PF comparator, but default
paired timing was clearly worse: `residual_pack` median ratio≈1.390,
`evaluate`≈1.009, and `evaluate_ring`≈1.018. The patch was reverted because the
compiler appears to scalarize or inline the tiny temporary tables better than it
handles forced static-object loads in this hot pack path.


A reduced-Taylor Estrin polynomial candidate was rejected. It regrouped the
retained `sin x^11` / `cos x^10` Horner form into `r2/r4/r8` terms to shorten the
apparent dependency chain. Correctness passed release CTest and the PF
comparator, but default paired timing was worse in the target bucket:
`geometry_phase_split_sincos` median ratio≈1.019, `geometry`≈1.007,
`evaluate`≈0.996, and `evaluate_ring`≈1.006. The patch was reverted because the
extra temporaries/multiplies and register pressure outweighed any dependency
benefit under the current compiler and RELAXED flags.

A source sign-normalization pass reduction was retained. The patch fuses the
weighted-sign dot for `psin_r` into the loop that negates and divides by the
radial `Kn`, removing one independent `weighted_profile_sign(psin_r)` scan while
leaving the pure `PF/psin/uniform/Ip` route kernel branch-free with respect to
finite values or solve-success decisions. Release/debug CTest passed and the
RELAXED PF Python/C++ comparator passed with `max_abs≈7.66e-10`. Nine paired
default-topology runs measured `source_update` median ratio≈0.977,
`evaluate`≈0.995, and `evaluate_ring`≈0.984. The full 45-topology `evaluate`
matrix was mixed but positive by median: 26/45 improved, median ratio≈0.995,
mean≈0.994, range≈0.851--1.035. The three apparent worst non-paired regressions
were retested with paired binaries and did not show a strong endpoint regression:
`32x8x8` measured `evaluate/evaluate_ring` medians≈0.998/0.994, `16x8x4`
≈0.995/0.986, and `64x16x1`≈0.996/1.001. Keep the change as a narrow
source-update pass deletion; do not reintroduce checked facades or finite-value
guards around it.

A geometry theta-loop vectorization-pragma candidate was rejected. It added
`#pragma clang loop vectorize(enable)` before the phase-synthesis theta sweep
and the metric/store theta sweep, leaving the formulas unchanged. Correctness
passed release CTest and the RELAXED PF comparator (`max_abs≈7.66e-10`). The
first 9-pair timing run looked mildly positive at the endpoint
(`evaluate` median ratio≈0.988, `evaluate_ring`≈0.994), but the target stage did
not move (`geometry`≈0.999). A longer 7-pair retest reversed the decision:
`geometry`≈0.998, `evaluate`≈1.015, and `evaluate_ring`≈1.002. The patch was
reverted because explicit loop hints do not create stable geometry work removal
here and can perturb surrounding codegen.

A residual pack unit-weight marker candidate was rejected. It changed
`unit_weights()` from a small all-ones `RadialVector` temporary into a zero-state
`UnitWeights` marker and routed `project_scaled()` through `weight_value()` so
unit weights returned `1.0` directly. Correctness passed release CTest and the
RELAXED PF comparator, but default paired timing regressed the target bucket and
the endpoint: `residual_pack` median ratio≈1.063, `evaluate`≈1.006, and
`evaluate_ring`≈1.007. The patch was reverted. This reinforces the earlier
static-weight-table rejection: the compiler currently scalarizes or optimizes
the tiny vector temporaries better than extra weight-dispatch abstraction.

A geometry surface row-padding candidate was rejected quickly. It changed the
physical geometry slab from `[rho][9 fields][theta]` to `[rho][10 field slots][theta]`
with one unused pad slot per radial row while preserving the logical accessor.
Correctness passed release CTest and the RELAXED PF comparator, but the first
paired default rounds already showed a severe regression: `geometry` ratios
≈2.33--2.35, `residual_update`≈1.13, `evaluate`≈1.54--1.59, and
`evaluate_ring`≈1.56--1.59. The long run was interrupted and the patch reverted.
This confirms that, after the accepted `[rho][field][theta]` layout, extra row
padding destroys useful compact row locality instead of improving cache behavior.

After retaining source sign-normalization dot fusion, the now-unused
`weighted_profile_sign()` helper was removed so future changes do not accidentally
reintroduce the independent sign scan. This cleanup does not alter the hot path;
release/debug CTest and the RELAXED PF comparator passed.

## Close-out for the remaining-candidates pass on 2026-06-21

The final adopted change from this pass is the source sign-normalization dot
fusion, followed by deletion of the now-unused `weighted_profile_sign()` helper.
The retained performance evidence is: default paired `source_update` median
ratio≈0.977, `evaluate`≈0.995, `evaluate_ring`≈0.984; the 45-topology
`evaluate` matrix had median ratio≈0.995; and paired retests of the apparent
worst topologies did not show a strong regression.

Rejected in this pass: geometry theta-loop vectorization pragmas
(`evaluate`≈1.015 on the longer retest), residual `UnitWeights` marker
(`residual_pack`≈1.063, endpoint regressed), and geometry surface row padding
(`geometry`≈2.33--2.35, endpoint≈1.54--1.59 in the first paired rounds). These
join the earlier rejected source/residual micro-candidates and should not be
retested without new assembly, compiler-version, or native-PMU evidence.

Final code baseline before this close-out documentation commit (`3bf2fe9`) stage-all smoke, measured with `taskset -c 2`,
`repeat=15`, `warmup=5`, `inner=10000`, `ring-size=16`:

| stage | median ns/call |
| --- | ---: |
| `profiles_all` | 125.7 |
| `geometry` | 1818.9 |
| `source_materialize` | 773.1 |
| `source_update` | 816.1 |
| `residual_update` | 798.4 |
| `residual_pack` | 136.8 |
| `evaluate` | 4682.3 |
| `evaluate_ring` | 4858.8 |

Final verification for the close-out: release/debug CTest passed and the RELAXED
PF Python/C++ comparator passed with `max_abs≈7.66e-10`. Remaining larger work
is outside this micro-candidate pass: native-PMU mechanism validation, a
deliberate external/vector math backend evaluation, and any broader Python/Numba
route-specialization work should be started as separate scoped phases.
