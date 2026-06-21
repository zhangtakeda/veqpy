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
| Residual surface physical layout `[rho][field][theta]` | re-test: `residual_update` 0.931; `residual_pack` 1.004 | 0.994 | retain; semantic layout alignment, no significant end-to-end regression |
| Residual theta-moment fusion | active-only `residual_fused / (update+pack)` 1.090; naive 1.166 | active-only 1.007; naive 1.021 | reject; scalar moment accumulation lost to materialized update + vectorized rowwise pack |
| Geometry residual-ready descriptor compression | `residual_update` 0.809, but `geometry` 1.021 | `evaluate` 1.004; `evaluate_ring` 1.008 | reject; moved arithmetic into dominant geometry stage without end-to-end gain |
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
`32x16x1` and `32x32x1`; the full large matrix remains a long-running experiment.

Geometry residual-ready descriptor compression was tested after `c56a5b6`: the
prototype replaced the 9 raw geometry surface fields with 7 residual-ready fields
(`qR`, `qZ`, `R2`, `JdivR`, `gttdivJR`, `dmetric`, `sin_tb`). Correctness passed
against Python (`max_abs≈5.578e-11`), but paired RELAXED timing rejected it:
`residual_update=0.809`, `geometry=1.021`, `evaluate=1.004`, and
`evaluate_ring=1.008`. Do not move residual recomputation into Geometry unless
the full `evaluate` path improves.

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
`evaluate` by about 4.3%. Geometry remains the dominant hotspot, so Phase 1b/2
should add the topology/state matrix and decompose geometry before attempting
more source micro-optimizations.

Phase 2 first geometry micro-results:

| Candidate | Paired result | Decision |
| --- | --- | --- |
| Reuse `inv_JR` for `JdivR = J*J/(J*R)` | 5 paired runs: median `geometry` ratio≈0.992, `evaluate` ratio≈0.990, but evaluate had reverse runs (`1.031`, `1.009`) | reject; too small/noisy |
| Hoist harmonic profile reads from theta loop to rho loop | 5 paired runs: median `geometry` ratio≈0.984, `evaluate` ratio≈0.993; all evaluate pairs improved | keep; small but stable low-risk gain |
