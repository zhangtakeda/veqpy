# P2 Geometry vectorization evidence

Baseline: `34ee457` after Source block-4 was retained. This phase records the
current Geometry code-generation state before attempting metric-loop rewrites.
No production code was changed.

Commands:

```bash
cmake -S veqlib -B veqlib/build/analysis \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_ENZYME=OFF \
  -DVEQLIB_ENABLE_NATIVE_OPTIMIZATIONS=ON \
  -DVEQLIB_FP_MODE=RELAXED \
  -DVEQLIB_ENABLE_THIN_LTO=OFF \
  -DVEQLIB_ANALYSIS_BUILD=ON \
  -DVEQ_NR=32 -DVEQ_NT=16 \
  -DVEQ_H_PROFILE_COUNT=3 -DVEQ_V_PROFILE_COUNT=0 \
  -DVEQ_KAPPA_PROFILE_COUNT=6 -DVEQ_PSIN_PROFILE_COUNT=6 \
  -DVEQ_F_PROFILE_COUNT=0 -DVEQ_COS_PROFILE_COUNTS=0 \
  -DVEQ_SIN_PROFILE_COUNTS=3 -DVEQ_PROFILE_KMAX_LIMIT=2
cmake --build veqlib/build/analysis --target veqlib_main
objdump -Cd --no-show-raw-insn --start-address=0x469c0 --stop-address=0x47e60 \
  veqlib/build/release/veqlib_main
llvm-mca-18 -mcpu=native -iterations=100 geometry-metric-vector-loop-release.s
```

Key evidence:

| source | observation |
| --- | --- |
| Clang remarks | `geometry.h:381` dynamic `sincos(tb)` pass vectorized with width 4. |
| Release objdump | Default `GeometryRuntime::update` contains 557 instruction lines referencing `%ymm`, 2 `vdivpd` instruction lines, and no `vdivsd`; the metric/reduction body is packed-double, not scalar-double. |
| Stack usage | Analysis `veqlib_main` reports the default `GeometryRuntime::update` frame as 2040 B; no Geometry frame warning crosses the 4096 B gate. |
| llvm-mca | Extracted release metric vector block reports block throughput about 30 cycles for a 4-lane body; load/store and FP ports are both active, with vector divides present. |
| Current stage cost | P1-D `stage-all.json` reports `geometry=1793.9 ns`, `evaluate=3877.9 ns`, so Geometry remains the largest single stage, but the compiler already handles the obvious vector radial accumulation shape. |

Decision: do not implement a manual “vector radial accumulator” rewrite as the
next production candidate. The release binary already shows vectorized metric
arithmetic and vector reductions; a hand rewrite would duplicate the compiler's
current shape and risks increasing stack/register pressure. Future Geometry work
should target a more concrete remaining cost, such as reducing stack-resident
phase arrays, simplifying metric formulas, or measuring a two-pass metric/store
split, and must be gated by release `geometry`, `evaluate`, and topology timing.
