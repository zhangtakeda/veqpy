# VEQlib hot-loop evidence ledger

This file is the decision ledger for VEQlib hot-path loops.  It exists so
SIMD, auto-vectorization, and scalar-retention choices are tied to reproducible
evidence instead of implementation style preferences.

## Scope

Record a loop here when it enters the `evaluate()` callback path and satisfies at
least one of these conditions:

- complexity is `O(Nr * Nt)` or higher;
- complexity is `O(Nr^2)` or higher;
- the loop accounts for at least 2% of `evaluate` in the stage benchmark.

Small setup loops, one-shot packing, validation checks, and non-hot diagnostic
paths do not need entries unless they become part of a measured callback path.

## Evidence contract

Every retained hot-loop implementation needs one of the following evidence
classes:

1. explicit SIMD with assembly evidence showing no obvious register spill and no
   unnecessary scalar tail in the measured topology;
2. compiler auto-vectorization with `-Rpass`/optimization-record evidence plus a
   final assembly check;
3. scalar retention with same-machine A/B benchmark evidence showing explicit
   SIMD does not provide stable end-to-end benefit.

Use `clang-analysis` for compiler reports and assembly inspection only.  It
intentionally disables ThinLTO and loop unrolling, so it is not a timing source.
Use `clang-release`, `clang-release-fma`, or a paired release build for timing.

## Standard commands

```bash
cmake --preset clang-analysis
cmake --build --preset clang-analysis --target veqlib_main

./build/release/veqlib_main --mode stage --stage all --repeat 30 --warmup 5 --inner 3000
./build/release/veqlib_main --mode stage --stage evaluate_ring --repeat 30 --warmup 5 --inner 3000
./build/release/veqlib_main --mode solve --repeat 30 --warmup 5

python stage_topology_matrix.py --matrix-preset simd-tail --stage evaluate_ring \
  --repeat 10 --warmup 4 --inner 5000 --output experiments/simd-tail-evaluate-ring.json
```

Pinning to one core with `taskset -c <core>` is recommended for retained timing
artifacts when the environment supports it.

## Current ledger

| Kernel / loop | Stage | Topology | FP mode | Vectorization evidence | Benchmark evidence | Decision |
| --- | --- | ---: | --- | --- | --- | --- |
| Source radial multi-matvec (`tensor_layout::RadialGridMatvecPlan<GridType>`) | `source_update` / `source_materialize` | 32x16x1 | RELAXED/FMA | Explicit AVX2/FMA path exists; analysis artifact still required after extraction | P1 source dual-output packed experiment summaries show retained source/evaluate gains | Keep extracted `tensor_layout.h` / `tensor_kernels.h` boundary |
| Geometry dynamic trig + metric sweep | `geometry` / `evaluate` | 32x16x1 and topology matrix | RELAXED | Reduced-Taylor path is part of current kernel contract; add `clang-analysis` record before further edits | Existing experiments record reduced-Taylor as retained across topology matrix | Keep; do not hand-write new SIMD without A/B evidence |
| Source normalization/sign/root-row passes | `source_normalize` / `source_update` | 32x16x1 | FMA | No retained explicit SIMD; scalar/fused candidates only | `experiments/8657d0a-20260622-pr5-source-fusion/stage-ab.json` rejects sign-store/direct-normalize candidates because production `source_update` did not clear the gate | Keep current production order; revisit only with source-update + solve evidence |
| Residual materialize/update + pack | `residual_update` / `residual_pack` | 32x16x1 | RELAXED | Missing final assembly record | Prior residual fusion probes did not justify replacing materialized update + pack | Keep current shape until a stronger moment-plan candidate clears stage + solve gates |

## Required artifact fields

When adding or updating a row, include the artifact path and record:

- commit or diff identifier;
- compiler preset and FP mode;
- topology (`Nr x Nt x Mmax`);
- stage command and timing statistics;
- whether `evaluate`, `evaluate_ring`, and `solve` moved in the same direction;
- vectorization report or assembly file path when the decision relies on codegen;
- correctness check (`ctest`, Python/C++ comparator, or route-specific validation).

## Non-goals

- Do not make compact `tensor::Tensor` globally padded.
- Do not expand logical `N x N` matrices to `P x P` only to simplify SIMD tails.
- Do not add callback-time route, topology, profile-family, ISA, or plan-validity
  dispatch to support an optimization experiment.
