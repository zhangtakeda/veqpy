# P1 Source packed block-4 matvec

Baseline: `d109fd3` (`KernelPlan` invalidation boundary). Candidate: current
Source path with packed block-4 D/A matvec for the two production psin passes.
Runs used `taskset -c 2` in the local WSL2 environment.

Focused default-topology probe (`Nr=32`, `Nt=16`, `Mmax=1`, repeat 20,
warmup 5, inner 10000):

| stage | row-dot median ns | block-4 median ns | ratio |
| --- | ---: | ---: | ---: |
| `source_DA_psin` | 420.0 | 102.9 | 0.245 |

Default production-stage smoke after enabling the candidate:

| stage | median |
| --- | ---: |
| `source_materialize` | 367.0 ns |
| `source_update` | 437.4 ns |
| `evaluate` | 3877.9 ns |
| `evaluate_ring` | 3909.0 ns |
| residual-only solve | 0.183 ms |

Representative topology A/B (`repeat=6`, `warmup=3`, `inner=4000`,
`ring-size=16`) compared current candidate against baseline `d109fd3`:

| stage | geomean ratio | improved rows | worst row |
| --- | ---: | ---: | --- |
| `source_update` | 0.650 | 9 / 9 | `32x64x1` ratio 0.933 |
| `evaluate_ring` | 0.929 | 7 / 9 | `64x16x1` ratio 1.082 |

Decision: keep the block-4 Source path. It clears the Source gate and the
representative full-evaluate geomean gate. The two `evaluate_ring` regressions
are topology-specific and outweighed by the positive default and geomean results;
future topology work should revisit `64x16x1` if it becomes a priority.
