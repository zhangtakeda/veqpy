# 2026-06-22 Source accumulator-only block-4 SIMD

This phase extends the retained Source packed output-block SIMD shape from the
paired D/A psin matvecs to the remaining production accumulator-only matvec in
`update_pf_ip_from_psin_uniform()`: the `A * integrand` pass that creates the
unnormalized `psin_r` profile. The old row-dot implementation remains available
as the `source_A_integrand_rowdot` benchmark stage; `source_A_integrand` now
measures the production block-4 path.

Default topology (`Nr=32`, `Nt=16`, `Mmax=1`, `taskset -c 2`, repeat 20,
warmup 5, inner 10000):

| stage | row-dot median | block-4 median | ratio |
| --- | ---: | ---: | ---: |
| `source_A_integrand` | 245.9 ns | 67.9 ns | 0.276 |

Production smoke after enabling the candidate:

| metric | median |
| --- | ---: |
| `source_update` | 244.8 ns |
| `evaluate` | 3682.7 ns |
| `evaluate_ring` | 3704.9 ns |
| residual-only solve | 0.1759 ms |

Representative topology comparison against the previous retained Source block-4
baseline (`d109fd3-20260622-p1-source-block4`):

| stage | geomean ratio | improved rows | worst row |
| --- | ---: | ---: | --- |
| `source_update` | 0.384 | 8 / 9 | `64x16x1` ratio 1.115 |
| `evaluate_ring` | 0.959 | 9 / 9 | `32x32x1` ratio 0.993 |

Decision: keep the accumulator-only block-4 path. It removes the last production
Source row-dot matvec from the PF/psin/uniform/Ip route, gives a large local
`A * integrand` speedup, and improves the full state-ring endpoint across all
representative topology rows. The lone `64x16x1` source-stage regression is not
visible at the full `evaluate_ring` endpoint.

Validation:

- `cmake --build --preset clang-debug -j$(nproc)`
- `ctest --test-dir build/debug --output-on-failure` (5/5)
- `cmake --build --preset clang-release -j$(nproc)`
- `ctest --test-dir build/release --output-on-failure` (5/5)
- `../.venv/bin/python compare_pf_psin_uniform_veqpy.py --cxx-exe build/release/veqlib_main --tolerance 1e-9` (`max_abs≈7.66e-10`)
