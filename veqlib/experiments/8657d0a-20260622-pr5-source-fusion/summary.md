# PR5 source fusion rejection A/B

Base commit: `8657d0a` (`Prepare VEQlib operators from setup-only state`).
Candidate: discarded working tree that extracted the PF/psin/uniform/Ip transform+weighted-sum loop and fused sign application with root-row store. A direct root-row normalization fusion was tested first and rejected because `source_update` regressed.

Decision: reject the candidate for production. It improves the isolated `source_normalize` probe and this pinned solve sample moved favorably, but the production `source_update` stage regressed. That is not strong enough evidence to change the numerical hot path.

Build preset: `clang-release-fma`; pinned with `taskset -c 2` to match earlier local evidence style.

| stage | base median ns | candidate median ns | candidate/base | base avg ns | candidate avg ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `source_normalize` | 73.8 | 68.6 | 0.930 | 76.2 | 73.5 |
| `source_update` | 288.3 | 294.4 | 1.021 | 292.4 | 301.6 |
| `evaluate` | 7411.3 | 7411.3 | 1.000 | 7466.6 | 7408.0 |

| solve | base | candidate | candidate/base |
| --- | ---: | ---: | ---: |
| median ms | 0.319759 | 0.315971 | 0.988 |
| avg ms | 0.349285 | 0.329035 | 0.942 |
| nfev | 38 | 38 | 1.000 |

Correctness gates while testing the candidate:

- `git diff --check` passed.
- `cmake --build --preset clang-debug -j2` passed.
- Debug `probe`, `temp-validation`, and `pf-validation` passed.
- Debug `ctest --output-on-failure --timeout 180` passed 5/5.
- `compare_pf_psin_uniform_veqpy.py --tolerance 1e-8` passed with `max_abs=7.655671652173623e-10`.
- `benchmark_pf_psin_uniform_compare.py --repeat 3 --warmup 1 --no-write` passed with C++ accepted, `nfev=38`, and `max_final_x_abs_diff=2.052519265660635e-11`.
