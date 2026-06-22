# 2026-06-22 P3 parameter-scan continuation benchmark

This phase adds a benchmark-only continuation scan to `veqlib_main --mode solve`.
It varies the scaled `Ip` constraint by a fixed relative step and compares three
initial-guess policies for the same PF/psin/uniform/Ip residual-only CMINPACK
path:

- `cold`: every point starts from the original zero-profile benchmark guess;
- `warm`: point `i>0` starts from the accepted solution of point `i-1`;
- `secant`: point `i>1` starts from a first-order predictor
  `x_i = x_n + (Ip_i-Ip_n)/(Ip_n-Ip_{n-1}) * (x_n-x_{n-1})`, with explicit
  fallback accounting for the first two history-building points.

The scan JSON records each point's `nfev`, solve wall time, success flag, raw
residual vector/norm, initial-guess policy, and predictor fallback reason. The
benchmark intentionally keeps Jacobian/Broyden state out of scope; it first
measures how far continuation alone reduces residual callbacks.

Validation command used for the representative corpus:

```bash
taskset -c 2 build/release/veqlib_main \
  --mode solve \
  --scan-points 11 \
  --scan-policy all \
  --scan-relative-step 0.02 \
  > experiments/0225aa8-20260622-p3-scan-continuation/scan-all-11-step0p02.json
```

Representative result for `relative_step=0.02`:

| policy | total `nfev` | point `nfev` pattern | median solve ms | result |
| --- | ---: | --- | ---: | --- |
| `cold` | 428 | `38,38,38,38,38,38,39,40,40,41,40` | 0.2125 | all accepted |
| `warm` | 238 | `38,20,20,20,20,20,20,20,20,20,20` | 0.0878 | all accepted |
| `secant` | 238 | `38,20,20,20,20,20,20,20,20,20,20` | 0.0857 | all accepted |

Additional sweeps at relative steps `0.005`, `0.01`, and `0.05` showed the same
callback pattern: `cold` stays near 38--41 residual evaluations per point, while
warm/secant solve the ten continuation points with 20 residual evaluations each.
For this smooth one-parameter `Ip` scan, warm-start already reaches the observed
CMINPACK evaluation floor, so the secant predictor does not reduce `nfev` beyond
warm-start; it remains useful as an explicit benchmark policy with fallback
telemetry for less linear future scan families.

Decision: keep the benchmark-level warm/secant scan support. Do not re-open
Enzyme/KINSOL/Broyden work before a scan case shows high post-warm-start `nfev`;
continuation alone removes about 43--44% of residual callbacks over this 11-point
corpus, which is larger than the remaining single-callback micro-optimization
headroom.
