# Solver

`Solver` owns the nonlinear solve workflow. It does not define the packed layout, implement residuals, or hold physical runtime state; those responsibilities belong to `Operator`. `Solver` organizes initial values, SciPy methods, fallback, residual acceptance criteria, collocation polish, and history around a residual callable.

Source location: `veqpy/solver/`.

## Configuration and Initial Values

`SolverConfig` stores control parameters such as the main method, maximum residual, maximum function evaluations, initial-value policy, fallback toggle, residual normalization, and collocation-polish toggle.

Initial values are chosen by priority:

| Source | Behavior |
| ------ | -------- |
| Explicit `x0` | Validated and used as the current initial value |
| `warm` | Reuses the previous final solution |
| `zeros` | Uses a zero packed vector |
| `homothetic` | Uses a boundary-shape estimate |
| Default | Encodes the initial value from `OperatorCase.profile_coeffs` |

After a solve, `Solver.x0` is updated to the final solution, which supports warm starts in parameter scans or continuation workflows.

## Solve Flow

A normal variational solve proceeds as follows:

1. Merge default config with temporary overrides passed to `solve()`.
2. Construct the initial value and call the main SciPy method.
3. If it fails and fallback is enabled, try fallback methods.
4. Select a successful result whose residual passes the threshold; if none succeeds, retain the smallest residual norm for diagnosis.
5. Return `SolverResult` and optionally write history.

Success is not based only on SciPy's `success`; the final residual norm must also pass the acceptance threshold.

## Residual Normalization

Residual normalization reduces amplitude imbalance across residual blocks. The strategy is selected through a registry, so the solver workflow depends only on `make_residual_scale()`. Current modes include `none`, `fast` (`block_rms`), `balance`/`balanced` (`block_huber`), and `safe` (`block_sensitivity`).

## Collocation Polish

When `enable_collocation=True`, the solver first completes the variational solve and then warm-starts collocation polish from that result. `collocation_weight` controls whether polish is skipped, mixed with a coefficient-space anchor, or optimized only against the collocation residual.

Collocation polish is a post-processing improvement; it does not change VEQPy's primary solve definition. The solver still uses only residuals and block metadata exposed by the operator and does not interpret the physical details of the residual.

## Result and History

`SolverResult` stores the initial value, final solution, success flag, message, residual norm, call counts, and elapsed time. When history is enabled, case, config, and result snapshots are recorded for comparing solve attempts.
