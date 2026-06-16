# Solver

`Solver` owns the nonlinear solve workflow for one fixed `Operator` layout. It
does not define the packed layout, implement source routes, or hold physical
runtime fields; those responsibilities belong to `Operator`. The solver decides
which packed initial state to use, which SciPy optimizer to call, how residuals
are scaled for the nonlinear method, how fallback attempts are selected, and
whether to run a post-solve collocation polish.

Source location: `veqpy/solver/`.

## Configuration and Initial Values

`SolverConfig` stores the default solve policy. Per-call keyword arguments to
`solve()` create a temporary config snapshot by replacing only the fields passed
for that call; the solver's default config is otherwise unchanged.

The default variational method is SciPy `root(..., method="hybr")`. The default
fallback is `least_squares(..., method="lm")`. `trf` is also available through
`method` or `fallback_methods`. Collocation polish uses least-squares methods
only, with `lm` as the default collocation method.

The solver owns one packed vector, `Solver.x0`. At construction time, this is
initialized from `Operator.zero_state()` for the operator topology. Callers that
already have profile coefficients can build an explicit initial vector with
`Operator.pack_coefficients(...)` and pass it as `x0`. After every solve,
`Solver.x0` is replaced by the final packed solution. This is the state that
later warm starts and `build_equilibrium()` use.

Initial values are chosen by priority:

| Source | Behavior |
| ------ | -------- |
| Explicit `x0` | Validated by the operator, copied into `Solver.x0`, and used for this solve |
| `initial_policy="warm"` | Reuses the current solver-owned `Solver.x0` |
| `initial_policy="zeros"` | Uses a zero packed vector |
| `initial_policy="geometric"` | Uses geometric shape/axis estimates without source/profile refinement |
| `initial_policy="geometric-refined"` | Adds source-implied low-order active-profile seeds to the geometric estimate |
| `initial_policy="legacy-geometric"` | Uses the legacy `0.66 * a / R0` axis-shift estimate and geometric shaping terms |
| `initial_policy="auto"` | Uses zeros below the boundary curve-strain threshold, otherwise `geometric-refined` |
| Default (`initial_policy=None`) | Uses the operator zero state |

The geometric initializer is meant as a cheap geometry-based guess for nested
flux surfaces. It delegates to the operator's boundary-slope estimate: active
Fourier shaping coefficients receive first-coefficient values derived from the
boundary offsets, and `h` receives a continuous Shafranov-shift estimate from
pressure/current source roughness moments. Small source structure and small axis
shifts are treated as zero, so uniform Solovev-like sources decay smoothly to
zero instead of taking a separate branch. The initializer uses one conservative
operator-side estimate rather than exposing a separate scale factor.

The refined geometric policy starts from the same geometric estimate and adds
cheap low-order active-profile predictions from source-implied targets. At
present this projects the normalized flux target onto only the first active
`psin` basis term when the operator route owns an active `psin` profile. This
remains a scalar weighted least-squares correction, applies a conservative
half-step seed, and clips the coefficient to preserve positive `psin_r` for the
initialized profile.

The `auto` policy is a cheap boundary-geometry gate. It samples the boundary
curve implied by the Fourier angle correction, compares its arclength speed
`ds/dtheta` with the ellipse using the same `a/R0/Z0/ka`, and computes the RMS
relative curve strain. Values below the built-in threshold `0.32` use a zero
packed initial vector; stronger non-elliptic boundary curves use
`geometric-refined`. The elongation `ka` is part of the reference ellipse and
does not trigger the refined path by itself.

Whenever the solve starts from an explicit `x0`, zeros, or any internally built
geometric state, the operator invalidates route-local source state before the
attempt. `warm` keeps the current source state paired with the current `x0`.

## Solve Flow

A normal variational solve proceeds as follows:

1. Merge default config with temporary overrides passed to `solve()`.
2. Construct the packed initial value according to the priority above.
3. Call the primary SciPy method against the variational residual.
4. If it fails and fallback is enabled, retry configured fallback methods from
   the same initial guess.
5. Select the first accepted successful attempt; if none is accepted, retain the
   finite-result attempt with the smallest residual norm for diagnosis.
6. Return `SolverResult` and optionally write history.

For the variational solve, success is strict: SciPy `success` alone is not
enough. The residual norm must also pass the solver acceptance threshold, which
is `max(10 * max_residual, 1e-5)`. If an optimizer raises an exception but the
starting point already satisfies that residual threshold, the attempt can still
be accepted as an already-solved state.

## Residual Normalization

Residual normalization reduces amplitude imbalance across packed residual
blocks before the nonlinear method sees them. The raw residual reported in
`SolverResult` is still the operator residual, not the scaled vector used inside
the optimizer.

Current modes are:

| mode | Meaning |
| ---- | ------- |
| `none` | Use the raw residual directly |
| `fast` / `block_rms` | Scale each active residual block by its initial RMS, with a floor of 1 |
| `balance` / `balanced` / `block_huber` | Build robust block scales using Huber-style RMS, a floor, and a maximum scale ratio |
| `safe` / `block_sensitivity` | Combine robust block amplitude with finite-difference sensitivity probes |

The default config uses `fast`. Passing `residual_normalization=None` resolves
to the same package default. For `hybr`, enabling normalization also tightens
the initial trust-region factor to reduce large first steps in the scaled
residual space.

## Collocation Polish

When `enable_collocation=True`, the solver first completes the variational solve
and then warm-starts a second least-squares solve from that result. This is a
two-stage workflow: variational solve first, collocation polish second. The
collocation stage disables fallback and uses `collocation_method`,
`collocation_max_residual`, and `collocation_max_evaluations` when provided.

`collocation_weight` selects the polish objective:

| weight | Objective |
| ------ | --------- |
| `0` | Skip the collocation objective and keep the variational solution |
| `(0, 1)` | Minimize a blended vector: coefficient-space distance from the variational solution plus the point-collocation residual |
| `1` | Optimize only the point-collocation residual |

The blended objective keeps the polish local in coefficient space unless the
collocation residual has enough weight to move away from the weak-form solution.
Collocation polish is therefore a post-processing improvement; it does not
change VEQPy's primary solve definition.

## Result and History

`SolverResult` stores the initial packed vector, final packed vector, success
flag, message, final residual norm, function/Jacobian/iteration counts,
`elapsed` solve time, and `total_elapsed` call time. For built-in initial
policies, `elapsed` starts before initial-state construction and covers
initialization, nonlinear attempts, and the final residual check. If the caller
passes an explicit `x0`, caller-side construction of that vector is naturally
outside `SolverResult.elapsed`. `total_elapsed` starts at `solve()` entry and
includes per-call config resolution plus the solve work. After each solve,
`Solver.result` points to the newest result and `Solver.x0` is updated to the
final solution.

When history is enabled, `SolverRecord` snapshots the current problem, the
per-solve config, and the result. `clear()` removes this history without
changing `Solver.x0`; `reset()` zeros `Solver.x0` in place.
