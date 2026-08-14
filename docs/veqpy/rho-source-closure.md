# Native rho Source Closure

VEQ's geometric `r` remains the operator coordinate. A topology with
`coordinate="rho"` instead states that the pressure and route-driver
samples are functions of

```text
s = sqrt(Phi_N) = rho.
```

Because `s(r)` depends on the equilibrium produced by the same source stage,
this is a local nonlinear coordinate closure. It is not adapter-time
resampling and does not add coordinate coefficients to the outer
Grad--Shafranov unknown vector.

## Coordinate Map

Each residual evaluation starts from a deterministic geometry seed. Assuming
the exactly known edge field throughout the plasma gives

```text
F_0(r) = F_edge,
Phi_N,0(r) = integral_0^r F_edge*Ln_r dr / integral_0^1 F_edge*Ln_r dr,
s_0(r) = sqrt(Phi_N,0).
```

The seed is deliberately not taken from the preceding residual evaluation.
Line-search order and nonlinear-solver history therefore cannot alter the
source result. If a non-physical outer trial geometry cannot construct this
map, the source stage falls back to `s_0=r, ds_0/dr=1`, preserving the old
trial-state domain.

For PF, PP, PI, PJ1, and PQ, one local iteration performs the following
operations.

1. Interpolate both source profiles from their fixed `s` nodes to the current
   `s(r)`.
2. Transform derivative-valued inputs by the chain rule:

   ```text
   dp/dr = dp/ds * ds/dr,
   d(FF)/dr = d(FF)/ds * ds/dr       # PF only
   ```

   PI, PJ1, PJ2, PJ3, and PQ drivers are value profiles. PP retains the public
   `psi_r` driver semantics and only changes the coordinate on which that
   profile is sampled.
3. Evaluate the existing geometric-r source closure.
4. Recover the physical toroidal field. With `F_edge = R0*B0`,

   ```text
   G = F*dF/dr = alpha1*alpha2*FFn_psin*psin_r
   F(r)^2 = F_edge^2 - 2*integral_r^1 G(r) dr.
   ```

5. Rebuild the normalized toroidal-flux coordinate:

   ```text
   g = F*Ln_r
   Phi_N(r) = integral_0^r g(r) dr / integral_0^1 g(r) dr
   s_new = sqrt(Phi_N)
   ds_new/dr = g / (2*s_new*integral_0^1 g(r) dr).
   ```

The factor `2*pi` in `Phi_r` cancels under normalization. The accepted fixed
point must have positive `F^2`, finite nonzero edge flux, and strictly
increasing `s` with positive derivative. Invalid physics raises an explicit
source-stage error.

The coordinate defect is

```text
max(
    max(abs(s_new - s)),
    max(abs(s_r_new - s_r) / (1 + abs(s_r_new))),
).
```

Production uses unrelaxed Picard iteration, an internal and non-configurable
`1e-6` tolerance, and a hard limit of 16 iterations. Reaching the limit is a
solve failure; no partially converged coordinate is silently accepted.

## Single-Layer PJ2/PJ3 Closure

PJ2 and PJ3 do not place a converged `(u, C)` current subproblem inside every
coordinate iteration. Their local state is instead advanced by one joint map:

```text
(s, s_r, u, C) -> (s_new, s_r_new, u_new, C_new),
u = log(F^2/F_edge^2),
C = K_n*dpsi/dr = mu0*I_tor/(2*pi).
```

One joint iteration remaps the source profiles at the current `s`, applies the
pressure chain rule and any algebraic beta multiplier, evaluates exactly one
strict PJ2/PJ3 physics map for `(u_new, C_new)`, and then rebuilds
`(s_new, s_r_new)` directly from `u_new`. Its convergence certificate is

```text
max(coordinate_value_defect,
    coordinate_derivative_defect,
    dimensionless_(u,C)_physics_defect) <= 1e-6.
```

The whole state cold-starts from the geometry seed with `(u, C)=(0, 0)`. This
makes the initial coordinate consistent with the initial `F=F_edge` state.
The map evaluates the source profiles, `F`, and physics right-hand side at the
current state before measuring its defect. Once that defect passes, those
already-consistent fields are published directly; the implementation does not
advance and reevaluate a redundant final Picard map. Thus PJ2/PJ3 have one local
iteration level, one tolerance, one iteration counter, and no history-dependent
warm state.

## Node Semantics

- `nodes="uniform"` means endpoint-inclusive uniform samples in `s`.
- `nodes="grid"` uses the operator Gauss node values as coordinates in `s`.
  The samples are not preprojected values on geometric `r`; a global
  barycentric interpolant supplies the changing `s(r)` queries.
- `nodes="explicit"` retains caller-supplied, endpoint-inclusive arbitrary
  nodes and their original source profiles. A precomputed PCHIP representation
  is queried directly at every changing `s(r)` iterate; there is no
  adapter-time projection through the operator grid.

All node paths use preallocated coordinate, derivative, transformed-source, `F`,
and physical-state arrays. The hot residual loop allocates no per-iteration
arrays. The Cxx backend does not yet implement this local closure and rejects
the coordinate explicitly.

## Convergence Qualification

The isolated coordinate experiment
[`rho_fixed_point_experiment.py`](../../benchmarks/rho_fixed_point_experiment.py)
uses converged source profiles and fixed geometry to scan the coordinate map.
It covers all 27 legal route/constraint combinations, initial guesses
`s_0=r^p` for `p` from `0.25` through `2.0`, and relaxation factors `1.0`,
`0.7`, and `0.5`. At the production `1e-6` tolerance all 567 cases converge.

| relaxation | cases passed | iteration range | median |
| --- | ---: | ---: | ---: |
| `1.0` | 189/189 | 3--5 | 4 |
| `0.7` | 189/189 | 12--15 | 12 |
| `0.5` | 189/189 | 19--26 | 20 |

The production-form joint scan
[`pj23_rho_joint_experiment.py`](../../benchmarks/pj23_rho_joint_experiment.py)
covers PJ2/PJ3, uniform/grid nodes, four constraints, five coordinate seeds,
and nine `(u, C)` seed scalings. All 720 cases converge without relaxation in
6--9 iterations. PJ2 takes 6--7 iterations and PJ3 takes 7--9; the largest
accepted joint defect is `9.98e-7`. The physics defect controls the stopping
point in this scan, while the largest accepted coordinate defect is
`9.70e-8`.

A separate native solve matrix at `Nr=24`, `Nt=12` covers both node modes and
every legal constraint. All 54 solves succeed in 3--9 local iterations, with a
maximum local defect of `8.82e-7`. The converged internal coordinate agrees
with the independently reconstructed `Equilibrium.rho` within
`1e-7`, below the local closure gate. These scans qualify the current
synthetic route domain; they are not a proof of global Picard convergence for
arbitrary physical inputs.

## Route Timing

[`rho_route_benchmark.py`](../../benchmarks/rho_route_benchmark.py)
compares every native route with its same-node geometric-r case at `Nr=32`,
`Nt=16`. The current-case hot residual excludes source lowering, case rebinding,
JIT compilation, and setup. The complete solve reuses a warmed Kernel handle,
but every timed solve starts from the same cold outer initial state.

| route | nodes | hot residual ratio | complete solve ratio |
| --- | --- | ---: | ---: |
| PF | uniform | 1.73x | 1.59x |
| PP | uniform | 1.71x | 1.34x |
| PI | uniform | 1.73x | 1.58x |
| PJ1 | uniform | 1.76x | 1.52x |
| PJ2 | uniform | 1.44x | 1.33x |
| PJ3 | uniform | 1.37x | 1.31x |
| PQ | uniform | 2.18x | 2.08x |
| PF | grid | 1.86x | 1.74x |
| PP | grid | 1.89x | 1.43x |
| PI | grid | 1.87x | 1.67x |
| PJ1 | grid | 1.90x | 1.62x |
| PJ2 | grid | 1.75x | 1.55x |
| PJ3 | grid | 1.66x | 1.46x |
| PQ | grid | 2.29x | 2.18x |

Across the 14 rows, the hot-residual multiplier is 1.37--2.29x with a median
of 1.76x. The complete-solve multiplier is 1.31--2.18x with a median of 1.56x
and mean of 1.60x. Absolute complete-solve medians are 0.489--0.749 ms for r
and 0.772--1.631 ms for `rho`. The smaller complete-solve multiplier occurs
because outer function-evaluation counts are similar or slightly lower after
the coordinate change.

The corresponding r and `rho` cases express the same analytic profiles
in different source coordinates. Maximum normalized profile differences are
`1.58e-5` for `psin`, `2.29e-4` for `psi_r`, `7.29e-6` for `F`, `5.42e-5` for
`P`, `3.16e-3` for `q`, and `1.20e-2` for the derivative-sensitive `jtor`.

A same-process seed ablation over all 27 legal route/constraint combinations
compares the previous neutral `s=r` seed with the geometry seed while keeping
the compiled closure and converged outer state fixed. Median hot-residual
changes are -4.4% for PF, -5.1% for PP, -7.6% for PI, -5.4% for PJ1, +0.4% for
PJ2, -1.5% for PJ3, and -15.7% for PQ. All cases continue to converge. The
geometry seed is therefore retained as a deterministic reduction in closure
work, not as history-dependent continuation.

## GEQDSK Timing

[`rho_geqdsk_benchmark.py`](../../benchmarks/rho_geqdsk_benchmark.py)
compares the official GEQDSK PF/uniform/ip Numba configuration in `psin` with
the corresponding native `rho` solve. GEQDSK `q=dPhi/dPsi` is integrated
to construct `sqrt(Phi_N)` source nodes, and PF derivatives are transformed by
the exact chain rule. Both sides use Powell, cold outer initialization, five
warmups, and 30 timed solves.

| case | config | psin (ms) | rho (ms) | ratio | qualified |
| --- | --- | ---: | ---: | ---: | --- |
| Solovev | Low | 1.386 | 1.570 | 1.13x | yes |
| Solovev | Medium | 1.452 | 1.609 | 1.11x | yes |
| Solovev | High | 1.698 | 1.882 | 1.11x | yes |
| Solovev | Ref | 5.613 | 5.913 | 1.05x | yes |
| CHEASE | Low | 4.233 | 6.746 | 1.59x | yes |
| CHEASE | Medium | 4.757 | 10.113 | n/a | no |
| CHEASE | High | 9.733 | 9.516 | 0.98x | yes |
| CHEASE | Ref | 64.044 | 31.907 | n/a | no |
| EFIT | Low | 2.498 | 2.827 | 1.13x | yes |
| EFIT | Medium | 3.182 | 3.627 | 1.14x | yes |
| EFIT | High | 7.804 | 8.911 | 1.14x | yes |
| EFIT | Ref | 15.707 | 19.327 | 1.23x | yes |

For the 10 of 12 rows where both solves pass the same residual qualification,
changing from `psin` to `rho` gives a 0.98--1.59x multiplier, a median of
1.13x, and a mean of 1.16x. CHEASE Medium and Ref are excluded rather than
timed as valid comparisons: their `rho` raw residuals are `2.61e-5` and
`5.93e-5`, respectively, above the `1e-6` acceptance threshold. The GEQDSK
matrix therefore does not yet establish universal production qualification for
native `rho`; those two nonlinear-solver cases remain explicit follow-up
work.
