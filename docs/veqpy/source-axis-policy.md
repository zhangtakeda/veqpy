# Rho-Route Magnetic-Axis Policy

## Purpose

VEQPy accepts finite source data without silently imposing magnetic-axis parity
at the public input boundary. This report asks a narrower question: after a
rho-coordinate route has closed its physics, which derived profiles still need
an axis limit inside the hot source kernel?

The governing rule is that a route must not overwrite the quantity it has just
made authoritative. A numerical repair is retained only when it represents a
removable coordinate limit or stabilizes a differentiated, non-authoritative
intermediate without changing the route equation.

## Experiment

[`benchmarks/source_axis_policies.py`](../../benchmarks/source_axis_policies.py)
constructs PF, PP, PI, PJ1, PJ2, PJ3, and PQ solves from one converged synthetic
reference equilibrium. Every route is exercised with both native Gauss samples
and endpoint-inclusive uniform source samples. The cases include:

- an unmodified smooth reference;
- smooth axis-local changes to the pressure or route driver;
- finite but deliberately parity-violating pressure or driver samples;
- a separate invalid-coordinate case in which a coordinate-like driver crosses
  zero.

The benchmark records nonlinear convergence separately from physical
qualification. It checks the route-authoritative profile, the equilibrium shape,
the reconstructed pressure and F profiles, `psin_r`, and exact PJ1/PI current
closure. `fix_rho` is changed only through the private Numba runtime for this
research experiment; it is not a public tuning parameter.

The final policy was tested on 168 valid cases spanning `Nr=32, 48, 64`, all
seven routes, both node contracts, and four smooth or controlled perturbations.
All 168 cases were finite and converged. The pre-existing 54-case route
qualification matrix retained exactly the same pass/fail classification and
the same nonlinear evaluation counts. The failures in that matrix are profile
qualification failures for deliberately strong perturbations, not nonlinear
solver failures.

The parity-violating cases are not claimed to be good physical equilibria.
After removing a hidden post-fit, some of them move farther from the smooth
reference because the solver now exposes the supplied irregularity instead of
silently changing it. That is the intended contract: route closure and input
preservation are tested separately from optional physical-input validation.
Smooth and axis-regular cases retain their previous profile accuracy.

## Decisions

| Route | Removed for rho | Retained for rho | Evidence |
| --- | --- | --- | --- |
| PF | final generic `FFn_psin` even-axis fit | `psin_r` limit | The authoritative radial FF source is recovered to roundoff; the old fit produced axis shape errors up to order unity in perturbed cases. |
| PP | final generic `FFn_psin` even-axis fit | `psin_r` limit | The fit is not part of the PP closure and is redundant for smooth data; removing it preserves the discrete PP relation for irregular but finite inputs. |
| PI | final generic `FFn_psin` fit | `psin_r` limit, axis reconstruction of `dItor/drho`, and current-primitive floor | Removing the FFn fit reduces the pressure-perturbed current-derivative closure error from about `6.5e-2` to `5e-6`. Removing the derivative limit instead worsens the smooth uniform F error from about `1.3e-3` to `1.3e-1`. |
| PJ1 | final generic `FFn_psin` fit and direct jtor even-axis rewrite | `psin_r` limit and enclosed-current primitive floor | Exact pointwise current closure improves from `3.2e-7` to roundoff for the smooth case and from `1.54e1` to order `1e-15` for the pressure-irregular stress case. |
| PJ2 | final generic `FFn_psin` even-axis fit | `psin_r` limit | The final fit is numerically redundant and is not part of the optimized-F closure. |
| PJ3 | final generic `FFn_psin` even-axis fit | `psin_r` limit | PJ3 shares the optimized-F closure with PJ2; the same ablation leaves results and convergence unchanged. |
| PQ | final generic `FFn_psin` even-axis fit | `psin_r` limit and the zero-value limit of `d(F**2)/drho` | Directly deleting all treatment raises the smooth uniform F-profile RMS error from `1.7e-3` to `1.5e-1`. Applying the removable limit to the odd numerator instead retains the `1.7e-3` error while avoiding a post-closure fit of `FFn_psin`. |

The signed floor on current primitives is retained in both coordinate modes.
It protects residual evaluations whose nonlinear trial state temporarily
crosses the physical current-primitive domain; it is not a final-profile parity
repair and is therefore outside this ablation.

## Why `psin_r` Is Different

`psin_r` is both the Jacobian of the normalized-flux coordinate and a denominator
in every route. On the magnetic axis it is proportional to rho, so direct
spectral differentiation is poorly conditioned at the innermost open Gauss
nodes. VEQPy reconstructs the smooth ratio `psin_r/rho` from the first two
samples outside `rho=0.05`, extrapolates it inward as a linear function of
`rho**2`, and then applies a small positive domain floor.

Removing the extrapolation while retaining only the floor reduced the expanded
axis suite from 68 to 51 successful solves out of 70 and left five non-finite
cases. Even smooth PF and PJ1 cases acquired roughly 6% and 18% `psin_r` errors,
and uniform PJ2/PJ3 cases failed. This is therefore a coordinate-limit closure,
not cosmetic source smoothing.

The positive floor is a residual-domain guard rather than a statement about the
accepted final equilibrium. It remains common to Numba and C++ so nonlinear
trial states have the same protected evaluation domain.

## PQ Quotient Limit

The rho-coordinate PQ route uses the strict q relation to solve a first-order
equation for $Y=F^2$. With $W=K_nL_{n,r}/q$, its local form is

$$
(W+q)Y_r+2W_rY=
-\frac{V_r p_r q}{2\pi^2L_{n,r}},
$$

up to the route's pressure normalization. The magnetic source is then

$$
FF' = \frac{Y_r}{2\alpha_1\alpha_2\psi_{n,r}}.
$$

Both $Y_r$ and $\psi_{n,r}$ vanish linearly at the magnetic axis, so the last
expression is a removable $0/0$ limit. The old implementation first formed the
quotient and then imposed an even fit on `FFn_psin`. The revised implementation
reconstructs $Y_r/\rho$ from the first two samples outside the axis interval,
sets the corresponding odd zero-value limit of $Y_r$, and only then evaluates
the quotient. It therefore regularizes the quantity whose value is known at the
axis, without imposing a separate derivative condition on the final source.

Across `Nr=32, 48, 64`, native Gauss and endpoint-inclusive uniform source
samples, and smooth, pressure-regular, and q-regular perturbations, all 18 PQ
cases remain finite and converge. For the smooth `Nr=32` uniform case, the
F-profile RMS errors are `1.673e-3` with the previous final-source fit,
`1.672e-3` with the numerator limit, and `1.529e-1` when both are omitted. The
q-authority shape error remains approximately `2e-7` in both stable forms.

## PPP Confirmation

The production PPP/PJ1 fixed point was rerun with the final rho policy. It still
converged in four outer iterations. The native VEQ current profile now matches
the materialized PJ1 target pointwise:

| Metric | Previous policy | Final policy |
| --- | ---: | ---: |
| unweighted current relative L2 | `9.49e-2` | `3.04e-16` |
| rho-integrated current relative L2 | `2.48e-2` | `3.13e-16` |
| innermost current defect | `7.01e5 A/m2` | `0.0 A/m2` |
| innermost-current deviation from an off-axis rho-squared fit | `4.76e-1` | `5.58e-6` |

The pressure derivative supplied to this historical PPP case remains visibly
irregular near the axis. Under the final policy, the mathematically required
irregularity stays in `Pn_psin` and `FFn_psin`; their current contributions
cancel exactly, and the authoritative jtor profile remains smooth. The old
policy instead smoothed `FFn_psin` alone after closure and manufactured a current
spike.

## Reproduction

Run the complete matrix with:

```bash
python benchmarks/source_axis_policies.py \
  --nr 32 --nr 48 --nr 64 \
  --perturbation smooth \
  --perturbation pressure-regular \
  --perturbation pressure-irregular \
  --perturbation driver-regular \
  --output benchmarks/results/source_axis_policy/rho_grid_refinement.json
```

Benchmark result files are intentionally ignored by Git. This document records
the accepted numerical conclusions; the script is the reproducible source of
the matrix.
