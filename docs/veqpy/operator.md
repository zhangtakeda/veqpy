# Operator

`Operator` is the numerical center of one fixed-boundary equilibrium solve. It
turns a `Problem`, a `Grid`, and a packed coefficient vector $x$ into the
finite-dimensional Grad--Shafranov residual used by `Solver`. In physical terms,
it updates the flux-surface shape, constructs the source profiles implied by the
chosen source route, and projects the strong-form residual onto the active
coefficient basis.

The relevant source files mainly live in `veqpy/operator/`, `veqpy/layout/`, `veqpy/workspace/`, and `veqpy/engine/`.

## Problem

`Problem` describes one fixed-boundary solve input: source route, source coordinate, node semantics, active profile lengths, boundary, heat/current-related inputs, and optional `Ip` or `beta` constraints. It is the public problem-definition type passed into `Operator`.

`Problem.active_profiles` maps active profile names to coefficient counts.
Profiles absent from this mapping are passive. `Problem` does not carry initial
coefficient values; callers that have coefficient arrays pass them to
`Operator.pack_coefficients(...)` after the operator topology is built.

`route`, `coordinate`, and `nodes` jointly form the source route key:

```python
(route, coordinate, nodes)
```

This key selects the source kernel and the interpretation of the input arrays. `heat_input` and `current_input` remain one-dimensional data; their physical meaning is determined by the selected route.

`Problem` keeps setup arrays and constraints in the raw user-facing convention.
When an `Operator` is built, `SourcePlan` validates those magnitudes and
materializes plan-ready source inputs. `scaled_heat` is always pressure-like
setup data after `mu0` scaling. `scaled_Ip` stores the `mu0 * Ip` constraint.
`scaled_current` is the plan-ready current/source driver; for current-profile
routes (`PI`, `PJ1`, and `PJ2`) it is also `mu0`-scaled, while in other routes it
remains a normalized or field-derived driver such as `FF'`, `psin_r`, or `q`.
Inputs with magnitudes outside the expected setup ranges are rejected while
building the source plan.

## Source Routes

All routes produce the same root fields for residual assembly:

- `psin`, `psin_r`, and `psin_rr`, the normalized flux coordinate and its radial
  derivatives on the operator grid;
- `Pn_psin`, the normalized pressure derivative with respect to `psin`;
- `FFn_psin`, the normalized derivative of the toroidal-field function product;
- `alpha1` and `alpha2`, scalar source normalizations chosen from the route and
  any global constraints.

The route only changes how these fields are reconstructed from one-dimensional
inputs:

| route | `heat_input` meaning                      | `current_input` meaning                                              |
| ----- | ----------------------------------------- | -------------------------------------------------------------------- |
| `PF`  | pressure-gradient data (`P_r` or `P_psi`) | toroidal-field source (`FF'`)                                        |
| `PP`  | pressure-gradient data                    | normalized flux-gradient driver `psin_r`                             |
| `PI`  | pressure-gradient data                    | enclosed toroidal current `I_tor`                                    |
| `PJ1` | pressure-gradient data                    | toroidal current density `j_tor`                                     |
| `PJ2` | pressure-gradient data                    | parallel current density `j_parallel`, using the current `F` profile |
| `PQ`  | pressure-gradient data                    | safety factor `q`                                                    |

`coordinate="rho"` means the source samples are parameterized by the radial
label. `coordinate="psin"` means the samples are parameterized by normalized
flux. `nodes="grid"` means the arrays already live on the operator radial grid;
`nodes="uniform"` means they are remapped from a uniform source axis. The
`PP/psin/uniform` route uses a `sqrt(psin)` uniform source parameterization so
edge-resolved input can be sampled more evenly.

For non-`PJ2` `psin/uniform` routes (`PF`, `PP`, `PI`, `PJ1`, and `PQ`), `psin`
must be an active optimized profile because the source samples need the current
flux coordinate during every residual evaluation. `psin/grid` inputs are already
materialized on the operator radial nodes, so they do not own an active `psin`
profile. Active `F` is required only by `PJ2`, where the parallel-current source
uses the current optimized toroidal-field profile; active `F` and active `psin`
are mutually exclusive. `PQ` is also strict about the toroidal-field profile: it
solves the `F` or `F^2` profile from `q` and the edge value `R0 * B0`, so an
active `F` profile is not accepted.

## Constraints and Scaling

Without a global constraint, the source route chooses `alpha1` and `alpha2`
from the radial normalization of the reconstructed source profiles. When `Ip`
is present, the route instead chooses the scale that makes the integrated
toroidal current match the requested current. When `beta` is present, the route
chooses the pressure scale from a volume-weighted pressure integral and the
reference field `B0`. Most routes can combine `Ip` and `beta`; `PF` rejects that
combination because both constraints would overdetermine the two source drivers
available in that route.

## Packed Layout

The packed layout defines where each coefficient lives in the optimization vector $x$. The current profile family includes shape profiles `h`, `v`, `k`, `c0`, `c`, `s`, and source/flux-related profiles `psin`, `F`. Only active profiles enter the packed vector.

The default layout is degree-first: all active profiles contribute their low-order coefficients first, and higher degrees follow. This gives residual blocks, profile refresh, and solver initial values a shared index semantics.

The shape profiles determine the continuous fixed-boundary surface family.
`h`, `v`, and `k` carry low-order radial shaping, while `c*` and `s*` carry the
Fourier harmonics. Optional `psin` and `F` profiles become active when the problem
includes their names and lengths; route validation decides whether that ownership is
physically meaningful for the selected source model.

## Residual Pipeline

When an `Operator` is built, the active profile set, coefficient lengths, source
route, residual blocks, and fixed grid are frozen into one solve topology. If
the active profile set, coefficient lengths, or route topology changes, the
operator should be rebuilt. A replacement problem may reuse the same operator only
when it preserves that packed topology.

One residual call has four main stages:

| Stage    | Role                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------- |
| profile  | Refresh active profiles from packed $x$                                                        |
| geometry | Compute geometry fields and flux-surface averages from shape profiles                          |
| source   | Reconstruct `psin`, pressure, field, and current-related source fields from the selected route |
| residual | Assemble the Grad--Shafranov residual and project it onto each active coefficient block        |

`Operator.__call__(x)` returns the variational/Galerkin residual. Each residual
block corresponds to one active profile block and uses the same basis ordering
as the packed state. This is the primary equation solved by VEQPy.

`residual_collocation(x)` returns the quadrature-scaled pointwise residual used
for collocation polish. It evaluates the local strong-form force-balance
residual on every `(rho, theta)` grid point and returns a vector of length
`Nr * Nt`. This is a diagnostic or post-processing objective; it does not
replace the Galerkin residual as the primary solve definition.

## Runtime Memory Vocabulary

The mutable operator runtime uses a narrow vocabulary so engine bindings remain
explicit without exposing Python workspace objects to Numba kernels.
`*_fields` names mean sampled slabs whose rows or axes carry physical or
numerical samples, such as grid radial tables, geometry surface rows, residual
root rows, and profile value/derivative rows. `*_operators` names are linear
maps such as radial differentiation or accumulation matrices. `*_metadata`
names are layout decisions: route codes, profile ids, block codes, coefficient
index rows, active lengths, and grid size metadata. `*_scratch` buffers are
temporary workspaces with no persistent physical meaning after a kernel call,
while small mutable vectors such as `alpha_state` are state.

Hot-path engine calls receive field slabs, operators, scalar constants, and
metadata directly. Workspace properties such as `GridWorkspace.T` or
`GeometryWorkspace.R_surface` are view accessors for debug and row-contract
documentation; they are not required by the main fused runtime ABI.

## Snapshot Boundary

After the solve, `build_equilibrium(x)` refreshes the runtime with the final solution and writes only snapshot-relevant root fields and shape profiles into `Equilibrium`. Runtime buffers are not transferred into the model object. This boundary keeps the operator in a high-throughput mutable form while making `Equilibrium` a serializable, interpretable physical snapshot.
