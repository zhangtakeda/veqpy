# Source Coordinates and Operator Ownership

VEQ uses one fixed Gauss--Legendre grid in geometric `r` for the
Grad--Shafranov residual, quadrature, differentiation, geometry, and profile
bases. `KernelTopology.coordinate` does not replace that operator coordinate.
It states which coordinate parameterizes the supplied source profiles.

This distinction separates three questions that otherwise look similar:

1. where the GS equations are discretized;
2. which coordinate labels the input source samples;
3. which nonlinear layer owns the map from geometric `r` to that source
   coordinate.

## Current Ownership

| source coordinate | source nodes | coordinate-map owner |
| --- | --- | --- |
| `r` | `uniform` | fixed interpolation onto the operator grid |
| `r` | `explicit` | retained arbitrary nodes; fixed query onto the operator grid |
| `r` | `grid` | already materialized on the operator grid |
| `psin` | `uniform` or `explicit`, PF/PP/PI/PJ1/PQ | outer optimized `psin` profile |
| `psin` | `uniform` or `explicit`, PJ2/PJ3 | outer optimized `F` plus a bounded local source-query loop |
| `psin` | `grid` | expert input already materialized at operator nodes |
| `rho` | `uniform`, `explicit`, or `grid` | deterministic source-local closure |

Thus all routes are r-grid discretizations, but not every source coordinate
uses the same nonlinear ownership. In particular, `psin/grid` does not mean
that an arbitrary poloidal-flux grid becomes the GS grid. It means that the
caller has already supplied derivative or value data at the operator nodes.

The retained legacy PJ2/PJ3 `psin/uniform` query loop uses the same maximum
coordinate change of `1e-10` and 16-iteration cap in both backends. The tighter
legacy threshold is retained because Cxx and Numba use different interpolation
implementations and require it for backend parity; new native `rho`
closures use the standard internal `1e-6` physics tolerance. The Numba path now
reports reaching the cap as an error instead of silently publishing its last
iterate. The legacy Cxx loop still has no public convergence-status channel;
this is recorded as a backend limitation rather than hidden behind a parity
claim. Native `rho` is Numba-only.

## Why `psin` Was Not Moved Entirely Inside the Source Stage

An experimental implementation removed the outer `psin` or `F` profile and
cold-started a local coordinate map from

```text
psin_0(r) = r**2,
dpsin_0/dr = 2*r.
```

PF, PP, PI, PJ1, and PQ used a local `(psin, psin_r)` Picard map. PJ2 and PJ3
used one joint `(psin, psin_r, u, C)` map, where

```text
u = log(F**2 / F_edge**2),
C = Kn*dpsi/dr.
```

The candidate used a fixed `1e-6` tolerance, bounded iterations, no warm state,
and under-relaxation scans. Qualification had two separate stages.

### Fixed-equilibrium source map

[`psin_internal_closure_experiment.py`](../../benchmarks/psin_internal_closure_experiment.py)
holds the converged outer unknown vector and geometry fixed and scans the local
coordinate map at `Nr=32`, `Nt=16` over all 27 legal route/constraint pairs.

| relaxation | converged | iteration range | median iterations |
| ---: | ---: | ---: | ---: |
| `1.0` | 27/27 | 6--18 | 8 |
| `0.5` | 27/27 | 20--41 | 22 |
| `0.2` | 16/27 | 56--64 | 61.5 |

At relaxation `1.0`, route iteration counts are PF 18, PP 10--11, PI 15,
PJ1 6, PJ2 7, PJ3 6, and PQ 8. This proves that the map is locally usable near
the qualified reference state. It does not qualify the coupled nonlinear
solve.

### Complete nonlinear solve

The production-shaped prototype was then exercised at `Nr=16` and `Nr=32`,
with uniform and grid sources, all legal constraints, and Powell/Hybr and LM
outer solves. It failed the production gate:

- PP with uniform source samples failed every constraint under both tested
  outer-solver families. Its axis behavior is naturally represented through
  the existing `sqrt_psin` parameterization; eliminating that representation
  makes the coupled map poorly conditioned.
- PI frequently required more than 32 source iterations, so a source-local
  replacement would make each outer residual both expensive and fragile.
- PJ2's joint local state often converged in 23--24 iterations, but complete
  solves still stagnated or failed for several `Ip` and unconstrained cases.
- PJ3 was better conditioned than PJ2 but remained solver- and
  constraint-dependent.
- PF, PJ1, and PQ were the most reliable candidates, but route-specific
  ownership would make the public coordinate semantics less uniform without
  solving the difficult routes.

The rejected implementation also required monotonicity floors for nonphysical
outer trial states. Those floors changed the residual outside the accepted
physical domain and therefore were not retained.

The decision is to preserve the existing outer `psin`/`F` ownership and keep
native `rho` as the source-local coordinate closure. A locally convergent
Picard map is not sufficient evidence for replacing an outer nonlinear
unknown.

## Arbitrary Input Grids

`nodes="explicit"` is the native arbitrary-grid contract. Its node count and
normalized positions are both runtime `KernelSource` data; consequently
`KernelTopology.sample_count` must be `None`. Pressure, route driver, and
`source_nodes` share a runtime length and remain on those nodes in the Kernel
source plan. Only grid-sized query and physical work arrays are topology-sized.
The nodes must be finite, strictly increasing, and include 0 and 1.

At case binding, Numba constructs a shape-preserving PCHIP representation of
each retained source profile. It does **not** first project that representation
onto the geometric-r operator grid:

- `r/explicit` queries it once at the fixed operator nodes;
- `psin/explicit` queries it from the current outer `psin(r)` state, including
  every bounded PJ2/PJ3 source-query iteration;
- `rho/explicit` queries it again in every coordinate or joint-physics
  fixed-point iteration.

The dynamic interpolation kernel evaluates precomputed coefficients into
preallocated grid-sized arrays. There are no per-iteration SciPy objects or
temporary arrays. Retaining the native representation matters most for a
moving source coordinate: a one-time adapter projection would discard source
resolution before the coordinate is known and subsequent interpolation would
operate on the wrong representation.

### Interpolation policy

The original `uniform` contract has endpoint-inclusive, equally spaced source
nodes. Its default runtime interpolant is an eight-point local barycentric
polynomial; registered linear, quadratic, cubic, and not-a-knot alternatives
remain available. For PP/psin/uniform only, the public uniform parameter is
`sqrt(psin)`, so the actual source nodes are squared before evaluation.

The earlier arbitrary-State prototype first used PCHIP to project external
nodes onto this uniform representation, after which Kernel interpolated the
uniform samples again. That two-stage map discarded native resolution and
could combine two different interpolants.

The production `explicit` contract instead constructs one shape-preserving
PCHIP directly on the caller's arbitrary normalized nodes. Each internal query
locates one source interval and evaluates its four local coefficients by
Horner form. Pressure and driver share interval lookup, and the fused
psin/`rho` loops repeat this direct native query without an intermediate
uniform or GS-grid representation. Fusion changes dispatch and loop placement,
not these interpolation polynomials.

For `explicit` input in every coordinate, a pressure primitive and its
derivative share one representation. Static `r` differentiates the retained
PCHIP at operator nodes; dynamic `psin` and `rho` differentiate it at each
current nonlinear query. VEQ therefore does not apply a global differentiation
matrix to clustered native nodes and then interpolate the resulting derivative
as a second profile. PP and PI apply the same rule to `psi_r` and cumulative
`itor`: each value and its radial derivative are evaluated from one coefficient
table in one interval lookup. PP then applies its existing odd axis extension
to `psi_r` and differentiates that extension analytically in the affected
region; the outer region retains the PCHIP derivative. Only routes whose
physics consumes a derivative materialize it. VEQ does not manufacture unused
derivative arrays for PF, PJ1, PJ2, PJ3, or PQ.

`nodes="grid"` remains a distinct expert contract: its arrays already carry
values at operator-node indices. It is not a synonym for arbitrary source
nodes. The Cxx backend currently rejects `explicit`; the qualified native path
is Numba-only.

[`explicit_source_benchmark.py`](../../benchmarks/explicit_source_benchmark.py)
uses a PJ1/`rho` case with a narrow edge-localized source feature. At the
same 17-sample count, an outer-region-refined explicit grid reduces the dense
pressure-source interpolation error from 27.1% to 4.22%. Relative to an
801-sample solve, the final pressure, safety-factor, and toroidal-current
profile errors change from 1.58%, 0.497%, and 9.36% to 0.0835%, 0.0698%, and
1.00%, respectively. After source-loop fusion, batched timing of the
already-bound nonlinear hot loop gives 0.0152 ms for the uniform source and
0.0151 ms for the explicit source. The all-route matrix in
[`native_source_performance.py`](../../benchmarks/native_source_performance.py)
places explicit/uniform ratios at 0.987--1.011 for r, 0.969--1.013 for psin,
and 0.966--0.994 for `rho`; there is no resolved native-grid hot-loop
penalty. Runtime case binding distinguishes numerical changes from object
identity. A fresh but numerically equal Source snapshot is detected by exact
array comparison and reuses validated lowering, interpolation coefficients,
and bound runners. A changed Source with the same Boundary refreshes only
source-owned arrays and bindings, not profile/Fourier/Stage-A state.

For the `Nr=24`, `Nt=12`, 51-sample PJ1/r/IP audit, fresh equivalent snapshot
binding is 0.0055 ms for uniform, 0.0060 ms for grid, and 0.0086 ms for explicit.
A genuinely changed pressure snapshot takes 0.168, 0.0876, and 0.243 ms,
respectively. Before this split the corresponding fresh-snapshot costs were
0.457, 0.369, and 0.530 ms. The equivalence path is therefore 61--82x faster;
even changed snapshots are 2.2--4.2x faster.

An experiment also placed the explicit node array conceptually in Topology and
removed runtime node validation/copying. It could not precompute complete PCHIP
weights because PCHIP slopes and coefficients depend nonlinearly on source
values, while dynamic psin/`rho` queries do not have fixed interpolation
weights. The measured setup upper bound was only a 1.030--1.034x speedup
(2.6--3.1% saving) for 17--2001 nodes, below the 10--20% retention threshold.
That design was therefore rejected: arbitrary node positions remain runtime
KernelSource data and do not enter Topology identity.

[`explicit_derivative_experiment.py`](../../benchmarks/explicit_derivative_experiment.py)
qualifies the value/derivative rule with 17 edge-refined native samples against
an 801-sample PI/r reference at `Nr=24`, `Nt=12`. For an edge-localized
pressure/current-gradient case, the old native-global-D then PCHIP pressure
path has an 807x maximum relative derivative defect, while differentiating the
retained PCHIP gives 2.80%. For cumulative current, differentiating the
PCHIP-remapped Gauss values gives 40.6%, while the retained PCHIP derivative
gives 2.80%. The final pressure and safety-factor errors fall from 695% and
15.1% to 0.183% and 0.878%; `psin` falls from 2.08% to 0.0386%. A fresh
equivalent source now uses the exact-equivalence fast path described above,
while the already bound residual remains about 0.0086 ms. The experiment's `legacy_two_step`
timing uses the current runner with legacy numerical fields and therefore is
not an old-versus-new hot-kernel timing claim.

[`pp_explicit_derivative_experiment.py`](../../benchmarks/pp_explicit_derivative_experiment.py)
repeats the qualification for PP under `none`, `Ip`, `beta`, and combined
constraints. With the same 17/801 source comparison, differentiating remapped
Gauss values gives 122% `psin_rr` error; differentiating the retained PCHIP and
the axis extension gives 69.4%. The final `jtor` error changes from 82.8--97.6%
to 52.3--57.3%, while `q` changes from 6.64--11.3% to 6.25--11.1%. The
remaining large second-derivative/current error is source-resolution error from
asking 17 samples to resolve the narrow feature, not a second interpolation
stage. A smooth polynomial-profile control reduces `psin_rr` error from about
1.17% to 0.22% and `jtor` from about 0.32% to 0.067%. Across all constraints,
the bound hot residual remains statistically unchanged at about 0.008 ms.
