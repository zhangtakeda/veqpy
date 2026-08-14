# Kernel

Source-coordinate semantics and ownership are specified separately in
[`source-coordinate-ownership.md`](source-coordinate-ownership.md). All Kernel
routes retain geometric `r` as the GS operator grid; `coordinate` identifies
the parameterization of source data rather than replacing the operator grid.

The coordinate vocabulary is intentionally unambiguous: geometric VEQ `rho`
from the earlier API is now `r`, while the earlier `sqrt-phi` source coordinate
is now `rho = sqrt(Phi_N)`. There are no compatibility aliases for either old
name, because accepting them would make a valid-looking profile carry the wrong
derivative semantics.

The public runtime entrypoint is the Kernel API:

```python
from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
```

`KernelTopology` fixes the active packed coefficient counts, grid size, source
route, coordinate system, node semantics, source constraints, and capacity
limits. Active count fields determine the public packed vector size. `L_max`,
`M_max`, and `K_max` are setup/capacity limits: explicit values may exceed the
minimal active-count requirement, but may not be smaller. For
`nodes="explicit"`, source count and node positions belong to `KernelSource`
and `KernelTopology.sample_count` is `None`. `KernelBoundary` and
`KernelSource` carry per-case physical inputs. `KernelConfig` carries nonlinear
solve policy. `Kernel.solve(...)` returns a shared `SolveResult`, and
`Kernel.build_equilibrium(grid=...)` materializes the current `Equilibrium`
snapshot directly on an optional output grid, avoiding a separate public
`equilibrium.resample(grid)` step.
For sine Fourier data, Kernel-level public inputs are s1-started:
`KernelTopology.s_counts=(n1, n2, ...)` and `KernelBoundary.s_offsets=(s1, s2, ...)`.
The runtime adds the structural s0=0 slot before backend calls.
`KernelBoundary` accepts either explicit parameterized geometry
(`a`, `R0`, `Z0`, `B0`, `ka`, `c_offsets`, `s_offsets`) or raw LCFS point arrays
(`R_boundary`, `Z_boundary`) plus `c_order`/`s_order`. Raw point boundaries are
stored without mutating the input arrays and are materialized into Fourier
coefficients when a Kernel backend consumes the boundary. The optional
`method` selects the fitter: `qr` is weighted QR, `gnqr` is weighted QR plus two
fixed-geometry Gauss-Newton steps, and `least-square` is the bounded full R/Z
least-squares fit initialized from the analytic `R0/Z0/a/ka` estimates. The
default is `gnqr`. Fit diagnostics are recorded on `fit_*` metadata after
materialization; the original frozen `KernelBoundary` is not updated in place.
Use `raw_boundary.fit(backend="numba")` to explicitly fit once and return an
equivalent parameterized `KernelBoundary` carrying `fit_rms`,
`fit_max_curve_error`, `fit_c_order`, `fit_s_order`, and `fit_method` metadata.
The default manual-fit backend is `numba`; `numpy` is available for the shared
Python fitter, and `cxx` uses the optional native fitter when its toolchain is
available. The returned boundary can be reused in later solves without fitting
the raw points again.

The direct Numba implementation is a private Kernel backend. Its internal runtime
owns packed layout metadata, source materialization, residual workspaces, and
equilibrium snapshot assembly, but those details are not separate public objects.
`KernelRecipe.backend` selects the backend implementation; user code continues
to call the same `Kernel` methods.

Public source inputs stay raw. Every `KernelSource` requires exactly one pressure
representation:

- `p=p`: absolute pressure samples; `p0` is derived and must not be supplied.
- one coordinate-explicit derivative plus the optional LCFS pressure `p0`,
  which defaults to zero: `P_r=dP/dr`, `P_rho=dP/drho`, or
  `P_psin=dP/dpsin`. The field must match `KernelTopology.coordinate`; the
  Kernel rejects a mismatched derivative before backend execution and applies
  any later chain rule internally.

It also requires exactly one driver selected by the topology route:

| route | required driver |
| --- | --- |
| `PF` | coordinate-matched `FF_r`, `FF_rho`, or `FF_psin` |
| `PP` | `psi_r` |
| `PI` | `itor` |
| `PJ1` | `jtor` |
| `PJ2` | `jpara` |
| `PJ3` | `jtotal` |
| `PQ` | `q` |

For example, a PQ case can be constructed as either
`KernelSource(p=p, q=q, beta=beta)` or
`KernelSource(P_r=P_r, p0=p0, q=q, beta=beta)` for an `r` topology. Supplying both pressure
representations, neither pressure representation, more than one driver, no
driver, or a driver that does not match `KernelTopology.route` is rejected
before backend execution. The old generic `pprime`, `ffprime`, and
`heat_profile`/`current_profile` keywords are not accepted.

For uniform `p` samples, the final sample is the LCFS pressure and the
differentiation matrix acts in the selected source coordinate. The PP
`sqrt_psin` parameterization applies the chain rule without differentiating on
an ill-conditioned squared node grid. For Legendre grid samples in `r`, the
runtime uses the grid differentiation matrix and interpolates `p` to `r=1`
because Gauss-Legendre nodes exclude the edge. `p` is intentionally rejected for
`coordinate="psin", nodes="grid"`: those nodes are fixed in `r`, while
`psin(r)` is part of the unknown equilibrium. Use explicit `P_psin` for that
topology or switch to uniform psin samples.

`P_psin` and `FF_psin` are not the conventional Grad-Shafranov `P_psi` and
`FF_psi`. The materialized equilibrium obeys `psi = alpha2 * psin`,
`P_psin = alpha2 * P_psi`, and `FF_psin = alpha2 * FF_psi`. The PF backend
closes the required source scale in an allocation-free scalar iteration. Its
fixed equal-weight Picard average removes
the reciprocal PF map's otherwise neutral two-cycle; it is not a runtime solver
parameter. PF closes the magnitude of `alpha2` because `psin` remains oriented
from axis to edge; an `Ip` constraint retains ownership of the physical flux
direction. This keeps both public derivatives tied to
their stated normalized coordinate without adding a profile unknown to the
outer nonlinear solve.

For arbitrary `explicit` samples, `p` and `dp/dcoordinate` are evaluated from
one retained shape-preserving PCHIP. Static `r` evaluates the derivative on
the operator grid; dynamic `psin` and `rho` evaluate it at the current
nonlinear query. PP likewise evaluates `psi_r` and its outer
radial derivative from one PCHIP, while deriving the inner derivative
analytically from its existing odd axis extension. PI evaluates `itor` and
`dItor/dr` from one PCHIP instead of differentiating remapped Gauss-grid
values. These paths preserve the mathematical identity between a supplied
field and the derivative consumed by the route. Coefficients and derivative
arrays are built once per numerical source snapshot. Dynamic-coordinate routes
already query their retained sources in the nonlinear residual loop, so
differentiating that same local cubic adds no extra interval search.

Source lowering preserves every finite coordinate-explicit pressure derivative
and route-driver sample exactly apart from documented unit scaling. In
particular, PF retains `FF_r`, `FF_rho`, or `FF_psin` on its native source
coordinate before applying the required chain rule. It does not impose magnetic-axis parity or
rewrite an inner radial interval. Route closures may still regularize their own
derived quantities before an internal division; those backend-local limits
never modify the public source arrays.

For `coordinate="r"`, the route stage follows a stricter ownership rule. No
route applies a generic even-axis fit to `FFn_psin` after closing its defining
equation. PI still reconstructs the zero-value axis limit of `dItor/dr`,
because differentiating the current primitive is ill-conditioned there. PQ
solves a first-order equation for `Y = F**2`; it reconstructs the odd
zero-value limit of `dY/dr` before dividing by `psin_r`, instead of fitting
the resulting `FFn_psin`. Source-owned algebraic routes reconstruct and floor
`psin_r` before using it as a flux coordinate or denominator. Strict r
PJ2/PJ3 instead construct the axis-anchored primitive
`C=K_n*dpsi/dr` directly; they do not post-fit or floor that accepted closure.
These are limits of derived coordinates or odd differentiated quantities, not
edits to the authoritative source profile and not generic zero-derivative
parity constraints.

The route-by-route ablation and convergence evidence is recorded in
[`source-axis-policy.md`](source-axis-policy.md).

For `coordinate="rho"`, source samples are functions of the normalized
toroidal-flux radius `s = sqrt(Phi_N) = rho`, while geometric `r`
remains the operator coordinate. The Numba source stage cold-starts from the
deterministic toroidal-flux map obtained with `F=F_edge` and the current
`Ln_r`, remaps the source profiles, applies the derivative chain rule, runs the
corresponding r closure, and rebuilds `s` from `F*Ln_r` until a bounded local
fixed point reaches the internal, non-configurable tolerance `1e-6`. Invalid
trial geometry falls back to `s=r` rather than shrinking the nonlinear
solver's previous trial-state domain.
PJ2/PJ3 advance `(s, ds/dr, u, C)` in one joint Picard map rather than
nesting their current closure inside a coordinate loop. This coordinate adds
neither `psin` nor `F`
coefficients to the outer packed state. Its equations, node semantics,
convergence scan, and timing evidence are recorded in
[`rho-source-closure.md`](rho-source-closure.md).

`KernelSource.p0` is the pressure at the LCFS in Pa; `Ip` and `beta` are the
optional global constraints. The runtime reconstructs the complete pressure
from the selected coordinate derivative and `p0`; a beta constraint scales both pieces,
including `p0`, by one common factor. After route constraints have been applied, the source stage
fixes the alpha gauge from `max(abs(mu0*p))` and inversely rescales `Pn` and
`FFn`, preserving the physical Grad-Shafranov sources. Consequently a zero
`P_r`, `P_rho`, or `P_psin` with `p0!=0` is a valid constant-pressure input,
while a completely zero
pressure remains rejected.

Numba and Cxx share canonical lowering for
`(scaled_pprime, scaled_driver, scaled_p0, scaled_Ip, beta)`. Absolute `p`
therefore remains one public input representation rather than a second
backend-specific source mode. For the Numba-only `explicit` contract, the same
lowering additionally retains the scaled pressure primitive so the runtime can
differentiate its PCHIP representation at the current source-coordinate query.
Route-dependent scaling and internal materialized source arrays are backend
runtime details, not user-facing data fields. Native `rho` coordinate
closure is currently Numba-only; requesting it from Cxx raises an explicit
unsupported-backend error.

PJ2 and PJ3 share the same F-coupled current closure but not the same public
physics. PJ2 accepts
`jpara = <J·B> / (F <R^-2>)`; PJ3 accepts the IMAS convention
`jtotal = <J·B> / B0`. PJ3 evaluates

```text
jpara = B0 * jtotal / (F * gm1),    gm1 = <R^-2> = (2*pi)^2 * Ln_r / V_r
```

inside every source evaluation using the current geometry and F profile. It is
therefore not a setup-time alias or a frozen conversion.

For `coordinate="r"`, `F_count=0` selects the Numba strict closure. The source
stage solves the coupled radial equations for enclosed current and
`u=log(F**2/F_edge**2)`, certifies a dimensionless fixed-point defect below
the internal, non-configurable tolerance `1e-6`, and publishes F and normalized
flux without adding F coefficients to
the outer Grad-Shafranov solve. A positive `F_count` retains the optimized-F
approximation for controlled comparisons and Cxx parity. `coordinate="psin"`
still requires positive `F_count`, because those samples must be remapped
through the evolving flux coordinate. The strict equations, iteration budgets,
and benchmark evidence are recorded in
[`pj23-strict-closure.md`](pj23-strict-closure.md).

## Radial Endpoints

The default Legendre grid is open: its first and last nodes are interior to
``(0, 1)`` and therefore are not the magnetic axis or LCFS. The shared grid
contract exposes three distinct operations:

- ``grid.axis_eval(profile)`` evaluates a nodal field at ``r=0``;
- ``grid.edge_eval(profile)`` evaluates it at ``r=1``;
- ``grid.full_integral(profile)`` integrates it over the complete ``[0, 1]``
  interval.

Numba and Cxx source routes use the corresponding precomputed endpoint weights
and full quadrature when applying ``Ip`` constraints. A raw ``profile[0]`` or
``profile[-1]`` is only an endpoint value when the profile's own public sampling
contract is explicitly endpoint-inclusive.

The solved ``psin`` array likewise stores values at the internal radial nodes;
it is not overwritten to begin at zero and end at one. Its normalization is
set by the full integral of ``psin_r``. Consequently ``axis_eval(psin)=0`` and
``edge_eval(psin)=1`` while, on an open grid, ``0 < psin[0] < psin[-1] < 1``.

``Equilibrium.resample()`` evaluates a native polynomial-collocation snapshot
at the target nodes, including true axis and LCFS values when the target grid
owns them. The IMAS coordinate ``rho`` is normalized by the complete
toroidal-flux integral rather than the first and last stored samples. GEQDSK
export similarly materializes explicit axis and LCFS surfaces before building
its boundary, one-dimensional profiles, and two-dimensional flux mesh.

## Solve Flow

```python
from veqpy import Kernel, KernelConfig, KernelRecipe

kernel = Kernel(
    topology=topology,
    recipe=KernelRecipe(backend="numba", layout="degree"),
    config=KernelConfig(method="powell"),
)
result = kernel.solve(boundary, source)
equilibrium = kernel.build_equilibrium()
```

`KernelConfig` controls the runtime solve method, residual normalization,
initial-state policy, continuation policy, residual acceptance threshold, and
evaluation limits. `SolveResult` records the final packed state, raw residual,
scaled residual, source `alpha` values, function/iteration counters, success
flag, and elapsed time. `Kernel.jvp(...)` and `Kernel.jacobian(...)` are
finite-difference numerical queries over the same residual runtime.

Warm continuation is handle-local: after a solve, the next `Kernel.solve(...)`
can reuse the previous solution when the continuation policy is warm. Use
`kernel.clear()` to drop the stored result and history.

An explicit `x0=` overrides both warm continuation and the configured cold
initial-state policy. It may be a packed array-like, a previous `SolveResult`,
or a complete dictionary of active profile coefficient arrays:

```python
result = kernel.solve(
    boundary,
    source,
    x0={"h": h_coeff, "k": k_coeff, "s1": s1_coeff, "psin": psin_coeff},
)
```

Named dictionaries are encoded according to `KernelRecipe.layout`; every active
profile must be supplied and inactive or unknown names are rejected. Explicit
states are per-call runtime data and do not enter native artifact identity.

## Topology Variants

`Kernel.variant(...)` is for multi-topology count sweeps on Numba-backed
handles: within fixed setup and capacity limits, it switches the active
topology counts without rebuilding fixed grid, capacity workspace, and compiled
runtime state. It is an in-place state switch that returns the same `Kernel`
object and constructs a fresh immutable `KernelTopology`; previously saved
topology objects are not mutated.

Only active count fields may be changed:

```python
kernel.variant(
    h_count=6,
    c_counts=(2, 1),
)
```

Omitted arguments and explicit `None` inherit the current active count.
`variant()` does not change fixed setup or capacity fields such as
`Nr/Nt/route/coordinate/nodes/sample_count/constraint`,
`quadrature`, `calculus`, `L_max`, `M_max`, or `K_max`. New counts must fit the
current capacity limits: radial counts require `count <= L_max + 1`, cosine
orders require `order <= M_max`, and sine orders use the public s1-started
indexing.

Successful variants clear the current result, last runtime case, and prepare
cache so old packed solutions cannot warm-start a different active layout.
`history` is retained as a per-handle solve log and can contain entries from
different active topologies and `x_size` values. `Kernel.variant()` is not
thread-safe and must not run concurrently with solve, residual, derivative,
prepare, or equilibrium snapshot calls.

## Pareto Topology Reduction

`Kernel.pareto(...)` is a Numba-only evaluator for intentionally over-large
active-count topologies. The active topology is the high-parameter reference.
Callers pass explicit reduced count-only candidates, and `pareto()` solves those
candidates through the existing Numba variant runtime. It returns a
`ParetoResult` containing the reference, evaluated samples, and frontier. The
method restores the caller-visible Kernel topology, result, history, and last
runtime case before returning.

```python
reference = kernel.solve(boundary, source)

pareto = kernel.pareto(
    boundary,
    source,
    candidates=[
        {"h": 6, "v": 4, "psin": 3},
        kernel.topology,  # the reference topology is ignored if supplied
    ],
    config=KernelConfig(method="powell"),
    reference=reference,       # optional; omitted means solve the reference first
    target="complexity",      # "counts", "time", or "complexity"
    metric="rms",             # "rms" or "max"
)
```

Candidate entries may be `KernelTopology` instances, internal count signatures,
or mappings such as `{"h": 4, "psin": 2, "c0": 1}` and
`{"h_count": 4, "c_counts": (1,)}`. Candidates are canonicalized with the same
capacity and source-family rules as `Kernel.variant(...)`; duplicate candidates
and the reference topology are ignored.

Shape error is measured directly on the Kernel solve grid using only the
major-radius surface `R`, without constructing candidate `Equilibrium`
snapshots. `metric="rms"` computes `sqrt(mean((R_candidate - R_ref)**2))`;
`metric="max"` computes `max(abs(R_candidate - R_ref))`. Both values are in
meters and are not normalized.

Each `ParetoSample` reports three cost columns whose names match `target`:
`counts` is `topology.x_size`, `time` is `SolveResult.elapsed_ms`, and
`complexity` is the fixed integer score
`nfev*Nx + jvp_evaluations*Nx**2 + jacobian_component_evaluations*Nx**2 + linear_iterations*Nx**2`,
where `Nx=counts`. `samples` contains evaluated reduced candidates only;
`reference` stores the high-parameter reference; `frontier` includes the
reference as the exact endpoint. Threshold-based choices are caller-side
post-processing over `frontier`, not part of the `ParetoResult` contract.
