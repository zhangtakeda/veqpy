# Model

The `model` layer stores interpretable, serializable physical objects in VEQPy.
It is not a mirror of the Kernel runtime; solve setup lives in
`KernelTopology`, `KernelBoundary`, and `KernelSource`, while solved snapshots
are represented by `Equilibrium`. `Profile` is the serializable parameter object
used by solved shape-profile snapshots. Derived grid geometry and equilibrium
diagnostics are reconstructed lazily through `Reactive` properties. Files store
only root state.

The main source files live in `veqpy/model/`.

## Objects

| Object        | Responsibility                                                                                            |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `Grid`        | Radial/angular discretization, quadrature weights, differentiation/integration matrices, and basis tables |
| `Profile`     | Parameterized representation of a one-dimensional radial profile                                          |
| `Geqdsk`      | GEQDSK data loading, storage, and conversion                                                              |
| `Equilibrium` | Solved continuous equilibrium snapshot and diagnostic interface                                           |

GEQDSK LCFS points remain passive data in `Geqdsk.boundary`. Runtime boundary
parameters and boundary-point fitting belong to `KernelBoundary`, not to a
separate model object.

`Equilibrium.to_geqdsk()` rasterizes the solved closed surfaces inside the LCFS.
Its default exterior solves the current-free Grad-Shafranov equation with zero
flux increment on the LCFS and positive controlled Dirichlet data on the
rectangular box. The boundary control leaves a narrow collar of closed surfaces
near the LCFS and opens the far exterior without joining the `psi_bound` LCFS to
the box. `Geqdsk.boundary` retains the exact, explicitly closed LCFS polygon.
Supplying `psi_outside` explicitly instead requests the legacy scalar exterior
value.

`Profile` represents a one-dimensional radial profile with scale, power,
envelope, offset, and optional Chebyshev coefficients. Its persistent state is
only those root parameters. When a `Grid` is bound, it lazily materializes
`value`, `derivative`, and `second_derivative` on `grid.r`; without a bound
grid those fields are unavailable. Kernel runtime setup lowers active profile
topology to flat arrays; `Profile` remains on the model side for
`Equilibrium.shape_profiles` and other serializable snapshots.

## Equilibrium Snapshot

`Equilibrium` is the main output object. It receives root fields after the solve, including:

- geometry scales: `R0`, `Z0`, `B0`, `a`;
- discretization object: `grid`;
- fixed-boundary and shape profiles: `shape_profiles`;
- flux and source derivatives: `psin`, `psin_r`, `psin_rr`, `FFn_psin`, `Pn_psin`;
- pressure integration constant at the LCFS: `p0`;
- scaling coefficients: `alpha1`, `alpha2`.

These fields are sufficient to reconstruct common physical quantities without
retaining solver-hot-path memory. When the user reads properties such as `R`,
`Z`, `F`, `P`, `q`, `Ip`, `beta_t`, `jtor`, `jpara`, `jtotal`, `jphi`, `Psi`, or
`Phi`, a self-contained Numba kernel materializes a new result and `Reactive`
caches it until a direct dependency changes. The integration grid is explicitly
`(r, theta)`: `r` is VEQ's normalized
geometric radius and `theta` is the poloidal angle. The snapshot also exposes
the IMAS toroidal-flux derivative `Phi_r`, coordinates `rho_tor`, `rho`, and
their VEQ-`r` derivatives, together with the IMAS properties `gm1` through
`gm9`.

Pressure derivatives are named by their actual independent variable:
`P_r=dP/dr`, `P_rho=dP/drho`, and `P_psin=dP/dpsin`. `P_psi` is the conventional
Grad-Shafranov derivative with respect to the unnormalized poloidal flux per
radian `psi = Psi/(2*pi)`; hence `P_psin = alpha2 * P_psi` in VEQ's flux
normalization. It is not an alias for `P_psin`.

The toroidal-field source follows the same naming rule:
`FF_r=F*dF/dr`, `FF_rho=F*dF/drho`, and `FF_psin=F*dF/dpsin`.
`FF_psi=F*dF/dpsi` is the conventional GEQDSK quantity, so
`FF_psin = alpha2 * FF_psi`. The model exposes all four explicitly.

`F` retains the sign of `R0 * B0`. The PJ2 diagnostic `jpara` denotes
`<J·B> / (F <R^-2>)`, while `jtotal` directly exposes the IMAS convention
`j_total = <J·B> / B0`. Internally it uses
`gm1 = <R^-2> = (2π)^2 Ln_r / V_r` and therefore equals
`jpara * F * gm1 / B0`.

## Geometry and Diagnostics

The model layer exposes stable snapshot diagnostics, not every intermediate array used during residual assembly. Geometrically, `Equilibrium` provides flux-surface mapping, Jacobian-related fields, area/volume, and flux-surface-averaged geometry factors. Physically, it provides pressure, toroidal-field function, safety factor, current, and flux diagnostics.

A small set of packed geometry fields is retained because those combinations
have stable meaning for plotting, comparison, and GEQDSK export. Finer local
derivative combinations, residual projection matrices, and backend workspaces
remain in the Kernel runtime layer and do not become public snapshot API.

### IMAS gm geometry

`Equilibrium` follows the IMAS flux-surface-average definitions. Here
`<...>` is the Jacobian-weighted flux-surface average and
`rho_tor = sqrt(Phi / (pi B0))` is the physical toroidal-flux coordinate in
metres. It must not be replaced by the normalized coordinate
`rho` when calculating gradient-bearing coefficients.

| Property | Definition | Unit |
| -------- | ---------- | ---- |
| `gm1` | `<1/R^2>` | `m^-2` |
| `gm2` | `<\|grad rho_tor\|^2/R^2>` | `m^-2` |
| `gm3` | `<\|grad rho_tor\|^2>` | `1` |
| `gm4` | `<1/B^2>` | `T^-2` |
| `gm5` | `<B^2>` | `T^2` |
| `gm6` | `<\|grad rho_tor\|^2/B^2>` | `T^-2` |
| `gm7` | `<\|grad rho_tor\|>` | `1` |
| `gm8` | `<R>` | `m` |
| `gm9` | `<1/R>` | `m^-1` |

All nine coefficients share one lazy Numba materialization because the surface weights,
magnetic-field magnitude, and `grad rho_tor` metric are common intermediates.

## Numba Field Materialization

`Equilibrium` field materialization is independent of `veqpy.kernels`, source
lowering, adapters, and solver workspaces. Its private Numba implementation
consumes only root scalars/profiles and tables supplied by `Grid`. Numba removes
Python loop overhead and vectorized temporary arrays, but does not change the
value semantics of Reactive properties.

The calculation graph follows three rules:

1. Each physical property is a normal reactive node. Reading `q`, for example,
   does not calculate current-density or GS-residual fields. Its newly allocated
   immutable result is retained by the Reactive cache, so repeated reads do not
   recalculate or reallocate it.
2. Work is grouped only where the same coordinate derivatives dominate every
   member: the `R` coordinate stage, the `Z` coordinate stage, and the
   Jacobian/metric plus five radial geometry integrals. Reading `R` alone does
   not materialize `Z` or the metric group.
3. When a dependency changes, recomputation returns a new array. References to
   an older cached result remain unchanged; public values are never live views
   of shared mutable execution memory.

Root radial profiles are shape-checked at the dependency boundary that consumes
them. A malformed `FFn_psin` therefore blocks `FFn_r`, `F`, or `q`, but it does
not prevent independent geometry such as `R` from being read. Replacing `Grid`
invalidates grid-dependent descendants through the ordinary Reactive graph.

### Magnetic-axis limits

On grids that contain `r=0`, the public coordinate Jacobian is the raw
Jacobian produced by the surface map. It is not floored or clamped. Consequently
`J`, `S`, `V`, `S_r`, `V_r`, and `Ln_r` vanish at the magnetic axis through their
defining formulas. The model repairs only removable singularities in quantities
that divide these primitives:

- the local `gttdivJR` row uses its leading `r*(A + O(r))` form; its radial
  derivative and the finite mixed metric coefficient use the corresponding
  first-order-in-`r` local limit (local fixed-angle geometry is not generally
  even in `r`);
- flux functions `q`, `jtor`, `jpara`, and `jtotal` use an even-in-`r^2`
  limit reconstructed from the first two off-axis surfaces;
- `ftrap` follows its leading `sqrt(r)*(A + O(r))` behavior, and the local
  `jphi` formula is evaluated directly at the axis rather than extrapolated
  independently for every poloidal angle.

These repairs apply only to the coordinate singularity at the first radial
point. A zero or non-finite metric denominator on an off-axis surface is an
invalid equilibrium and raises an error; it is neither clamped nor
extrapolated.

## Snapshot Responsibilities

The model layer follows the principle "minimal independent state plus interpretable derived quantities." It gives users an equilibrium object that can be read, plotted, compared, and serialized. It does not solve the Grad--Shafranov equation again and does not perform high-frequency residual refreshes. New diagnostics should usually be added as `Equilibrium` properties derived from existing root state, rather than by exposing solver or engine internals directly.
