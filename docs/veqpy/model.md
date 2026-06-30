# Model

The `model` layer stores interpretable, serializable physical objects in VEQPy. It is not a mirror of the operator runtime; it organizes independent inputs and solved snapshots into a stable API. `Problem` keeps user-facing source, boundary, and active-profile topology inputs, while `Profile` is the serializable parameter object used by solved shape-profile snapshots. Derived grid geometry and equilibrium diagnostics are reconstructed lazily through `Reactive` properties. Files store only root state.

The main source files live in `veqpy/model/`.

## Objects

| Object        | Responsibility                                                                                            |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `Grid`        | Radial/angular discretization, quadrature weights, differentiation/integration matrices, and basis tables |
| `Profile`     | Parameterized representation of a one-dimensional radial profile                                          |
| `Boundary`    | Fixed-boundary geometry parameters, including fitting from a GEQDSK boundary                              |
| `Geqdsk`      | GEQDSK data loading, storage, and conversion                                                              |
| `Equilibrium` | Solved continuous equilibrium snapshot and diagnostic interface                                           |

`Profile` represents a one-dimensional radial profile with scale, power, envelope, offset, and optional Chebyshev coefficients. Operator setup lowers active profile topology to flat arrays; `Profile` remains on the model side for `Equilibrium.shape_profiles` and other serializable snapshots.

## Equilibrium Snapshot

`Equilibrium` is the main output object. It receives root fields after the solve, including:

- geometry scales: `R0`, `Z0`, `B0`, `a`;
- discretization object: `grid`;
- fixed-boundary and shape profiles: `shape_profiles`;
- flux and source derivatives: `psin`, `psin_r`, `psin_rr`, `FFn_psin`, `Pn_psin`;
- scaling coefficients: `alpha1`, `alpha2`.

These fields are sufficient to reconstruct common physical quantities, but they do not store temporary buffers from the operator hot path. When the user reads properties such as `R`, `Z`, `F`, `P`, `q`, `Ip`, `beta_t`, `jtor`, `jpara`, `jphi`, `Psi`, or `Phi`, the object computes the required values by formula and lets `Reactive` maintain dependency consistency.

## Geometry and Diagnostics

The model layer exposes stable snapshot diagnostics, not every intermediate array used during residual assembly. Geometrically, `Equilibrium` provides flux-surface mapping, Jacobian-related fields, area/volume, and flux-surface-averaged geometry factors. Physically, it provides pressure, toroidal-field function, safety factor, current, and flux diagnostics.

A small set of packed geometry fields is retained because those combinations have stable meaning for plotting, comparison, and GEQDSK export. Finer local derivative combinations, residual projection matrices, and backend workspaces remain in the operator/runtime layer and do not become public snapshot API.

## Boundary

The model layer follows the principle "minimal independent state plus interpretable derived quantities." It gives users an equilibrium object that can be read, plotted, compared, and serialized. It does not solve the Grad--Shafranov equation again and does not perform high-frequency residual refreshes. New diagnostics should usually be added as `Equilibrium` properties derived from existing root state, rather than by exposing solver or engine internals directly.
