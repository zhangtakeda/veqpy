# VEQPy 1.3.4 — CHEASE-compatible GEQDSK exterior continuation

VEQPy 1.3.4 changes the default LCFS-exterior flux written by
`Equilibrium.to_geqdsk()` from a scalar `psi_bound` fill to the CHEASE
`NEQDXTPO=1` continuation. The final strictly interior flux interval is
extrapolated linearly along rays from the magnetic axis, so rectangular-grid
points outside the LCFS remain strictly on the exterior side of `psi_bound`.

This maintenance release is based directly on VEQPy 1.3.3. It does not include
the later solve-map JVP, native toroidal-flux-radius source closure, or source
API migration currently present on the main development branch.

## Highlights

### One-sided LCFS crossing

- Each exterior grid point is mapped to a ray from the magnetic axis.
- Intersections with the LCFS and the final strictly interior surface determine
  the normalized radial interval used for linear flux continuation.
- Normalized exterior flux is kept strictly above one for either sign of the
  physical poloidal-flux span.
- Nearby exterior contours remain ordered continuations of the supplied LCFS,
  and the continued edge gradient supports contour searches initialized from
  either side of the boundary.
- This field is export continuation data rather than a vacuum
  Grad-Shafranov solution.

### Compatibility

- Python 3.12 or newer is required.
- No public symbols, parameters, or source-coordinate names are removed or
  renamed relative to VEQPy 1.3.3.
- Calls that omit `psi_outside` now receive the CHEASE-style continuation.
- Passing `psi_outside=<physical psi>` explicitly retains the previous scalar
  exterior behavior.
- `Geqdsk.boundary` remains the exact, explicitly closed LCFS polygon introduced
  in VEQPy 1.3.3.
- Ray continuation requires the LCFS and reference interior surface to be
  star-shaped about the magnetic axis; unsupported geometry raises a clear
  error instead of emitting invalid exterior flux.

## Validation

- Ruff static checks passed.
- The complete local test suite passed.
- Analytic elliptical-LCFS tests cover both signs of the physical flux span,
  the CHEASE linear continuation formula, nested exterior contours, and
  LCFS-gradient contour convergence.
- Asymmetric GEQDSK write/read round trips passed for both physical flux
  orientations.
- The source distribution and wheel passed `twine check`.
- An isolated Python 3.12 wheel installation reported version 1.3.4, imported
  outside the source tree, and produced nonconstant normalized exterior flux
  strictly above one.

**Full changelog:** [v1.3.3...v1.3.4][full-changelog]

[full-changelog]: https://github.com/zhangtakeda/veqpy/compare/v1.3.3...v1.3.4
