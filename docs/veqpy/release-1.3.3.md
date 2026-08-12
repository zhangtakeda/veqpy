# VEQPy 1.3.3 — Reactive snapshots and closed GEQDSK boundaries

VEQPy 1.3.3 adds immutable reactive snapshots and corrects LCFS boundary
serialization in GEQDSK output. Exported boundaries are now explicitly closed
and use at least 64 unique periodic samples, while finer source grids retain
their original angular resolution.

This release does not intentionally remove or rename any public API introduced
in VEQPy 1.3.x.

## Highlights

### Frozen and thawed reactive snapshots

- `Reactive.freeze()` recursively freezes an object and any nested reactive
  snapshots in place, rejecting subsequent replacement or deletion of
  authoritative root properties.
- Frozen objects retain lazy derived-property evaluation, so freezing state
  does not require eagerly populating derived caches.
- Reactive children created by a lazy derived property after its parent is
  frozen are frozen before they are exposed.
- `Reactive.thaw()` returns an independent, mutable deep snapshot with empty
  derived caches.
- Pickle state now records whether a reactive object is frozen without
  serializing caches or observer state. Objects written by the older pickle
  representation remain loadable as mutable snapshots.

### Closed and refined GEQDSK LCFS export

- `Equilibrium.to_geqdsk()` explicitly repeats the first LCFS coordinate as
  the final boundary coordinate, producing a conventionally closed polygon.
- Export uses at least 64 unique periodic LCFS samples plus the closing point,
  so `nbound` is at least 65 even when the source angular grid is coarser.
- Source grids finer than 64 angular points are preserved rather than
  downsampled.
- Endpoint-inclusive radial export and asymmetric vertical geometry remain
  registered during GEQDSK write/read round trips.
- The GEQDSK demo and Cxx parity coverage now verify both the minimum boundary
  resolution and exact closure.

## Compatibility and migration

- Python 3.12 or newer is required.
- No public symbols are intentionally removed or renamed.
- Existing mutable `Reactive` behavior is unchanged until `freeze()` is called.
- Consumers that previously closed `boundary` arrays themselves should tolerate
  the repeated endpoint now emitted by `to_geqdsk()`.
- GEQDSK files exported from coarse angular grids will contain more LCFS points
  than before.

## Validation

- The complete local test suite passed with 357 tests and 16 optional skips.
- Ruff static checks passed.
- The source distribution and wheel passed `twine check`.
- The wheel passed isolated-install metadata, import, reactive-snapshot, GEQDSK
  boundary, and default-backend smoke checks.
- CI validates Python 3.12 and 3.13, both solver backends, package construction,
  and isolated wheel installation.

**Full changelog:** [v1.3.2...v1.3.3][full-changelog]

[full-changelog]: https://github.com/zhangtakeda/veqpy/compare/v1.3.2...v1.3.3
