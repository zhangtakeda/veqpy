# Reactive

`Reactive` is the infrastructure used by VEQPy's model layer to express "minimal state plus formula-derived quantities." It separates object attributes into two classes: root properties are independent state written by construction, deserialization, or the user; derived properties are computed from root and other derived properties by formula. Derived properties are cached, but every read first validates dependency versions and recomputes when needed.

Source location: `veqpy/base/reactive.py`.

## Why It Exists

Fixed-boundary equilibrium snapshots have only a small independent state: grid, shape profiles, flux profile, source derivatives, and scaling coefficients. Many quantities users read, such as $R,Z,J,q,\beta_t,j_\phi$ and GEQDSK export profiles, are determined by that root state.

If every intermediate value were stored as a mutable field, an object would contain two truths: state and formula. It would also need to maintain the update order among profile, geometry, source, and diagnostics. `Reactive` turns this order problem into a dependency-graph problem: writing root state updates versions, and reading a derived property validates the cache through dependencies.

## Core Mechanism

Each subclass declares:

```python
root_properties = {"a", "b", ...}
```

Root setters inspect and normalize input values, store them, and increment versions. Non-root properties are treated as derived properties. During class creation, `Reactive` parses `self.xxx` accesses to build a dependency graph; indirect dependencies can be added with `@depends_on(...)`.

When a derived property is read, its getter compares dependency tokens. Tokens include direct field versions and revisions of nested `Reactive` objects. Therefore changes inside `Equilibrium.grid` or `shape_profiles` invalidate dependent geometry and diagnostics on the next read.

## Boundary

`Reactive` is used for model objects and solved snapshots, not for the operator hot path. Solver runtime needs explicit workspaces and in-place array refreshes; snapshots need an interpretable, serializable, on-demand formula system. Separating the two lets VEQPy keep solver memory locality while preventing stale caches and duplicated truths in public model objects.
