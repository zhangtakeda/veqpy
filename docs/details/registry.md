# Registry

`Registry` is VEQPy's lightweight dispatch mechanism for "finite but extensible method families." It is a decorator-backed typed mapping: registry construction declares key and value types, and registration binds one or more keys to an implementation function. String keys are normalized for case-insensitive lookup, and the mapping is exposed as read-only data.

Source location: `veqpy/base/registry.py`.

## Basic Use

```python
registry = Registry(str, Callable)

@registry("name", "alias")
def build(...):
    ...
```

Upper-level factories normalize user input, query the registry, and construct error messages. Concrete implementations remain near the module where their mathematics or physics belongs. Adding a quadrature, differentiation, interpolation, residual-scale, or serialization method therefore does not require growing a central `if/elif` dispatcher.

## Method Families

Many VEQPy choices are discrete method spaces:

- quadrature scheme;
- calculus scheme;
- uniform source interpolation format;
- source route kernel;
- residual normalization;
- JSON, pickle, GEQDSK, and other serialization formats.

Representing these choices as enumerable mappings lets factories, tests, and error messages derive supported options from the same public collection. Implementations, aliases, and registration declarations stay local, which makes each method's assumptions and tests easier to maintain.

## Source Route

The source-route registry uses the key

```python
(route, coordinate, nodes)
```

where `route` distinguishes PF/PP/PI/PJ1/PJ2/PQ constraint paths, `coordinate` distinguishes `rho` and `psin`, and `nodes` distinguishes `uniform` and `grid`. This tuple is not a loose plugin name; it is a coordinate representation of the source-modeling space.

As a result, the operator build plan can validate the route during construction, the backend ABI can declare supported combinations, and tests can enumerate whether each expected combination has a unique implementation. This is the deeper role of the registry: it lifts physical branching out of implicit control flow into an explicit, finite, verifiable model coordinate.

## Boundary

`Registry` does not decide whether a method is physically appropriate and does not hide differences among implementations. It provides a stable entry point and controlled namespace so solver/operator code can be organized around fixed interfaces while numerical methods and source routes evolve independently in their own modules.
