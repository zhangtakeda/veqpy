# Operator

`Operator` is the runtime center of the solve. It receives a packed vector $x$, refreshes profile, geometry, source, and residual stages on fixed layout/workspace objects, and returns a packed residual. Unlike an `Equilibrium` snapshot, `Operator` is a hot-path object: memory layout is explicit, arrays are updated in place, and intermediate values are not exposed as public properties.

The relevant source files mainly live in `veqpy/operator/`, `veqpy/layout/`, `veqpy/workspace/`, and `veqpy/engine/`.

## OperatorCase

`OperatorCase` describes one fixed-boundary solve input: source route, source coordinate, node semantics, active profile coefficients, boundary, heat/current-related inputs, and optional `Ip` or `beta` constraints.

`route`, `coordinate`, and `nodes` jointly form the source route key:

```python
(route, coordinate, nodes)
```

This key selects the source kernel and the interpretation of the input arrays. `heat_input` and `current_input` remain one-dimensional data; their physical meaning is determined by the selected route.

## Packed Layout

The packed layout defines where each coefficient lives in the optimization vector $x$. The current profile family includes shape profiles `h`, `v`, `k`, `c0`, `c`, `s`, and source/flux-related profiles `psin`, `F`. Only active profiles enter the packed vector.

The default layout is degree-first: all active profiles contribute their low-order coefficients first, and higher degrees follow. This gives residual blocks, profile refresh, and solver initial values a shared index semantics.

## Build Plan and Pipeline

When an `Operator` is built, `Grid` is reduced to an array-only workspace, and `OperatorBuildPlan` binds profile layout, source route, backend ABI, residual-block metadata, and offsets/scales. This plan describes solve topology. If the active profile set, coefficient lengths, or route topology changes, the operator should be rebuilt.

One residual call has four main stages:

| Stage | Role |
| ----- | ---- |
| profile | Refresh active profiles from packed $x$ |
| geometry | Compute geometry fields and flux-surface averages from shape profiles |
| source | Generate flux/source root fields from route inputs and constraints |
| residual | Assemble and pack the Grad--Shafranov residual |

`Operator.__call__(x)` returns the variational/Galerkin residual. `residual_collocation(x)` returns the pointwise residual used for collocation polish.

## Snapshot Boundary

After the solve, `build_equilibrium(x)` refreshes the runtime with the final solution and writes only snapshot-relevant root fields and shape profiles into `Equilibrium`. Runtime buffers are not transferred into the model object. This boundary keeps the operator in a high-throughput mutable form while making `Equilibrium` a serializable, interpretable physical snapshot.
