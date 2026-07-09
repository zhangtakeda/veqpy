# Architecture

VEQPy is a single public package with four user-visible layers:

| Layer            | Responsibility                                                                          |
| ---------------- | --------------------------------------------------------------------------------------- |
| `veqpy.base`     | Reactive derived-state utilities, serialization, and registry helpers                   |
| `veqpy.numerics` | General interpolation, differentiation, quadrature, and projection helpers              |
| `veqpy.model`    | Serializable model objects: `Grid`, `Profile`, `Geqdsk`, and `Equilibrium`              |
| `veqpy.kernels`  | Public Kernel contract, backend dispatch, and private Numba/Cxx runtime implementations |

The dependency direction is intentionally one-way: model code may use `base` and
`numerics`, while Kernel backends may use model objects to build solved
snapshots. Model objects do not depend on Kernel runtime workspaces, source
route lowering, or backend-specific packed arrays.

Public construction stays concentrated at the package root:

```python
from veqpy import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    build,
    solve,
    fit,
    pareto,
)
```

`veqpy.api` contains function-style helpers that forward to the same `Kernel`
surface. Backend implementations remain private and are selected only through
`KernelRecipe.backend`.

## User-Facing Kernel Surface

The recommended Kernel API is the package-root surface:

| Name | Role |
| --- | --- |
| `Kernel` | Stateful backend-neutral runtime handle for repeated solves, variants, diagnostics, and equilibrium construction. |
| `KernelRecipe` | Backend and artifact/layout recipe. Users mainly set `backend`, `layout`, and Cxx build options. |
| `KernelTopology` | Fixed route, grid, active-count, and capacity contract. It derives `x_size`, route codes, active profiles, and canonical `key`. |
| `KernelBoundary` | Runtime boundary input. Users may pass parameterized `a/R0/Z0/B0/ka/c_offsets/s_offsets` values or raw `R_boundary/Z_boundary` points for explicit fitting. |
| `KernelSource` | Runtime heat/current profile arrays plus physical constraints such as `Ip`, `beta`, and optional `case_name`. |
| `KernelConfig` | Solver policy for one invocation: method, residual limits, evaluation budget, initial/continuation policy, and normalization mode. |
| `SolveResult` | Snapshot returned by `solve()`, containing timing, convergence counters, residuals, packed solution `x`, and scaling data. |
| `ParetoSample` | One verified Pareto topology sample with `counts`, `time`, `complexity`, `shape_error`, and its `SolveResult`. |
| `ParetoResult` | Result returned by `pareto()`, containing the reference sample, evaluated samples, and frontier. |

The function-style helpers are short-lived wrappers:

| Helper | Use |
| --- | --- |
| `build(topology=..., recipe=..., config=...)` | Construct and prepare a `Kernel`. |
| `solve(boundary, source, topology=..., ...)` | Construct a short-lived kernel, solve once, and close it. |
| `fit(boundary, backend="numba", ...)` | Explicitly fit a raw-point `KernelBoundary` and return an equivalent parameterized boundary. |
| `pareto(boundary, source, topology=..., candidates=..., ...)` | Construct a short-lived Numba kernel and evaluate explicit topology-reduction candidates. |

`Kernel` exposes the following user methods and properties:

| Member | Use |
| --- | --- |
| `topology`, `recipe`, `config`, `x_size` | Inspect the active topology, recipe, config, and packed-solution size. |
| `history`, `result` | Inspect accepted solve results for the current handle. |
| `prepare(force=False, dry_run=False)` | Prepare the backend artifact/workspace. The returned object is informational and is not part of the root public type surface. |
| `variant(...)` | Mutate active counts within the fixed capacity topology and return the same handle. |
| `solve(boundary, source, ...)` | Solve one runtime case. |
| `pareto(boundary, source, candidates=..., ...)` | Evaluate explicit reduced topology candidates against the active Numba capacity topology. |
| `residual`, `residual_into`, `jvp`, `jvp_into`, `jacobian`, `jacobian_into` | Low-level numerical diagnostics for a supplied packed state. |
| `build_equilibrium(x=None)` | Build a model `Equilibrium` from the latest or supplied packed state. |
| `clear()`, `close()`, `pinned()` | Manage handle state, backend resources, and optional CPU pinning context. |

Implementation helper types are intentionally not package-level exports.
`KernelParetoSignature` is an internal candidate-signature key,
`KernelPrepareResult` is the concrete informational return record for
`Kernel.prepare()`, and `TopologyError` is the internal validation exception
type used by topology/recipe normalization. User code should rely on the public
constructor and method contracts above rather than importing those names from
`veqpy`.
