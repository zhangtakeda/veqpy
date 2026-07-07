# Kernel

The public runtime entrypoint is the Kernel API:

```python
from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
```

`KernelTopology` fixes the packed coefficient topology, grid size, source route,
coordinate system, node semantics, and source constraints. `KernelBoundary` and
`KernelSource` carry per-case physical inputs. `KernelConfig` carries nonlinear
solve policy. `Kernel.solve(...)` returns a shared `SolveResult`, and
`Kernel.build_equilibrium()` materializes the current `Equilibrium` snapshot.
For sine Fourier data, Kernel-level public inputs are s1-started:
`KernelTopology.s_counts=(n1, n2, ...)` and `KernelBoundary.s_offsets=(s1, s2, ...)`.
The runtime adds the structural s0=0 slot before backend calls.

The direct Numba implementation is a private Kernel backend. Its internal runtime
owns packed layout metadata, source materialization, residual workspaces, and
equilibrium snapshot assembly, but those details are not separate public objects.
`KernelRecipe.backend` selects the backend implementation; user code continues
to call the same `Kernel` methods.

Public source inputs stay raw: `KernelSource.heat_profile`,
`KernelSource.current_profile`, `KernelSource.Ip`, and `KernelSource.beta`.
Route-dependent scaling and internal materialized source arrays are backend
runtime details, not user-facing data fields.

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
