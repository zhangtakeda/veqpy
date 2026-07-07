# Kernel Runtime

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
Route-dependent `mu0` scaling and the internal `scaled_*` arrays are materialized
inside the Kernel runtime layer before residual evaluation.
