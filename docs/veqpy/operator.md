# Kernel Runtime

The public runtime entrypoint is the Kernel API:

```python
from veqpy.facade import Kernel, KernelBoundary, KernelConfig, KernelSource, KernelTopology
```

`KernelTopology` fixes the packed coefficient topology, grid size, source route,
coordinate system, node semantics, and source constraints. `KernelBoundary` and
`KernelSource` carry per-case physical inputs. `KernelConfig` carries nonlinear
solve policy. `Kernel.solve(...)` returns a shared `SolveResult`, and
`Kernel.build_equilibrium()` materializes the current `Equilibrium` snapshot.

The direct Numba implementation lives under `veqpy.kernel`. Its internal runtime
still owns packed layout metadata, source materialization, residual workspaces,
and equilibrium snapshot assembly, but those details are not separate public
problem/operator objects.

Public source inputs stay raw: `KernelSource.heat_profile`,
`KernelSource.current_profile`, `KernelSource.Ip`, and `KernelSource.beta`.
Route-dependent `mu0` scaling and the internal `scaled_*` arrays are materialized
inside the facade/runtime layer before residual evaluation.
