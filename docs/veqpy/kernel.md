# Kernel

The public runtime entrypoint is the Kernel API:

```python
from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
```

`KernelTopology` fixes the active packed coefficient counts, grid size, source
route, coordinate system, node semantics, source constraints, and capacity
limits. Active count fields determine the public packed vector size. `L_max`,
`M_max`, and `K_max` are setup/capacity limits: explicit values may exceed the
minimal active-count requirement, but may not be smaller. `KernelBoundary` and
`KernelSource` carry per-case physical inputs. `KernelConfig` carries nonlinear
solve policy. `Kernel.solve(...)` returns a shared `SolveResult`, and
`Kernel.build_equilibrium()` materializes the current `Equilibrium` snapshot.
For sine Fourier data, Kernel-level public inputs are s1-started:
`KernelTopology.s_counts=(n1, n2, ...)` and `KernelBoundary.s_offsets=(s1, s2, ...)`.
The runtime adds the structural s0=0 slot before backend calls.
`KernelBoundary` accepts either explicit parameterized geometry
(`a`, `R0`, `Z0`, `B0`, `ka`, `c_offsets`, `s_offsets`) or raw LCFS point arrays
(`R_boundary`, `Z_boundary`) plus `c_order`/`s_order`; the latter form performs
the least-squares Fourier projection during construction and records fit
diagnostics on `fit_*` fields.

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

## Topology Variants

`Kernel.variant(...)` is for multi-topology count sweeps on Numba-backed
handles: within fixed setup and capacity limits, it switches the active
topology counts without rebuilding fixed grid, capacity workspace, and compiled
runtime state. It is an in-place state switch that returns the same `Kernel`
object and constructs a fresh immutable `KernelTopology`; previously saved
topology objects are not mutated.

Only active count fields may be changed:

```python
kernel.variant(
    h_count=6,
    c_counts=(2, 1),
)
```

Omitted arguments and explicit `None` inherit the current active count.
`variant()` does not change fixed setup or capacity fields such as
`Nr/Nt/route/coordinate/nodes/sample_count/ip_constraint/beta_constraint`,
`quadrature`, `calculus`, `L_max`, `M_max`, or `K_max`. New counts must fit the
current capacity limits: radial counts require `count <= L_max + 1`, cosine
orders require `order <= M_max`, and sine orders use the public s1-started
indexing.

Successful variants clear the current result, last runtime case, and prepare
cache so old packed solutions cannot warm-start a different active layout.
`history` is retained as a per-handle solve log and can contain entries from
different active topologies and `x_size` values. `Kernel.variant()` is not
thread-safe and must not run concurrently with solve, residual, derivative,
prepare, or equilibrium snapshot calls.
