# Kernel Solve Flow

VEQPy solves through `veqpy.facade.Kernel`.

```python
kernel = Kernel(topology=topology, config=KernelConfig(method="levenberg-marquardt"))
result = kernel.solve(boundary, source)
equilibrium = kernel.build_equilibrium()
```

`KernelConfig` controls the runtime solve method, residual normalization,
initial-state policy, continuation policy, residual acceptance threshold, and
evaluation limits. The current Numba backend supports Powell-style `hybr` with
LM fallback through `method="powell"` and a direct LM path through
`method="levenberg-marquardt"`.

`SolveResult` records the final packed state, raw residual, scaled residual,
source `alpha` values, function/iteration counters, success flag, and elapsed
time. JVP and Jacobian counters remain zero for the Numba backend because those
APIs are explicit `NotImplementedError` surfaces.

Warm continuation is handle-local: after a solve, the next `Kernel.solve(...)`
can reuse the previous solution when the continuation policy is warm. Use
`kernel.clear()` to drop the stored result and history.
