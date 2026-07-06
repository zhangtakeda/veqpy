# Kernel Solve Flow

VEQlib solves through `veqlib.Kernel`; VEQPy supplies model-layer inputs and
receives `Equilibrium` snapshots.

```python
from veqlib import Kernel, KernelConfig, KernelRecipe

kernel = Kernel(
    topology=topology,
    recipe=KernelRecipe(backend="numba", layout="degree"),
    config=KernelConfig(method="levenberg-marquardt"),
)
result = kernel.solve(boundary, source)
equilibrium = kernel.build_equilibrium()
```

`KernelConfig` controls the runtime solve method, residual normalization,
initial-state policy, continuation policy, residual acceptance threshold, and
evaluation limits. The current Numba backend maps `method="powell"` to the
Powell-style `hybr` solver and `method="levenberg-marquardt"` to SciPy's LM
solver.

`SolveResult` records the final packed state, raw residual, scaled residual,
source `alpha` values, function/iteration counters, success flag, and elapsed
time. `Kernel.jvp(...)` and `Kernel.jacobian(...)` are finite-difference numerical
queries over the same residual runtime.

Warm continuation is handle-local: after a solve, the next `Kernel.solve(...)`
can reuse the previous solution when the continuation policy is warm. Use
`kernel.clear()` to drop the stored result and history.
