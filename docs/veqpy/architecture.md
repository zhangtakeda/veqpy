# Architecture

VEQPy is a single public package with four user-visible layers:

| Layer | Responsibility |
| --- | --- |
| `veqpy.base` | Reactive derived-state utilities, serialization, and registry helpers |
| `veqpy.numerics` | General interpolation, differentiation, quadrature, and projection helpers |
| `veqpy.model` | Serializable model objects: `Grid`, `Profile`, `Boundary`, `Geqdsk`, and `Equilibrium` |
| `veqpy.kernels` | Public Kernel contract, backend dispatch, and private Numba/Cxx runtime implementations |

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
)
```

`veqpy.api` contains function-style helpers that forward to the same `Kernel`
surface. Backend implementations remain private and are selected only through
`KernelRecipe.backend`.
