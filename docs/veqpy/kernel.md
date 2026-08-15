# Four-buffer Kernel API

The low-level numerical boundary has exactly four named data types:

- `KernelTopology`: frozen compile-time structure and `source_capacity`;
- `KernelInput`: mutable, preallocated per-case numeric buffers;
- `KernelConfig`: frozen integer option codes and numeric tolerances;
- `KernelOutput`: mutable, preallocated diagnostics and materialization roots.

```python
import veqpy

topology = veqpy.KernelTopology(
    h_count=2, v_count=0, kappa_count=2, psin_count=3, F_count=0,
    c_counts=(), s_counts=(2, 2), Nr=8, Nt=12,
    route="PF", coordinate="psin", nodes="uniform",
    constraint="ip", sample_count=8,
)
kernel = veqpy.Kernel(topology=topology, backend="numba")
kernel.prepare()
output = kernel.solve()
assert output is kernel.output
```

The Adapter normally fills `KernelInput` from a frozen Plasma. All source
arrays have the topology capacity. For a dynamic or explicit source grid,
`source_count` selects the active prefix and the unused suffix is zeroed before
the solve. Overflow is an error requiring a new topology and preparation.

`residual_into`, `residual_jvp_into`, and `jacobian_into` write into caller
owned numeric arrays. Structural variants require a new Kernel; a prepared
Kernel never changes topology. Backend build policy is private to dispatch.

The former multi-object case/result contracts are not part of the 2.x API.
