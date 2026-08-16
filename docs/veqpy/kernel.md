# Kernel boundary

VEQPy exposes the Module boundary, not the low-level buffer constructors. A
caller supplies an ordinary topology mapping and an optional solver mapping:

```python
import veqpy

module = veqpy.build(
    topology={
        "Nr": 8,
        "Nt": 12,
        "route": "PF",
        "coordinate": "psin",
        "constraint": "ip",
        "h_count": 2,
        "kappa_count": 2,
        "psin_count": 3,
        "s_counts": (2, 2),
    },
    solver={"max_evaluations": 800},
    backend="numba",
)
record = module.solve(
    boundary={
        "a": 0.9,
        "R0": 3.0,
        "Z0": 0.0,
        "B0": 5.0,
        "kappa_lcfs": 1.5,
        "c_lcfs": [0.0, 0.0, 0.0],
        "s_lcfs": [0.0, 0.0],
    },
    source={
        "psin": psin,
        "P_psin": P_psin,
        "FF_psin": FF_psin,
        "P0": 0.0,
    },
    targets={"Ip": 1.0e6},
)
```

Inside the Module, the numerical boundary has exactly four named records:
`KernelTopology`, `KernelInput`, `KernelConfig`, and `KernelOutput`. They are
private implementation records. `KernelTopology` contains only structural
quadrature/calculus and basis information; source nodes and source counts are
runtime data in `KernelInput`.

The Adapter owns one resident input buffer. It starts with capacity 256,
grows to 512 and then 1024 when a prepared source requires it, and never
shrinks. The source count selects the active prefix. Growing the buffer
increments its capacity epoch but preserves the array identities and does not
change topology identity or trigger compilation.

The coordinate, pressure, and route-driver arrays must already use one shared
grid; the Adapter validates and copies them without remapping. The Kernel may
evaluate that explicit representation on its own operator nodes as part of the
numerical route. A normal call can request materialization, Rich diagnostics,
and a complete JSON report independently of the build defaults.
