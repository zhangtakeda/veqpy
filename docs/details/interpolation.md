# Interpolation

The `interpolation` module remaps one-dimensional source data. Given values $f(s_j)$ on source nodes and target query points $q_i$, interpolation is represented as a matrix or equivalent local evaluation:

$$
\hat f(q_i) \approx \sum_j H_{ij} f(s_j).
$$

This layer mainly serves the source stage. External input may be sampled on uniform nodes or already live on the operator grid; the source coordinate may be `rho`, `psin`, or a route-derived coordinate. Implementation details live in `veqpy/math/interpolate.py`.

## Input Nodes

| nodes | Meaning | Handling |
| ----- | ------- | -------- |
| `grid` | Input values already live on the current radial grid | Used directly, with no remap |
| `uniform` | Input values live on uniformly spaced nodes in $[0,1]$ | Remapped to query points with the selected format |

When the source coordinate is `rho`, query points are fixed and a remap matrix can be precomputed. When the coordinate is `psin`, query points usually change with the current flux profile, so the source stage evaluates the interpolant after the updated query points are known.

## Formats

| Format | Typical use |
| ------ | ----------- |
| Global Lagrange / barycentric | General interpolation between arbitrary distinct nodes |
| `linear`, `quadratic`, `cubic` | Local polynomial interpolation for uniform source data |
| `not-a-knot` | Cubic not-a-knot spline for uniform source data |
| `barycentric` | Local barycentric stencil for uniform source data; this is the package default |

Local formats depend only on a finite neighborhood around the query point, which makes them better suited to robust remapping of external source samples. Global formats retain full polynomial closure, but each query point usually depends on every source sample. If the sample count is too small, local polynomial and spline formats degrade to the supported lower order.

## Boundary

The interpolation layer only handles normalized one-dimensional parameters. It does not decide the physical meaning of a source route or select a constraint equation. `operator/source_plan.py` organizes route, coordinate, and node semantics into a source plan; the interpolation module only provides stable numerical remapping tools. This boundary allows GEQDSK inputs, array inputs, and coefficient-based inputs to share the same source-kernel entry points.
