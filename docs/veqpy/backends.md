# Backends

The supported backend tokens are `numba`, `cxx`, `cxx-strict`, and
`cxx-relaxed`. `numba` is the default and always uses strict floating-point
math. `cxx` is the short spelling for the Release relaxed artifact;
`cxx-strict` and `cxx-relaxed` are independent Release artifacts.

All four paths consume the same runtime explicit source contract. A source
grid can contain any strictly increasing normalized nodes accepted by the
capacity policy, including two-point and nonuniform inputs. `rho` is a normal
coordinate choice, not a Cxx exclusion. Cxx keeps its native workspace bound
at 1024 while the Python buffer retains the active runtime count and capacity
epoch.

Build-only options are `artifact_dir`, `cpu_affinity`, and `rebuild`; they are
not solver or per-call options. The build policy is fixed before preparation.

```bash
.venv/bin/python -m veqpy --demo numba
.venv/bin/python -m veqpy --demo cxx-strict
.venv/bin/python -m veqpy --demo cxx-relaxed
```

The native toolchain is optional for importing VEQPy. A missing compiler or
native dependency is reported when a Cxx artifact is prepared.
