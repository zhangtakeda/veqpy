# Backends

Numba is the required and default execution path:

```python
module = veqpy.VEQ(topology=topology, backend="numba")
```

Cxx uses the same `KernelInput`/`KernelOutput` binding and is parity-tested on
the supported intersection. The current native capability boundary explicitly
rejects `nodes="explicit"` and `coordinate="rho"`; it does not silently
switch to Numba. A missing native toolchain is reported by the CLI and by the
benchmark command.

```bash
.venv/bin/python -m veqpy --demo numba
.venv/bin/python -m veqpy --demo cxx
```

The Cxx wheel includes its native source and third-party notices. Building a
wheel does not claim that every host has the compiler and native libraries
needed to prepare a Cxx artifact.
