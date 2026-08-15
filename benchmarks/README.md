# VEQPy 2.x benchmarks

`benchmark_v2.py` measures the public `VEQ` Module with the fixed four-buffer
Kernel contract. The benchmark reuses one prepared Module and one frozen
`Plasma` context, so the reported rows include Adapter, solve, and Record
overhead for repeated calls.

```bash
.venv/bin/python benchmarks/benchmark_v2.py --backend numba --repeat 5
.venv/bin/python benchmarks/benchmark_v2.py --backend cxx --repeat 5
```

Numba is the required backend. Cxx is measured only for its supported
intersection; explicit source nodes and `rho` coordinate closure are rejected
with a diagnostic rather than silently falling back to Numba.
