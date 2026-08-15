# VEQPy 2.x benchmarks

`benchmark_v2.py` measures the public dictionary-based `VEQ` Module. The
benchmark reuses one prepared Module and one frozen
`Plasma` context, so the reported rows include Adapter, solve, and Record
overhead for repeated calls.

```bash
.venv/bin/python benchmarks/benchmark_v2.py --backend numba --repeat 5
.venv/bin/python benchmarks/benchmark_v2.py --backend cxx --repeat 5
```

Numba is the required backend. The Cxx variants accept the same runtime
explicit source contract; `cxx` is the relaxed Release spelling.
