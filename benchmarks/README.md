# VEQPy 2.x benchmarks

`cxx_geqdsk.py` loads the passive GEQDSK payload through
`fusionprime_base.io.load_geqdsk`; its small Plasma fixture builder is local to
the benchmark. It measures the prepared Kernel/core path first and the public
`VEQ.solve(materialize=False)` path second. Artifact compilation, imports, and
Numba JIT work are outside the formal timing loop.

The formal run requires at least five warmups and 100 interleaved repetitions:

```bash
../.venv/bin/python benchmarks/cxx_geqdsk.py --warmup 5 --repeat 100
```

The JSON contains the SOLOVEV, CHEASE, and EFIT rows, historical profile
signatures and LCFS-fit metadata, p25/median/p75 timings, nfev, acceptance,
same-input residual parity, CPU/compiler/artifact metadata, and the secondary
nonmaterializing Module timings. The backend columns are `numba`,
`cxx-strict`, `cxx-relaxed`, and `cxx-enzyme`; the last is always explicitly
`skipped` while cxx-enzyme remains deferred.

The performance qualification compares `numba / cxx-relaxed` directly with the
historical main-branch README range of about 5–11x. A clear Cxx advantage below
that scale is recorded as a failed qualification; the benchmark then stops and
does not start enzyme work.
