# VEQPy 2.x benchmarks

VEQPy keeps two benchmark entry points: one for real GEQDSK qualification and
one for complete Kernel route coverage. Generated JSON remains under
`benchmarks/results/` and is not versioned.

## GEQDSK backend qualification

`cxx_geqdsk.py` loads the passive GEQDSK payload through
`fusionprime_base.io.load_geqdsk` and creates direct boundary/source/target
dictionaries on the native GEQDSK profile grid. The target current is
`abs(Geqdsk.Ip)` from the file header. It measures the prepared Kernel/core
path first and the public `VEQ.solve(materialize=False)` path second. Artifact compilation, imports, and
Numba JIT work are outside the formal timing loop.

The formal run requires at least five warmups and 100 interleaved repetitions:

```bash
../.venv/bin/python benchmarks/cxx_geqdsk.py --warmup 5 --repeat 100
```

The JSON contains the SOLOVEV, CHEASE, and EFIT rows, historical profile
signatures and LCFS-fit metadata, p25/median/p75 timings, nfev, acceptance,
same-input residual parity, CPU/compiler/artifact metadata, and the secondary
nonmaterializing Module timings. The backend columns are `numba`,
`cxx-strict`, `cxx-relaxed`, and `cxx-enzyme`. Enzyme is executed when
selected, with plugin/build failures recorded as backend failures.

The performance qualification compares `numba / cxx-relaxed` directly with the
historical main-branch README range of about 5–11x. A clear Cxx advantage below
that scale is recorded as a failed qualification without deleting the measured
backend evidence.

## Kernel route matrix

`kernel_routes.py` restores the historical synthetic route benchmark as one
fixed `4 Kernel x 7 route x 3 coordinate` matrix:

- Kernels: `numba`, `cxx-strict`, `cxx-relaxed`, `cxx-enzyme`
- Routes: `PF`, `PP`, `PI`, `PJ1`, `PJ2`, `PJ3`, `PQ`
- Coordinates: `r`, `rho`, `psin`

All 21 route/coordinate cases are manufactured from one converged reference
equilibrium. They use physical sources with `constraint=none`, so target
normalization is not an extra benchmark dimension. The script checks backend
solution parity, same-input residual parity, and reconstructed normalized-flux
agreement before accepting a timing cell. Each route/coordinate row runs in a
fresh worker process: its four Kernels remain interleaved for fair timing, while
the 63 topology-specific nanobind libraries are released between rows.

The complete default run uses five warmups and 100 repeats per cell. On the
reference machine, compiling missing native artifacts and running all 84 cells
is expected to take roughly ten minutes:

```bash
../.venv/bin/python benchmarks/kernel_routes.py
```

Use `--no-run` to inspect the 84-cell plan without compiling or solving. Route,
coordinate, and backend filters are intended only for focused diagnosis; a
timed filtered run must retain `numba` as the parity baseline. Use `--rebuild`
after editing the native source tree so the benchmark republishes artifacts;
ordinary qualification runs intentionally exercise the cached-load path.
