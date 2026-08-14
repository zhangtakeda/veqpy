# PJ2/PJ3 Strict Radial Closure

## Contract

Geometric-r PJ2 and PJ3 no longer need an outer parameterized F profile.
With `F_count=0`, the Numba source stage solves for

$$
x=\frac{\mathrm d\psi}{\mathrm d\r},\qquad
C=K_nx=\frac{\mu_0 I_{\rm tor}}{2\pi},\qquad
u=\log\!\left(\frac{F^2}{F_{\rm edge}^2}\right).
$$

Defining

$$
g_1=\langle R^{-2}\rangle,\qquad
H=\frac{4\pi^2K_n}{V_r},\qquad
g_5=F^2g_1+x^2H,
$$

the two current semantics use

$$
\begin{aligned}
\text{PJ2:}\quad
u_r&=-\frac{2(\widehat p_r+xg_1\widehat j_2)}{g_5}, &
C_r&=L_{n,r}\widehat j_2+\frac12Cu_r,\\
\text{PJ3:}\quad
u_r&=-\frac{2(\widehat p_r+xB_0\widehat j_3/F)}{g_5}, &
C_r&=\frac{L_{n,r}B_0\widehat j_3}{Fg_1}+\frac12Cu_r.
\end{aligned}
$$

The prefix integration operator applies `C(0)=0`; subtracting the full radial
integral from the integrated u derivative applies `u(1)=0`. For an Ip
constraint, the current multiplier is eliminated algebraically on each sweep
from `C(1)=mu0*Ip/(2*pi)`. A beta constraint is another algebraic multiplier of
the complete pressure profile, so it does not add an inner nonlinear unknown.

The runtime always starts this local map from `u=C=0`, making a residual
evaluation independent of call history. Each Picard sweep is followed by a
dimensionless `(u, C)` defect test. Production uses the internal,
non-configurable tolerance `1e-6` and a hard limit of ten sweeps. Failure is
explicit; the source stage does not silently accept a truncated closure.

For native `coordinate="rho"`, this map is not solved as an inner
subproblem. PJ2/PJ3 instead advance `(s, ds/dr, u, C)` together and apply one
joint `1e-6` convergence test; see
[`rho-source-closure.md`](rho-source-closure.md).

`F_count>0` retains the previous optimized-F route. This is useful for
comparison and remains required by the Cxx and psin-coordinate implementations.

## Routes Benchmark

The comparison uses `Nr=32`, `Nt=16`, 51 uniform source samples where
applicable, cold continuation, Numba/Powell, ten warmups, and 100 timed solves.
Both implementations use the same reference-derived Ip and beta. This matters:
the older benchmark hard-coded `beta=0.02`, which did not describe its reference
profiles and caused correctly converged beta rows to fail physical
qualification.

| case | optimized F=6 (ms) | strict (ms) | old / strict |
| --- | ---: | ---: | ---: |
| PJ2 uniform both | 1.235 | 1.145 | 1.08 |
| PJ2 uniform Ip | 1.807 | 1.134 | 1.59 |
| PJ2 uniform beta | 1.183 | 1.158 | 1.02 |
| PJ2 uniform none | 1.299 | 1.170 | 1.11 |
| PJ2 grid both | 1.140 | 1.041 | 1.10 |
| PJ2 grid Ip | 1.721 | 1.059 | 1.62 |
| PJ2 grid beta | 1.110 | 1.066 | 1.04 |
| PJ2 grid none | 1.208 | 1.084 | 1.11 |
| PJ3 uniform both | 1.237 | 1.230 | 1.01 |
| PJ3 uniform Ip | 1.250 | 1.224 | 1.02 |
| PJ3 uniform beta | 1.252 | 1.335 | 0.94 |
| PJ3 uniform none | 1.212 | 1.299 | 0.93 |
| PJ3 grid both | 1.149 | 1.146 | 1.00 |
| PJ3 grid Ip | 1.158 | 1.141 | 1.01 |
| PJ3 grid beta | 1.167 | 1.242 | 0.94 |
| PJ3 grid none | 1.125 | 1.212 | 0.93 |

All 16 rows pass the outer residual and physical shape qualification. Removing
the six F coefficients reduces the outer unknown count from 18 to 12. PJ2's
median case time falls from 1.221 ms to 1.109 ms; PJ3's rises from 1.190 ms to
1.227 ms because its longer local closure outweighs the smaller outer solve in
the absolute-current branches.

The strict closure improves the quantities it owns:

| route | metric | optimized F=6 | strict |
| --- | --- | ---: | ---: |
| PJ2 | median prescribed-current relative L2 | 5.78e-6 | 4.30e-7 |
| PJ2 | maximum constrained Ip relative error | 1.04e-4 | 3.11e-16 |
| PJ2 | median FF_psi relative RMS | 1.04e-3 | 2.00e-4 |
| PJ3 | median prescribed-current relative L2 | 7.98e-5 | 1.21e-5 |
| PJ3 | maximum constrained Ip relative error | 1.20e-4 | 1.55e-16 |

PJ3's reference `FF_psi` error increases in this cross-grid table because
`jtotal`, geometry, and F are independently resampled from a 64-point reference
to the 32-point solve. Those resampled fields no longer satisfy the nonlinear
PJ3 identity exactly. The optimized-F route can stay closer to the original F
by relaxing pointwise current closure; strict PJ3 instead honors the supplied
resampled `jtotal`.

This interpretation is checked by a same-grid round trip through the production
Kernel. PJ2 and PJ3 respectively recover F with relative L2 errors
`2.45e-10` and `1.22e-10`, recover `psi_r` within `1.28e-8` and `1.26e-8`, and
close constrained Ip to machine precision. Thus the cross-grid PJ3 discrepancy
is a data-consistency effect, not an error in the strict radial equations.

## Reproduction

```bash
.venv/bin/python benchmarks/numba_routes.py \
  --scope full --case PJ2 --case PJ3 \
  --pj2-f-count 6 --repeat 100 --warmup 10 --no-write

.venv/bin/python benchmarks/numba_routes.py \
  --scope full --case PJ2 --case PJ3 \
  --pj2-f-count 0 --repeat 100 --warmup 10 --no-write
```
