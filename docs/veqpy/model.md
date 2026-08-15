# Physical State and GEQDSK I/O

Physical State ownership belongs to `fusionprime-base`. VEQPy's materializer
constructs a frozen base `Geometry` and `Equilibrium` with SI roots:
`FF_psi`, `P_psi`, `psi_r`, `psi_rr`, `B0`, and `P0`.

`Geqdsk` remains a passive VEQ-specific interchange payload. Use
`veqpy.io.export_geqdsk(base_equilibrium, ...)` to write a base State without
making the old VEQPy model classes part of the public physical API.

Matplotlib is optional and is imported only by the demos or plotting helpers;
`import veqpy` does not import it.

The version-controlled fixtures are `data/SOLOVEV.geqdsk`,
`data/CHEASE.geqdsk`, and `data/EFIT.geqdsk`. Generated GEQDSK and figure files
are local artifacts.
