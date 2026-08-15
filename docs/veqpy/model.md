# Physical State and file payloads

Physical State ownership belongs to `fusionprime-base`. VEQPy materialization
returns a frozen base `Equilibrium` with the roots required by the Module
contract. Numerical grid, interpolation, calculus, and equilibrium resampling
helpers are private implementation details under `veqpy.numerics`.

GEQDSK is a file payload, not a VEQPy physical-state model. Pure
`Geqdsk`/`load_geqdsk`/`save_geqdsk` support lives in
`fusionprime_base.io.geqdsk`; it only reads and writes the file payload and
does not convert to or from `Equilibrium`.

VEQPy has no plotting layer, Matplotlib dependency, or visualization API.
The checked-in GEQDSK files under `data/` are parser fixtures for the base
I/O smoke gate.
