# VEQPy architecture

VEQPy is the fixed-boundary equilibrium Module in the FusionPRIME workflow.
Its physical inputs are standalone `boundary`, `source`, and `targets`
dictionaries; its physical output is a new frozen
`fusionprime_base.Equilibrium` carried by `VEQRecord`.

```text
boundary + source + targets
    -> strict VEQ Adapter (copy only; no source-to-source remap)
    -> private four-record kernel ABI
    -> backend solver
    -> optional Equilibrium materialization
    -> VEQRecord
```

The public package is deliberately small: `VEQ`, `VEQRecord`, `build`, and
the one-shot `solve` helper. The private ABI is constructed by the Module and
Adapter, so application code cannot select legacy case/result objects or
compile-time source grids.

`VEQ` declares these three physical ports through the base `@module` lifecycle.
Plain dictionaries do not claim a generic tangent layout, so they do not bind
the base automatic Module finite-difference linearization. Kernel residual
JVP/Jacobian support remains a lower-level contract; a complete VEQ solve-map
derivative requires implicit input linearization and is not fabricated by the
Adapter. Pure numerical helpers live under `veqpy.numerics`; physical State
ownership remains in FusionPRIME-base.
