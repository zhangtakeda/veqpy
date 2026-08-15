# VEQPy architecture

VEQPy is the fixed-boundary equilibrium Module in the FusionPRIME workflow.
Its physical input is a frozen `fusionprime_base.Plasma`; its physical output
is a new frozen `fusionprime_base.Equilibrium` carried by `VEQRecord`.

```text
frozen Plasma
    -> VEQ Adapter
    -> private four-record kernel ABI
    -> backend solver
    -> optional Equilibrium materialization
    -> VEQRecord
```

The public package is deliberately small: `VEQ`, `VEQRecord`, `build`, and
the one-shot `solve` helper. The private ABI is constructed by the Module and
Adapter, so application code cannot select legacy case/result objects or
compile-time source grids.

`VEQ` follows the base `@module` lifecycle and JVP contract. A derivative
scratch Module owns independent input/output storage and suppresses both Rich
diagnostics and reports. Pure numerical helpers live under `veqpy.numerics`;
physical State ownership remains in FusionPRIME-base.
