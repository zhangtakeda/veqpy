# VEQPy 2.x architecture

VEQPy is the fixed-boundary equilibrium Module in the FusionPRIME workflow.
Its physical input is a frozen `fusionprime_base.Plasma`; its physical output
is a new frozen `fusionprime_base.Equilibrium` carried by `VEQRecord`.

```text
frozen Plasma
    -> VEQAdapter (validate/remap/fill)
    -> Kernel (Topology/Input/Config/Output)
    -> materializer
    -> base Equilibrium
    -> VEQRecord
```

`VEQ` is decorated with the base `@module` contract and uses the base default
forward finite-difference linearization. The scratch Module returned by
`new_runtime()` owns an independent Kernel, so derivative trials cannot
overwrite the primal output, input Plasma, or Record.

The public package exports `VEQ`, `VEQRecord`, `Geqdsk`, and the four named
Kernel data types. VEQPy's former reactive model objects are private numerical
implementation details and are not re-exported as physical State types.
