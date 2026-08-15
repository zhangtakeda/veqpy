from __future__ import annotations

import numpy as np

from veqpy import Kernel, KernelConfig, KernelInput, KernelTopology


def _topology() -> KernelTopology:
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2, 2),
        Nr=8,
        Nt=12,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=8,
    )


def _input(pressure_code: int) -> KernelInput:
    value = KernelInput.allocate(_topology())
    value.a = 0.9
    value.R0 = 3.0
    value.B0 = 5.0
    value.kappa_lcfs = 1.5
    value.pressure_code = pressure_code
    value.p0 = 1.0e3
    value.Ip = 1.0e6
    value.driver[:] = -0.042
    if pressure_code == 0:
        value.pressure[:] = value.p0 - 100.0 * np.linspace(0.0, 1.0, value.pressure.size)
    else:
        value.pressure[:] = -100.0
        value.pressure_derivative = np.full(value.pressure.size, -100.0, dtype=np.float64)
    return value


def test_public_input_pressure_codes_are_numerically_equivalent() -> None:
    topology = _topology()
    kernels = [
        Kernel(
            topology=topology,
            input=_input(code),
            config=KernelConfig(max_evaluations=800),
        )
        for code in (0, 1)
    ]
    try:
        outputs = [kernel.solve() for kernel in kernels]
    finally:
        for kernel in kernels:
            kernel.close()
    assert all(output.success for output in outputs)
    np.testing.assert_allclose(outputs[0].x, outputs[1].x, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(outputs[0].raw, outputs[1].raw, rtol=0.0, atol=2.0e-8)
