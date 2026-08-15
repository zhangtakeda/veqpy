from __future__ import annotations

import numpy as np

from veqpy import Kernel, KernelConfig, KernelInput, KernelTopology
from veqpy.adapter import VEQAdapter
from veqpy.demo_case import make_demo_plasma


def test_residual_jvp_writes_caller_buffer_without_changing_output_identity() -> None:
    topology = KernelTopology(
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
    input_buffer = KernelInput.allocate(topology)
    VEQAdapter(topology, input_buffer).fill(make_demo_plasma())
    kernel = Kernel(topology=topology, input=input_buffer, config=KernelConfig(max_evaluations=800))
    try:
        output = kernel.solve()
        output_ids = {name: id(getattr(output, name)) for name in ("x", "raw", "scaled")}
        x = output.x.copy()
        direction = np.linspace(-1.0, 1.0, topology.x_size, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        jvp = np.empty(topology.x_size, dtype=np.float64)
        kernel.residual_jvp_into(jvp, x, direction)

        step = 1.0e-6 * (1.0 + np.linalg.norm(x))
        oracle = (
            kernel.residual(x + step * direction) - kernel.residual(x - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(jvp, oracle, rtol=3.0e-5, atol=3.0e-7)
        assert output_ids == {name: id(getattr(output, name)) for name in output_ids}
    finally:
        kernel.close()
