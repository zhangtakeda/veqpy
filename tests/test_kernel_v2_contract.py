from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from veqpy import Kernel, KernelConfig, KernelInput, KernelOutput, KernelTopology
from veqpy.adapter import VEQAdapter
from veqpy.demo_case import make_demo_plasma


def _topology(*, nodes: str = "uniform", source_capacity: int | None = None) -> KernelTopology:
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
        nodes=nodes,
        constraint="ip",
        sample_count=None if nodes == "explicit" else 8,
        source_capacity=source_capacity,
    )


def test_four_public_buffers_have_stable_numeric_shapes_and_ownership() -> None:
    topology = _topology(source_capacity=12)
    input_buffer = KernelInput.allocate(topology)
    output_buffer = KernelOutput.allocate(topology)
    config = KernelConfig(max_evaluations=800)

    assert topology.source_capacity == 12
    assert topology.x_size == output_buffer.x.size
    with pytest.raises(FrozenInstanceError):
        topology.Nr = 9  # type: ignore[misc]
    assert all(field.type is not str for field in fields(config))
    assert all(field.name != "case_name" for field in fields(input_buffer))
    for name in ("pressure", "driver", "source_nodes", "c_lcfs", "s_lcfs"):
        array = getattr(input_buffer, name)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert array.flags.owndata
    for name in ("x", "raw", "scaled", "alpha", "psin", "psin_r", "psin_rr", "FF_psi", "P_psi"):
        array = getattr(output_buffer, name)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert array.flags.owndata


def test_numba_kernel_reuses_input_output_and_array_identity() -> None:
    topology = _topology()
    input_buffer = KernelInput.allocate(topology)
    output_buffer = KernelOutput.allocate(topology)
    adapter = VEQAdapter(topology, input_buffer)
    adapter.fill(make_demo_plasma())
    kernel = Kernel(
        topology=topology,
        input=input_buffer,
        output=output_buffer,
        config=KernelConfig(max_evaluations=800),
        backend="numba",
    )
    try:
        kernel.prepare()
        input_ids = {name: id(getattr(input_buffer, name)) for name in ("pressure", "driver", "source_nodes")}
        output_ids = {name: id(getattr(output_buffer, name)) for name in ("x", "raw", "scaled", "psin")}
        first = kernel.solve()
        first_x = first.x.copy()
        second = kernel.solve()
        assert first is output_buffer
        assert second is output_buffer
        assert first.x is second.x
        np.testing.assert_allclose(first.x, first_x, rtol=0.0, atol=2.0e-5)
        assert input_ids == {name: id(getattr(input_buffer, name)) for name in input_ids}
        assert output_ids == {name: id(getattr(output_buffer, name)) for name in output_ids}
        assert not hasattr(kernel, "history")
        assert not hasattr(kernel, "result")
        assert np.isfinite(second.raw_norm)
        equilibrium = kernel.build_equilibrium()
        assert equilibrium.is_frozen
        assert not np.shares_memory(equilibrium.psin, output_buffer.psin)
    finally:
        kernel.close()


def test_explicit_source_count_uses_prepare_capacity_without_resize() -> None:
    topology = _topology(nodes="explicit", source_capacity=16)
    input_buffer = KernelInput.allocate(topology)
    nodes = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    input_buffer.source_count = nodes.size
    input_buffer.source_nodes[: nodes.size] = nodes
    input_buffer.pressure[: nodes.size] = -1.0e5
    input_buffer.driver[: nodes.size] = -1.0
    input_buffer.Ip = 3.0e6
    input_buffer.clear_unused_source_tail()
    kernel = Kernel(topology=topology, input=input_buffer, config=KernelConfig(max_evaluations=800))
    try:
        pressure_id = id(input_buffer.pressure)
        output = kernel.solve()
        assert output.success
        assert input_buffer.source_count == 8
        assert id(input_buffer.pressure) == pressure_id
        assert np.all(input_buffer.pressure[8:] == 0.0)
        assert np.all(input_buffer.driver[8:] == 0.0)
        assert np.all(input_buffer.source_nodes[8:] == 0.0)
    finally:
        kernel.close()


def test_capacity_rejects_overflow_before_backend_execution() -> None:
    topology = _topology(source_capacity=8)
    input_buffer = KernelInput.allocate(topology)
    kernel = Kernel(topology=topology, input=input_buffer)
    input_buffer.source_count = 9
    try:
        with pytest.raises(ValueError, match="source_count exceeds"):
            kernel.solve()
    finally:
        kernel.close()
