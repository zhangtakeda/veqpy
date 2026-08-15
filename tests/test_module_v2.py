from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
from fusionprime_base import Equilibrium
from fusionprime_base.testing import check_jvp

from veqpy import VEQ, KernelConfig, KernelTopology, VEQRecord
from veqpy.demo_case import make_demo_plasma


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


def test_veq_run_consumes_frozen_plasma_and_returns_new_base_equilibrium() -> None:
    plasma = make_demo_plasma()
    original = np.array(plasma.equilibrium.P_psi, copy=True)
    module = VEQ(topology=_topology(), config=KernelConfig(max_evaluations=800))
    try:
        record = module.run(plasma=plasma)
        assert type(record) is VEQRecord
        assert is_dataclass(record)
        assert record.accepted
        assert record.materialized
        assert type(record.equilibrium) is Equilibrium
        assert record.equilibrium is not plasma.equilibrium
        assert record.equilibrium.is_frozen
        assert record.equilibrium.geometry.is_frozen
        assert np.array_equal(plasma.equilibrium.P_psi, original)
        assert record.source_count == 8
        assert record.route == "PF"
    finally:
        module.close()


def test_materialize_false_and_lifecycle_are_explicit() -> None:
    module = VEQ(topology=_topology())
    plasma = make_demo_plasma()
    try:
        record = module.run(plasma=plasma, materialize=False)
        assert record.accepted
        assert record.equilibrium is None
        assert not record.materialized
        module.clear()
        assert module.kernel.prepared
        module.prepare()
        runtime = module.new_runtime()
        try:
            assert runtime is not module
            assert runtime.kernel is not module.kernel
        finally:
            runtime.close()
    finally:
        module.close()


def test_base_forward_fd_jvp_uses_isolated_runtime_and_centered_oracle() -> None:
    module = VEQ(topology=_topology(), config=KernelConfig(max_evaluations=800))
    plasma = make_demo_plasma()
    try:
        module.run(plasma=plasma)
        point = module.linearization.point
        assert point is not None
        direction = point.input_layout.zeros()
        direction.data[0] = 1.0e-3
        before = module.kernel.output.x.copy()

        def evaluate(trial_plasma: object) -> tuple[object, ...]:
            runtime = module.new_runtime()
            try:
                trial = runtime.run(plasma=trial_plasma)  # type: ignore[arg-type]
                return (trial.equilibrium,)
            finally:
                runtime.close()

        check = check_jvp(
            module.linearization,
            direction,
            evaluate=evaluate,
            relative_step=1.0e-4,
            rtol=1.0e-2,
            atol=1.0e-6,
        )
        assert check.passed
        assert np.array_equal(module.kernel.output.x, before)
    finally:
        module.close()
