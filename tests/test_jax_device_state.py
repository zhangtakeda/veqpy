from __future__ import annotations

import sys

import numpy as np

from veqpy.engine.jax.state import JaxDeviceState, JaxRuntime, JaxStaticSpec


def _spec() -> JaxStaticSpec:
    return JaxStaticSpec(
        route_key=("PF", "rho", "grid"),
        nr=8,
        nt=8,
        k_max=1,
        l_max=3,
        m_max=2,
        x_size=9,
        profile_names=("h", "k", "s1"),
        active_profile_ids=(0, 2, 6),
        active_lengths=(2, 2, 2),
        residual_block_codes=(0, 2, 6),
        residual_block_orders=(0, 0, 1),
        residual_block_radial_powers=(0, 0, 1),
    )


def test_jax_static_spec_is_hash_stable_and_backend_private() -> None:
    spec = _spec()

    assert hash(spec) == hash(_spec())
    assert spec.route_key == ("PF", "rho", "grid")
    assert "jax" not in sys.modules


def test_jax_device_state_and_runtime_are_private_containers() -> None:
    state = JaxDeviceState(leaves={"rho": np.arange(3, dtype=np.float64)})
    runtime = JaxRuntime(static_spec=_spec(), device_state=state)

    assert runtime.static_spec == _spec()
    assert np.array_equal(runtime.device_state.leaves["rho"], np.arange(3, dtype=np.float64))
    assert runtime.compiled_residual is None
    assert "jax" not in sys.modules
