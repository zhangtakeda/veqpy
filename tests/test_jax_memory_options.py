from __future__ import annotations

import os
import sys

import numpy as np

from veqpy.engine.backend import JaxBackendOptions
from veqpy.engine.jax.config import apply_preimport_options
from veqpy.engine.jax.memory import copy_device_array_into, copy_device_array_to_numpy


def test_jax_preimport_options_set_environment_without_importing_jax(monkeypatch) -> None:
    sys.modules.pop("jax", None)
    for key in (
        "JAX_PLATFORM_NAME",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "JAX_ENABLE_X64",
    ):
        monkeypatch.delenv(key, raising=False)

    apply_preimport_options(
        JaxBackendOptions(
            platform="cpu",
            enable_x64=False,
            preallocate=False,
            mem_fraction=0.5,
            allocator="platform",
        )
    )

    assert "jax" not in sys.modules
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert os.environ["JAX_ENABLE_X64"] == "false"


class _FakeDeviceArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)
        self.blocked = False

    def block_until_ready(self):
        self.blocked = True
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


def test_jax_memory_helpers_copy_to_numpy_without_explicit_preblock() -> None:
    fake = _FakeDeviceArray(np.array([1.0, 2.0], dtype=np.float64))
    host = copy_device_array_to_numpy(fake)

    assert not fake.blocked
    assert np.array_equal(host, np.array([1.0, 2.0]))


def test_jax_memory_helpers_copy_into_caller_array() -> None:
    fake = _FakeDeviceArray(np.array([3.0, 4.0], dtype=np.float64))
    out = np.empty(2, dtype=np.float64)
    copy_device_array_into(fake, out)

    assert not fake.blocked
    assert np.array_equal(out, np.array([3.0, 4.0]))
