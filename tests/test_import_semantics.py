from __future__ import annotations

import importlib


def test_direct_current_module_imports_are_available() -> None:
    modules = [
        "veqpy.api",
        "veqpy.adapter",
        "veqpy.io",
        "veqpy.module",
        "veqpy.kernels.contracts",
        "veqpy.kernels.lowering",
        "veqpy.kernels.cxx_kernel.builder",
        "veqpy.kernels.numba_kernel.packed_layout",
        "veqpy.model.geqdsk",
        "veqpy.numerics.interpolate",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name)
