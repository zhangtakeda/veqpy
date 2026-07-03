from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from helpers import MU0, pf_reference_profiles
from numpy.testing import assert_allclose

import veqlib.facade as native_facade
import veqpy.facade as facade
from veqlib.facade import KernelRecipe
from veqpy.kernel import NumbaKernel


def make_kernel_topology() -> facade.KernelTopology:
    return facade.KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=9,
    )


def tiny_kernel_boundary() -> facade.KernelBoundary:
    return facade.KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
    )


def tiny_kernel_source() -> facade.KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    current_profile, scaled_heat = pf_reference_profiles(psin)
    return facade.KernelSource(
        heat_profile=scaled_heat / MU0,
        current_profile=current_profile,
        Ip=3.0e6,
    )


def test_veqpy_facade_root_exports_numba_surface() -> None:
    assert facade.__all__ == [
        "Kernel",
        "NumbaKernel",
        "KernelBoundary",
        "KernelConfig",
        "KernelRecipe",
        "KernelSource",
        "KernelTopology",
        "SolveResult",
        "build",
    ]
    assert facade.Kernel is NumbaKernel
    assert facade.NumbaKernel is NumbaKernel
    assert facade.KernelTopology is native_facade.KernelTopology


def test_veqpy_facade_build_returns_numba_kernel() -> None:
    topology = make_kernel_topology()
    kernel = facade.build(
        topology=topology,
        recipe=facade.KernelRecipe(backend="numba", layout="degree"),
        config=facade.KernelConfig(method="levenberg-marquardt", initial="cold-zeros"),
    )

    assert isinstance(kernel, NumbaKernel)
    assert kernel.topology is topology
    assert kernel.recipe.backend == "numba"


def test_veqpy_facade_kernel_matches_numba_kernel_residual() -> None:
    topology = make_kernel_topology()
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(topology.x_size, dtype=np.float64)
    facade_kernel = facade.Kernel(
        topology=topology,
        recipe=facade.KernelRecipe(backend="numba", layout="degree"),
    )
    direct_kernel = NumbaKernel(
        topology=topology,
        recipe=facade.KernelRecipe(backend="numba", layout="degree"),
    )

    assert_allclose(
        facade_kernel.residual(x, boundary, source),
        direct_kernel.residual(x, boundary, source),
    )


def test_veqpy_facade_keeps_native_facade_dispatch_separate() -> None:
    assert isinstance(sys.modules.get("veqpy.facade"), ModuleType)
    assert native_facade.Kernel is not facade.Kernel

    with pytest.raises(ValueError, match="only supports KernelRecipe backend='cxx'"):
        native_facade.Kernel(
            topology=make_kernel_topology(),
            recipe=KernelRecipe(backend="numba", layout="degree"),
        )
