from __future__ import annotations

import numpy as np
import pytest

from veqpy import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    TopologyError,
)


def make_topology(**overrides: object) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 3,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
        "Nr": 8,
        "Nt": 8,
        "route": "PF",
        "coordinate": "psin",
        "nodes": "uniform",
        "ip_constraint": True,
        "sample_count": 9,
    }
    params.update(overrides)
    return KernelTopology(**params)  # type: ignore[arg-type]


def tiny_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=(float(np.arcsin(0.2)),),
    )


def tiny_source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    return KernelSource(
        heat_profile=1.0e6 + 0.2e6 * psin,
        current_profile=1.0 + 0.1 * psin,
        Ip=3.0e6,
    )


def test_l_max_accepts_capacity_values() -> None:
    inferred = make_topology(L_max=None)
    expanded = make_topology(L_max=inferred.L_max + 3)

    assert inferred.L_max == 2
    assert expanded.L_max == 5
    with pytest.raises(TopologyError, match="L_max"):
        make_topology(L_max=1)


def test_kernel_variant_is_count_only_and_returns_self() -> None:
    topology = make_topology(L_max=5, M_max=3, K_max=3)
    kernel = Kernel(topology=topology, recipe=KernelRecipe(backend="numba"))
    old_topology = kernel.topology

    assert kernel.variant(h_count=6, c_counts=(2, 1), s_counts=None) is kernel

    assert old_topology.h_count == 2
    assert kernel.topology.h_count == 6
    assert kernel.topology.c_counts == (2, 1)
    assert kernel.topology.s_counts == old_topology.s_counts
    assert kernel.topology.Nr == old_topology.Nr
    assert kernel.topology.Nt == old_topology.Nt
    assert kernel.topology.route == old_topology.route
    assert kernel.topology.L_max == old_topology.L_max
    assert kernel.topology.M_max == old_topology.M_max
    assert kernel.topology.K_max == old_topology.K_max


def test_kernel_variant_rejects_counts_beyond_fixed_capacity() -> None:
    kernel = Kernel(
        topology=make_topology(h_count=2, L_max=5, M_max=2, K_max=2),
        recipe=KernelRecipe(backend="numba"),
    )

    kernel.variant(h_count=6)
    assert kernel.topology.h_count == 6
    with pytest.raises(TopologyError, match="L_max"):
        kernel.variant(h_count=7)
    with pytest.raises(TopologyError, match="M_max"):
        kernel.variant(c_counts=(1, 1, 1, 1))


def test_kernel_variant_noop_clears_runtime_state_but_keeps_history() -> None:
    kernel = Kernel(
        topology=make_topology(L_max=5, M_max=2, K_max=2),
        recipe=KernelRecipe(backend="numba"),
        config=KernelConfig(method="powell", max_evaluations=2),
    )
    result = kernel.solve(tiny_boundary(), tiny_source(), continuation="cold")
    prepared = kernel.prepare()

    assert kernel.result is not None
    assert len(kernel.history) == 1
    assert prepared.topology == kernel.topology
    assert result.x.shape == (kernel.x_size,)

    kernel.variant()

    assert kernel.result is None
    assert len(kernel.history) == 1
    with pytest.raises(RuntimeError, match=r"build_equilibrium"):
        kernel.build_equilibrium()
    refreshed = kernel.prepare(dry_run=True)
    assert refreshed.topology == kernel.topology


def test_kernel_variant_reuses_numba_workspace_for_contained_counts() -> None:
    kernel = Kernel(
        topology=make_topology(h_count=2, L_max=5, M_max=2, K_max=2),
        recipe=KernelRecipe(backend="numba"),
    )
    runtime = kernel._impl._solver.runtime  # type: ignore[attr-defined]
    workspace_ids = (
        id(runtime.profile_workspace),
        id(runtime.geometry_workspace),
        id(runtime.source_workspace),
        id(runtime.residual_workspace),
    )

    kernel.variant(h_count=6)
    runtime_after = kernel._impl._solver.runtime  # type: ignore[attr-defined]

    assert runtime_after is runtime
    assert (
        id(runtime_after.profile_workspace),
        id(runtime_after.geometry_workspace),
        id(runtime_after.source_workspace),
        id(runtime_after.residual_workspace),
    ) == workspace_ids
    assert kernel.x_size == 13
    assert runtime_after._case is None


def test_kernel_variant_rejects_source_family_invalid_counts() -> None:
    pj2 = Kernel(
        topology=make_topology(
            route="PJ2",
            coordinate="rho",
            nodes="uniform",
            psin_count=0,
            F_count=2,
            L_max=5,
        ),
        recipe=KernelRecipe(backend="numba"),
    )
    with pytest.raises(TopologyError, match="F_count"):
        pj2.variant(F_count=0)

    psin = Kernel(
        topology=make_topology(psin_count=2, L_max=5),
        recipe=KernelRecipe(backend="numba"),
    )
    with pytest.raises(TopologyError, match="psin_count"):
        psin.variant(psin_count=0)


def test_cxx_variant_is_not_supported() -> None:
    kernel = Kernel(topology=make_topology(), recipe=KernelRecipe(backend="cxx"))

    with pytest.raises(NotImplementedError, match="Numba backend"):
        kernel.variant(h_count=1)


def test_cxx_rejects_capacity_style_l_max() -> None:
    topology = make_topology(L_max=5)

    with pytest.raises(TopologyError, match="capacity-style L_max"):
        Kernel(topology=topology, recipe=KernelRecipe(backend="cxx"))
