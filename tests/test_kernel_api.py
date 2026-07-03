from __future__ import annotations

import numpy as np
import pytest
from helpers import MU0, pf_reference_profiles
from numpy.testing import assert_allclose

from veqlib.facade import KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqpy.kernel import NumbaKernel
from veqpy.model import Boundary, Grid, Problem
from veqpy.operator import Operator


def make_kernel_topology(**overrides: object) -> KernelTopology:
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


def tiny_kernel_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
    )


def tiny_kernel_source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    current_profile, scaled_heat = pf_reference_profiles(psin)
    return KernelSource(
        heat_profile=scaled_heat / MU0,
        current_profile=current_profile,
        Ip=3.0e6,
    )


def tiny_legacy_operator(topology: KernelTopology) -> Operator:
    source = tiny_kernel_source()
    boundary = tiny_kernel_boundary()
    return Operator(
        Grid(
            Nr=topology.Nr,
            Nt=topology.Nt,
            L_max=topology.L_max,
            M_max=topology.M_max,
            K_max=topology.K_max,
            quadrature_scheme=topology.quadrature,
            calculus_scheme=topology.calculus,
        ),
        Problem(
            route=topology.route,
            coordinate=topology.coordinate,
            nodes=topology.nodes,
            active_profiles=dict(topology.active_profiles),
            boundary=Boundary(
                a=boundary.a,
                R0=boundary.R0,
                Z0=boundary.Z0,
                B0=boundary.B0,
                ka=boundary.ka,
                c_offsets=boundary.c_offsets,
                s_offsets=boundary.s_offsets,
            ),
            heat_input=source.heat_profile,
            current_input=source.current_profile,
            Ip=source.Ip,
        ),
    )


def test_numba_kernel_recipe_validation_and_public_surface() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)

    assert kernel.x_size == topology.x_size
    assert kernel.recipe.backend == "numba"
    assert kernel.recipe.layout == "degree"
    assert kernel.prepare() is None

    with pytest.raises(ValueError, match="layout='degree'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="numba", layout="family"))
    with pytest.raises(ValueError, match="backend='numba'"):
        NumbaKernel(topology=topology, recipe=KernelRecipe(backend="cxx", layout="degree"))


def test_numba_kernel_residual_matches_legacy_operator_and_validates_buffers() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    legacy_operator = tiny_legacy_operator(topology)
    x = np.zeros(kernel.x_size, dtype=np.float64)

    residual = kernel.residual(x, boundary, source)
    assert_allclose(residual, legacy_operator.residual_var(x))

    out = np.empty(kernel.x_size, dtype=np.float64)
    kernel.residual_into(out, x.tolist(), boundary, source)
    assert_allclose(out, residual)

    with pytest.raises(ValueError, match="x must have shape"):
        kernel.residual(np.zeros(kernel.x_size + 1, dtype=np.float64), boundary, source)
    with pytest.raises(TypeError, match="dtype float64"):
        kernel.residual_into(np.empty(kernel.x_size, dtype=np.float32), x, boundary, source)
    with pytest.raises(ValueError, match="C-contiguous"):
        noncontiguous_out = np.empty((kernel.x_size, 2), dtype=np.float64)[:, 0]
        kernel.residual_into(noncontiguous_out, x, boundary, source)


def test_numba_kernel_solve_result_lifecycle_and_equilibrium_snapshot() -> None:
    topology = make_kernel_topology()
    kernel = NumbaKernel(topology=topology)
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()

    result = kernel.solve(
        boundary,
        source,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_evaluations=2,
        ),
    )

    assert result.x.shape == (kernel.x_size,)
    assert result.raw.shape == (kernel.x_size,)
    assert result.scaled.shape == (kernel.x_size,)
    assert result.alpha.shape == (2,)
    assert kernel.result is result
    assert kernel.history == [result]
    assert_allclose(kernel.build_equilibrium(result.x).Ip, kernel.build_equilibrium().Ip)

    kernel.clear()
    assert kernel.result is None
    assert kernel.history == []


def test_numba_kernel_jvp_and_jacobian_are_explicitly_unimplemented() -> None:
    kernel = NumbaKernel(topology=make_kernel_topology())
    boundary = tiny_kernel_boundary()
    source = tiny_kernel_source()
    x = np.zeros(kernel.x_size, dtype=np.float64)

    with pytest.raises(NotImplementedError):
        kernel.jvp(x, x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jvp_into(np.empty_like(x), x, x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jacobian(x, boundary, source)
    with pytest.raises(NotImplementedError):
        kernel.jacobian_into(np.empty((kernel.x_size, kernel.x_size)), x, boundary, source)
