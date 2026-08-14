from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import PchipInterpolator

from benchmarks._common import (
    ROUTE_BENCHMARK_CONSTRAINTS,
    RouteBenchmarkSpec,
    route_kernel_case,
)
from veqpy import Kernel, KernelRecipe, KernelSource
from veqpy.kernels.abi.enums import PRESSURE_DERIVATIVE_BY_COORDINATE
from veqpy.kernels.numba_kernel import source_runtime

EXPLICIT_NODES = np.array(
    [
        0.0,
        0.03,
        0.08,
        0.15,
        0.26,
        0.40,
        0.55,
        0.68,
        0.78,
        0.85,
        0.90,
        0.94,
        0.965,
        0.974,
        0.982,
        0.993,
        1.0,
    ],
    dtype=np.float64,
)


def _explicit_case(route: str, coordinate: str):
    uniform = route_kernel_case(
        RouteBenchmarkSpec(route, coordinate, "uniform", "ip"),
        nr=16,
        nt=8,
        sample_count=101,
        pj2_f_count=6 if coordinate == "psin" and route in {"PJ2", "PJ3"} else 0,
    )
    uniform_nodes = np.linspace(0.0, 1.0, uniform.source.pressure_profile.size)
    pressure = PchipInterpolator(uniform_nodes, uniform.source.pressure_profile)(EXPLICIT_NODES)
    driver = PchipInterpolator(uniform_nodes, uniform.source.driver_profile)(EXPLICIT_NODES)
    topology = replace(
        uniform.topology,
        nodes="explicit",
        sample_count=None,
        key=None,
    )
    source = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: pressure},
        **{uniform.source.driver_name: driver},
        source_nodes=EXPLICIT_NODES,
        Ip=uniform.source.Ip,
        beta=uniform.source.beta,
    )
    return replace(uniform, topology=topology, source=source)


@pytest.mark.parametrize("route", tuple(ROUTE_BENCHMARK_CONSTRAINTS))
def test_native_explicit_rho_routes_retain_original_nodes(route: str) -> None:
    case = _explicit_case(route, "rho")
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace

        assert result.success
        assert_array_equal(runtime.plan.source_plan.source_nodes, EXPLICIT_NODES)
        assert_array_equal(workspace.source_coordinate_nodes, EXPLICIT_NODES)
        assert workspace.source_coordinate_nodes.size != case.topology.Nr
        assert workspace.pprime_spline_coeff.shape == (EXPLICIT_NODES.size - 1, 4)
        assert workspace.driver_spline_coeff.shape == (EXPLICIT_NODES.size - 1, 4)
        assert workspace.rho_state[1] <= 1.0e-6
    finally:
        kernel.close()


def test_native_explicit_rho_publishes_source_at_final_fixed_point_query() -> None:
    case = _explicit_case("PJ1", "rho")
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            case.source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        iterations = int(workspace.rho_state[0])

        assert iterations > 1
        expected_pprime = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_pprime,
        )(workspace.rho_query)
        expected_driver = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_driver,
        )(workspace.rho_query)
        assert_allclose(workspace.materialized_pprime_input, expected_pprime)
        assert_allclose(workspace.materialized_driver_input, expected_driver)
    finally:
        kernel.close()


@pytest.mark.parametrize("route", ("PJ2", "PJ3"))
def test_native_explicit_psin_current_closure_publishes_final_native_query(
    route: str,
) -> None:
    case = _explicit_case(route, "psin")
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        config=case.config,
    )
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            case.source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        expected_pressure = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_pprime,
        )(workspace.psin_query) / workspace.alpha_state[1]
        expected_driver = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_driver,
        )(workspace.psin_query)

        assert_array_equal(workspace.source_coordinate_nodes, EXPLICIT_NODES)
        assert_allclose(workspace.materialized_pprime_input, expected_pressure)
        assert_allclose(workspace.materialized_driver_input, expected_driver)
    finally:
        kernel.close()


@pytest.mark.parametrize("route", tuple(ROUTE_BENCHMARK_CONSTRAINTS))
def test_native_explicit_psin_routes_solve_from_retained_nodes(route: str) -> None:
    case = _explicit_case(route, "psin")
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba"),
        # This test qualifies retained-node source closure rather than Powell's
        # convergence basin for the now-physical P_psin derivative.
        config=replace(case.config, method="levenberg-marquardt"),
    )
    try:
        result = kernel.solve(case.boundary, case.source)
        workspace = kernel._impl._solver.runtime.source_workspace

        assert result.success
        assert_array_equal(workspace.source_coordinate_nodes, EXPLICIT_NODES)
    finally:
        kernel.close()


def test_native_explicit_r_is_materialized_directly_from_retained_nodes() -> None:
    case = _explicit_case("PJ1", "r")
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            case.source,
        )
        runtime = kernel._impl._solver.runtime
        expected_pressure = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_pprime,
        )(runtime.plan.grid_workspace.r)
        expected_driver = PchipInterpolator(
            EXPLICIT_NODES,
            runtime.plan.source_plan.scaled_driver,
        )(runtime.plan.grid_workspace.r)

        assert_array_equal(runtime.source_workspace.source_coordinate_nodes, EXPLICIT_NODES)
        assert_allclose(runtime.source_workspace.materialized_pprime_input, expected_pressure)
        assert_allclose(runtime.source_workspace.materialized_driver_input, expected_driver)
    finally:
        kernel.close()


def test_explicit_source_nodes_are_validated_as_a_runtime_axis() -> None:
    case = _explicit_case("PF", "r")
    source_kwargs = {
        "P_r": case.source.pressure_profile,
        case.source.driver_name: case.source.driver_profile,
        "Ip": case.source.Ip,
    }
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        with pytest.raises(ValueError, match="requires KernelSource.source_nodes"):
            kernel.residual(
                np.zeros(case.topology.x_size),
                case.boundary,
                KernelSource(**source_kwargs),
            )
        bad_nodes = EXPLICIT_NODES.copy()
        bad_nodes[5] = bad_nodes[4]
        with pytest.raises(ValueError, match="strictly increasing"):
            kernel.residual(
                np.zeros(case.topology.x_size),
                case.boundary,
                KernelSource(**source_kwargs, source_nodes=bad_nodes),
            )
    finally:
        kernel.close()


def test_explicit_source_count_is_runtime_data_not_topology() -> None:
    case = _explicit_case("PF", "r")
    assert case.topology.sample_count is None
    with pytest.raises(ValueError, match="sample_count is runtime KernelSource data"):
        replace(case.topology, sample_count=EXPLICIT_NODES.size, key=None)

    shorter_nodes = EXPLICIT_NODES[::2].copy()
    shorter_nodes[-1] = 1.0
    shorter_source = KernelSource(
        P_r=np.interp(shorter_nodes, EXPLICIT_NODES, case.source.pressure_profile),
        FF_r=np.interp(shorter_nodes, EXPLICIT_NODES, case.source.driver_profile),
        source_nodes=shorter_nodes,
        Ip=case.source.Ip,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        residual = kernel.residual(
            np.zeros(case.topology.x_size),
            case.boundary,
            shorter_source,
        )
        assert np.all(np.isfinite(residual))
    finally:
        kernel.close()


def test_equivalent_immutable_explicit_snapshot_reuses_bound_pchip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _explicit_case("PJ1", "rho")
    calls = 0
    original = source_runtime.build_explicit_source_interpolation_coefficients

    def recording_builder(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(nodes, values)

    monkeypatch.setattr(
        source_runtime,
        "build_explicit_source_interpolation_coefficients",
        recording_builder,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    x = np.zeros(case.topology.x_size, dtype=np.float64)
    try:
        kernel.residual(x, case.boundary, case.source)
        assert calls == 2
        kernel.residual(x, case.boundary, case.source)
        assert calls == 2
        kernel.residual(x, case.boundary, replace(case.source))
        assert calls == 2
        changed_pressure = case.source.pressure_profile.copy()
        changed_pressure[4] *= 1.0001
        kernel.residual(
            x,
            case.boundary,
            replace(case.source, P_rho=changed_pressure),
        )
        assert calls == 4
    finally:
        kernel.close()


def test_r_explicit_p_and_pi_share_retained_pchip_derivatives() -> None:
    case = _explicit_case("PI", "r")
    pressure = 6.0e3 + 1.2e4 * (1.0 - EXPLICIT_NODES**2) ** 2
    itor = case.source.Ip * EXPLICIT_NODES**2 * (1.0 + 0.2 * EXPLICIT_NODES**2) / 1.2
    source = KernelSource(
        p=pressure,
        itor=itor,
        source_nodes=EXPLICIT_NODES,
        Ip=case.source.Ip,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        grid_r = runtime.plan.grid_workspace.r
        mu0 = 4.0e-7 * np.pi

        expected_pprime = PchipInterpolator(EXPLICIT_NODES, pressure).derivative()(grid_r)
        expected_itor_r = PchipInterpolator(EXPLICIT_NODES, itor).derivative()(grid_r)
        expected_native_pprime = PchipInterpolator(
            EXPLICIT_NODES,
            pressure,
        ).derivative()(EXPLICIT_NODES)
        assert_allclose(workspace.materialized_pprime_input / mu0, expected_pprime)
        assert_allclose(workspace.materialized_driver_derivative / mu0, expected_itor_r)
        assert_allclose(
            runtime.plan.source_plan.scaled_pprime / mu0,
            expected_native_pprime,
        )
        assert_allclose(runtime.plan.source_plan.scaled_pressure / mu0, pressure)
    finally:
        kernel.close()


@pytest.mark.parametrize("coordinate", ("psin", "rho"))
def test_dynamic_explicit_p_uses_one_retained_primitive(coordinate: str) -> None:
    case = _explicit_case("PJ1", coordinate)
    pressure = 6.0e3 + 1.2e4 * (1.0 - EXPLICIT_NODES**2) ** 2
    source = KernelSource(
        p=pressure,
        jtor=case.source.driver_profile,
        source_nodes=EXPLICIT_NODES,
        Ip=case.source.Ip,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        query = workspace.psin_query if coordinate == "psin" else workspace.rho_query
        mu0 = 4.0e-7 * np.pi
        interpolant = PchipInterpolator(EXPLICIT_NODES, pressure)

        assert_allclose(
            runtime.plan.source_plan.scaled_pprime / mu0,
            interpolant.derivative()(EXPLICIT_NODES),
        )
        assert_allclose(
            workspace.materialized_pprime_input / mu0,
            interpolant.derivative()(query),
        )
        assert_allclose(runtime.plan.source_plan.scaled_pressure / mu0, pressure)
    finally:
        kernel.close()


@pytest.mark.parametrize("coordinate", ("r", "psin", "rho"))
def test_pressure_derivative_is_in_selected_source_coordinate(coordinate: str) -> None:
    case = _explicit_case("PJ1", coordinate)
    native_pprime = 2.0e4 * (1.0 - 0.7 * EXPLICIT_NODES**2)
    source = KernelSource(
        **{PRESSURE_DERIVATIVE_BY_COORDINATE[coordinate]: native_pprime},
        p0=4.0e3,
        jtor=case.source.driver_profile,
        source_nodes=EXPLICIT_NODES,
        Ip=case.source.Ip,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        if coordinate == "r":
            query = runtime.plan.grid_workspace.r
        elif coordinate == "psin":
            query = workspace.psin_query
        else:
            query = workspace.rho_query
        expected_native = PchipInterpolator(EXPLICIT_NODES, native_pprime)(query)
        mu0 = 4.0e-7 * np.pi

        assert_allclose(workspace.materialized_pprime_input / mu0, expected_native)
        if coordinate == "rho":
            assert_allclose(
                workspace.rho_pprime / mu0,
                expected_native * workspace.rho_derivative,
            )
    finally:
        kernel.close()


def test_pp_r_explicit_uses_pchip_and_axis_extension_derivatives() -> None:
    case = _explicit_case("PP", "r")
    psi_r = EXPLICIT_NODES * (1.0 + 0.3 * EXPLICIT_NODES**2)
    source = KernelSource(
        P_r=case.source.pressure_profile,
        psi_r=psi_r,
        source_nodes=EXPLICIT_NODES,
        Ip=case.source.Ip,
    )
    kernel = Kernel(topology=case.topology, recipe=KernelRecipe(backend="numba"))
    try:
        kernel.residual(
            np.zeros(case.topology.x_size, dtype=np.float64),
            case.boundary,
            source,
        )
        runtime = kernel._impl._solver.runtime
        workspace = runtime.source_workspace
        r = runtime.plan.grid_workspace.r
        n_fix = int(np.searchsorted(r, runtime.fix_r))
        expected_r = PchipInterpolator(EXPLICIT_NODES, psi_r)(r)
        expected_pchip_rr = PchipInterpolator(EXPLICIT_NODES, psi_r).derivative()(r)
        expected_rr = expected_pchip_rr.copy()

        if n_fix > 0:
            anchor0 = n_fix
            anchor1 = n_fix + 1
            r0 = r[anchor0]
            x0 = r0 * r0
            x1 = r[anchor1] * r[anchor1]
            slope0 = expected_r[anchor0] / r0
            slope1 = expected_r[anchor1] / r[anchor1]
            slope_gradient = (slope1 - slope0) / (x1 - x0)
            for i in range(n_fix):
                x = r[i] * r[i]
                expected_r[i] = r[i] * (slope0 + slope_gradient * (x - x0))
                expected_rr[i] = slope0 + slope_gradient * (3.0 * x - x0)

        root_fields = runtime.residual_workspace.root_fields
        assert_allclose(workspace.materialized_driver_derivative, expected_pchip_rr)
        # Ip-constrained PP leaves the supplied psi_r scale in the root fields.
        assert_allclose(root_fields[1], expected_r)
        assert_allclose(root_fields[2], expected_rr)
    finally:
        kernel.close()


def test_explicit_source_nodes_are_rejected_by_cxx_before_build() -> None:
    case = _explicit_case("PF", "r")
    with pytest.raises(NotImplementedError, match="nodes='explicit'.*backend='numba'"):
        Kernel(topology=case.topology, recipe=KernelRecipe(backend="cxx"))
