from __future__ import annotations

from veqlib import benchmark_route_topology_matrix as matrix


def test_route_matrix_enumerates_benchmark_uniform_cases() -> None:
    benchmark = matrix._benchmark_module()

    specs = matrix._iter_route_specs(benchmark, include_grid=False)

    assert len(specs) == 46
    assert {spec.input_kind for spec in specs} == {"uniform"}
    assert "PF:psin:uniform:Ip" in {matrix._spec_selector(spec) for spec in specs}
    assert "PQ:rho:uniform:Ip_beta" in {matrix._spec_selector(spec) for spec in specs}


def test_route_matrix_can_extend_to_grid_topology_cases() -> None:
    benchmark = matrix._benchmark_module()

    specs = matrix._iter_route_specs(benchmark, include_grid=True)

    assert len(specs) == 92
    assert sum(spec.input_kind == "grid" for spec in specs) == 46


def test_route_matrix_builds_mvp_topology_from_benchmark_case() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PF",
        coordinate="psin",
        constraint="Ip",
        input_kind="uniform",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.route == "PF"
    assert topology.coordinate == "psin"
    assert topology.nodes == "uniform"
    assert topology.constraint == "Ip"
    assert topology.psin_count > 0
    assert topology.F_count == 0
    assert topology.sample_count == benchmark.TEST_SOURCE_SAMPLE_COUNT
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_source_owned_pf_grid_topology() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PF",
        coordinate="rho",
        constraint="beta",
        input_kind="grid",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.route == "PF"
    assert topology.coordinate == "rho"
    assert topology.nodes == "grid"
    assert topology.constraint == "beta"
    assert topology.psin_count == 0
    assert topology.source_active_family == "none"
    assert topology.sample_count == benchmark.TEST_GRID.Nr
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_tracks_pp_psin_parameterization() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PP",
        coordinate="psin",
        constraint="Ip_beta",
        input_kind="uniform",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.source_active_family == "psin"
    assert topology.psin_count > 0
    assert topology.source_parameterization == "sqrt_psin"
    assert topology.source_uses_ip_constraint is True
    assert topology.source_uses_beta_constraint is True
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_native_pi_profile_owned_topology() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PI",
        coordinate="psin",
        constraint="Ip_beta",
        input_kind="uniform",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.route == "PI"
    assert topology.source_active_family == "psin"
    assert topology.psin_count > 0
    assert topology.source_parameterization == "identity"
    assert topology.source_uses_ip_constraint is True
    assert topology.source_uses_beta_constraint is True
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_native_pj1_source_owned_topology() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PJ1",
        coordinate="rho",
        constraint="Ip_beta",
        input_kind="grid",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.route == "PJ1"
    assert topology.source_active_family == "none"
    assert topology.psin_count == 0
    assert topology.source_parameterization == "identity"
    assert topology.source_uses_ip_constraint is True
    assert topology.source_uses_beta_constraint is True
    assert topology.sample_count == benchmark.TEST_GRID.Nr
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_tracks_pj2_active_f_ownership() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PJ2",
        coordinate="psin",
        constraint="Ip_beta",
        input_kind="uniform",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.source_active_family == "F"
    assert topology.psin_count == 0
    assert topology.F_count > 0
    assert topology.source_parameterization == "identity"
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_native_pj2_one_pass_topology() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PJ2",
        coordinate="psin",
        constraint="Ip_beta",
        input_kind="grid",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.route == "PJ2"
    assert topology.source_active_family == "F"
    assert topology.psin_count == 0
    assert topology.F_count > 0
    assert topology.source_uses_ip_constraint is True
    assert topology.source_uses_beta_constraint is True
    assert topology.sample_count == benchmark.TEST_GRID.Nr
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_uses_lm_for_pj2_psin_grid_ip_runtime() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PJ2",
        coordinate="psin",
        constraint="Ip",
        input_kind="grid",
    )
    nearby_spec = benchmark.BenchmarkCaseSpec(
        mode="PJ2",
        coordinate="psin",
        constraint="Ip_beta",
        input_kind="grid",
    )

    assert matrix._cxx_solver_method_for_spec(spec) == matrix.SOLVER_METHOD_LEVENBERG_MARQUARDT
    assert matrix._cxx_solver_method_for_spec(nearby_spec) == matrix.SOLVER_METHOD_POWELL


def test_route_matrix_grid_cases_use_grid_sample_count() -> None:
    benchmark = matrix._benchmark_module()
    spec = benchmark.BenchmarkCaseSpec(
        mode="PQ",
        coordinate="rho",
        constraint="beta",
        input_kind="grid",
    )

    topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

    assert warnings == ()
    assert topology.nodes == "grid"
    assert topology.sample_count == benchmark.TEST_GRID.Nr
    assert topology.source_active_family == "none"
    topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_native_pq_rho_topology() -> None:
    benchmark = matrix._benchmark_module()
    for input_kind in ("uniform", "grid"):
        spec = benchmark.BenchmarkCaseSpec(
            mode="PQ",
            coordinate="rho",
            constraint="Ip_beta",
            input_kind=input_kind,
        )

        topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

        assert warnings == ()
        assert topology.route == "PQ"
        assert topology.coordinate == "rho"
        assert topology.source_active_family == "none"
        assert topology.psin_count == 0
        assert topology.F_count == 0
        topology.validate_supported_for_veqlib_mvp()


def test_route_matrix_builds_native_pq_psin_topology() -> None:
    benchmark = matrix._benchmark_module()
    expected_active_family = {"uniform": "psin", "grid": "none"}
    for input_kind in ("uniform", "grid"):
        spec = benchmark.BenchmarkCaseSpec(
            mode="PQ",
            coordinate="psin",
            constraint="Ip_beta",
            input_kind=input_kind,
        )

        topology, warnings = matrix._topology_from_spec(benchmark, spec, build="fastmath")

        assert warnings == ()
        assert topology.route == "PQ"
        assert topology.coordinate == "psin"
        assert topology.source_active_family == expected_active_family[input_kind]
        assert topology.F_count == 0
        if input_kind == "uniform":
            assert topology.psin_count > 0
        else:
            assert topology.psin_count == 0
        topology.validate_supported_for_veqlib_mvp()
