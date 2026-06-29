from __future__ import annotations

import json

from veqlib.benchmarks import benchmark_geqdsks, benchmark_routes


def test_retained_benchmark_defaults_write_under_outputs() -> None:
    assert benchmark_routes.DEFAULT_OUTPUT.parts[-2:] == ("outputs", "veqlib_routes.json")
    assert benchmark_geqdsks.DEFAULT_OUTPUT.parts[-2:] == (
        "outputs",
        "veqlib_geqdsk_configs.json",
    )


def test_routes_default_scope_is_ip_uniform_matrix() -> None:
    specs = benchmark_routes._iter_route_specs(
        benchmark_routes._benchmark_module(),
        scope=benchmark_routes.DEFAULT_SCOPE,
    )

    assert len(specs) == 12
    assert {spec.mode for spec in specs} == {"PF", "PP", "PI", "PJ1", "PJ2", "PQ"}
    assert {spec.coordinate for spec in specs} == {"rho", "psin"}
    assert {spec.input_kind for spec in specs} == {"uniform"}
    assert {spec.constraint for spec in specs} == {"Ip"}


def test_routes_no_run_smoke(tmp_path) -> None:
    output = tmp_path / "routes.json"

    status = benchmark_routes.main(
        [
            "--scope",
            "uniform",
            "--case",
            "PF:psin:uniform:Ip",
            "--no-run",
            "--skip-artifact-dry-run",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert status == 0
    assert payload["schema"] == "veqlib.routes.v2"
    assert payload["case_count"] == 1
    assert payload["rows"][0]["runtime"]["status"] == "not_requested"


def test_geqdsk_config_case_plan_smoke() -> None:
    cases = benchmark_geqdsks._make_cases(
        build="fastmath",
        selected_cases={"solovev"},
        selected_configs={"low"},
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.row_label == "solovev:low"
    assert case.topology.route == "PF"
    assert case.topology.coordinate == "psin"
    assert case.topology.nodes == "uniform"
    assert case.topology.build == "fastmath"
