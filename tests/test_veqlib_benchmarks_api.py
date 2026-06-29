from __future__ import annotations

import json

from veqlib.benchmarks import benchmark_geqdsk_configs, benchmark_routes


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
    cases = benchmark_geqdsk_configs._make_cases(
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
