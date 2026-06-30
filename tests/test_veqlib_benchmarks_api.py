from __future__ import annotations

import json

from benchmarks import (
    veqlib_continuation as benchmark_continuation,
)
from benchmarks import (
    veqlib_geqdsk as benchmark_geqdsk,
)
from benchmarks import (
    veqlib_routes as benchmark_routes,
)


def test_retained_benchmark_defaults_write_under_benchmarks_results() -> None:
    assert benchmark_routes.DEFAULT_OUTPUT.parts[-3:] == (
        "benchmarks",
        "results",
        "veqlib_routes.json",
    )
    assert benchmark_geqdsk.DEFAULT_OUTPUT.parts[-3:] == (
        "benchmarks",
        "results",
        "veqlib_geqdsk.json",
    )
    assert benchmark_continuation.DEFAULT_OUTPUT_DIR.parts[-3:] == (
        "benchmarks",
        "results",
        "veqlib_continuation",
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
    cases = benchmark_geqdsk._make_cases(
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


def test_continuation_offsets_center_on_zero() -> None:
    assert benchmark_continuation._scan_offsets(points=1, relative_span=0.2) == [0.0]
    assert benchmark_continuation._scan_offsets(points=3, relative_span=0.2) == [
        -0.1,
        0.0,
        0.1,
    ]


def test_continuation_comparison_rows_use_effective_nfev() -> None:
    def policy(mean: float) -> dict[str, object]:
        return {"effective_nfev": {"mean": mean}, "success_all": True}

    payload = {
        "policies": ["cold", "warm-predict", "warm"],
        "rows": [
            {
                "status": "passed",
                "experiment": "C1 Ip",
                "update": "ip",
                "relative_span": 0.005,
                "case": "solovev",
                "config": "Ref",
                "x_size": 10,
                "policies": {
                    "cold": policy(20.0),
                    "warm-predict": policy(8.0),
                    "warm": policy(8.0),
                },
            }
        ],
    }

    rows = benchmark_continuation._comparison_rows(payload)

    assert rows == [
        {
            "experiment": "C1 Ip",
            "status": "passed",
            "update": "ip",
            "span": 0.005,
            "case": "solovev",
            "config": "Ref",
            "x_size": 10,
            "best": "warm-predict",
            "best_nfev": 8.0,
            "vs_cold": 2.5,
            "vs_warm": 1.0,
            "success_all": True,
            "cold": 20.0,
            "warm-predict": 8.0,
            "warm": 8.0,
        }
    ]
