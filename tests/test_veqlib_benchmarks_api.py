from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from benchmarks import _common as benchmark_common
from benchmarks import (
    veqlib_continuation as benchmark_continuation,
)
from benchmarks import (
    veqlib_geqdsk_pareto as benchmark_geqdsk,
)
from benchmarks import (
    veqlib_routes as benchmark_routes,
)
from benchmarks import (
    veqpy_geqdsk_routes as benchmark_veqpy_geqdsk_routes,
)
from benchmarks import (
    veqpy_routes as benchmark_veqpy_routes,
)
from veqpy.solver import SolverConfig


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
    assert benchmark_veqpy_routes.DEFAULT_OUTPUT.parts[-3:] == (
        "benchmarks",
        "results",
        "veqpy_routes.json",
    )
    assert benchmark_veqpy_geqdsk_routes.DEFAULT_OUTPUT.parts[-3:] == (
        "benchmarks",
        "results",
        "veqpy_geqdsk_routes.json",
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


def test_veqpy_route_benchmarks_default_scope_is_ip_uniform_matrix() -> None:
    route_specs = benchmark_veqpy_routes._iter_route_specs(
        scope=benchmark_veqpy_routes.DEFAULT_SCOPE,
    )
    geqdsk_specs = benchmark_veqpy_geqdsk_routes._iter_route_specs(
        scope=benchmark_veqpy_geqdsk_routes.DEFAULT_SCOPE,
    )

    for specs in (route_specs, geqdsk_specs):
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
    assert payload["solver_policy"] == {
        "initial": "cold",
        "continue": "cold",
        "norm": "fast",
    }
    assert payload["rows"][0]["runtime"]["status"] == "not_requested"


def test_veqpy_routes_no_run_smoke(tmp_path) -> None:
    output = tmp_path / "veqpy_routes.json"

    status = benchmark_veqpy_routes.main(
        [
            "--scope",
            "uniform",
            "--case",
            "PF:psin:uniform:Ip",
            "--no-run",
            "--quiet-progress",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert status == 0
    assert payload["schema"] == "veqpy.routes.v1"
    assert payload["case_count"] == 1
    assert payload["engine"] == "veqpy-numba-hybr"
    assert payload["rows"][0]["runtime"]["status"] == "not_requested"


def test_veqpy_geqdsk_routes_no_run_smoke(tmp_path) -> None:
    output = tmp_path / "veqpy_geqdsk_routes.json"

    status = benchmark_veqpy_geqdsk_routes.main(
        [
            "--scope",
            "uniform",
            "--case",
            "PF:psin:uniform:Ip",
            "--no-run",
            "--quiet-progress",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert status == 0
    assert payload["schema"] == "veqpy.geqdsk_routes.v1"
    assert payload["case_count"] == 1
    assert payload["geqdsk"].endswith("data/SOLOVEV.geqdsk")
    assert payload["engine"] == "veqpy-numba-lm"
    assert payload["run_mode"] == "plan-only"
    assert payload["skip_reason"] == "no_run"
    assert payload["layout"]["solve_grid"]["Nr"] == 32
    assert payload["layout"]["solve_grid"]["Nt"] == 32
    assert payload["rows"][0]["runtime"]["status"] == "not_requested"


def test_veqpy_geqdsk_routes_defaults_to_solovev(tmp_path, monkeypatch) -> None:
    output = tmp_path / "veqpy_geqdsk_routes_default.json"
    captured: dict[str, str] = {}

    def fake_reference(path: str, **_: object) -> object:
        captured["path"] = path
        return object()

    def fake_measure(_: object, __: object, *, args: object) -> dict[str, object]:
        assert int(args.repeat) == 1
        return {"status": "passed", "x_size": 1}

    monkeypatch.setattr(benchmark_veqpy_geqdsk_routes, "_geqdsk_reference", fake_reference)
    monkeypatch.setattr(benchmark_veqpy_geqdsk_routes, "_measure_spec", fake_measure)

    status = benchmark_veqpy_geqdsk_routes.main(
        [
            "--scope",
            "uniform",
            "--case",
            "PF:psin:uniform:Ip",
            "--quiet-progress",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert status == 0
    assert captured["path"].endswith("data/SOLOVEV.geqdsk")
    assert payload["geqdsk"].endswith("data/SOLOVEV.geqdsk")
    assert payload["run_mode"] == "run"
    assert payload["skip_reason"] is None
    assert payload["rows"][0]["runtime"]["status"] == "passed"


def test_route_reference_solve_respects_initial_policy(monkeypatch) -> None:
    calls: dict[str, object] = {"packed": False}

    class FakeProblem:
        active_profiles: dict[str, int] = {"h": 1}
        boundary = object()

        def copy(self) -> "FakeProblem":
            return self

    class FakeOperator:
        def __init__(self, grid: object, problem: FakeProblem) -> None:
            self.grid = grid
            self.problem = problem

        def pack_coefficients(self, coeffs: object) -> np.ndarray:
            calls["packed"] = True
            return np.zeros(0, dtype=np.float64)

        def build_coeffs(self, x: object, *, include_none: bool = False) -> dict[str, np.ndarray]:
            del x, include_none
            return {}

    class FakeSolver:
        def __init__(self, *, operator: FakeOperator, config: SolverConfig) -> None:
            self.operator = operator
            self.config = config
            self.result = SimpleNamespace(x=np.zeros(1, dtype=np.float64))

        def solve(self, **kwargs: object) -> None:
            calls["kwargs"] = kwargs

        def build_equilibrium(self) -> SimpleNamespace:
            axis = np.array([0.0, 1.0], dtype=np.float64)
            zeros = np.zeros(2, dtype=np.float64)
            ones = np.ones(2, dtype=np.float64)
            return SimpleNamespace(
                rho=axis,
                psin=axis,
                psin_r=ones,
                alpha2=1.0,
                FFn_r=zeros,
                Pn_r=zeros,
                FF_r=zeros,
                P_r=zeros,
                Itor=zeros,
                jtor=zeros,
                jpara=zeros,
                q=ones,
                Ip=1.0e6,
                beta_t=0.1,
            )

    monkeypatch.setattr(benchmark_common, "Operator", FakeOperator)
    monkeypatch.setattr(benchmark_common, "Solver", FakeSolver)

    benchmark_common.solve_route_reference(
        FakeProblem(),
        grid=object(),
        config=SolverConfig(initial_policy="geometric-refined"),
        coeffs={"h": [0.0]},
    )

    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["initial_policy"] == "geometric-refined"
    assert "x0" not in kwargs
    assert calls["packed"] is False


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
    assert case.kernel_solve.method == "powell"
    assert case.kernel_solve.initial == "cold"
    assert case.kernel_solve.continuation == "cold"
    assert case.kernel_solve.norm == "fast"


def test_continuation_offsets_center_on_zero() -> None:
    assert benchmark_continuation._scan_offsets(points=1, relative_span=0.2) == [0.0]
    assert benchmark_continuation._scan_offsets(points=3, relative_span=0.2) == [
        -0.1,
        0.0,
        0.1,
    ]


def test_continuation_warm_policies_start_from_cold_initial() -> None:
    assert benchmark_continuation._initial_policy_for_policy("cold-geometric") == "cold-geometric"
    assert benchmark_continuation._initial_policy_for_policy("warm-predict") == "cold"


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
