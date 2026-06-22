#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veqpy.model import Boundary, Grid, Problem  # noqa: E402
from veqpy.operator import Operator  # noqa: E402
from veqpy.solver import Solver, SolverConfig  # noqa: E402
from veqpy.solver.residual_scale import _build_block_rms_scale  # noqa: E402
from veqpy.solver.solver import _build_x_block_scale_vector, _root_options_for  # noqa: E402

MU0 = 4.0e-7 * np.pi
CPP_HEAT = np.array([2.0, 2.75, 3.5, 4.25, 5.0], dtype=np.float64)
CPP_CURRENT = np.array([0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float64)


def _repo_relative_default_exe() -> Path:
    return REPO_ROOT / "veqlib" / "build" / "debug" / "veqlib_main"


def _run_cxx_report(executable: Path) -> dict[str, Any]:
    if not executable.exists():
        raise FileNotFoundError(
            f"C++ validation executable not found: {executable}. "
            "Build it with `cmake --build --preset clang-debug --target "
            "veqlib_main`."
        )
    completed = subprocess.run(
        [str(executable), "--mode", "pf-validation"],
        check=True,
        cwd=executable.parent,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def _fixed_case_operator() -> Operator:
    problem = Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles={"psin": 1},
        boundary=Boundary(
            a=0.42,
            R0=1.8,
            Z0=-0.25,
            B0=2.1,
            ka=1.45,
            c_offsets=np.array([0.0], dtype=np.float64),
        ),
        # VEQPy accepts physical heat input and SourcePlan applies mu0.  The C++
        # harness consumes engine-ready scaled heat, so divide here to fix the
        # same source samples at the engine boundary.
        heat_input=CPP_HEAT / MU0,
        current_input=CPP_CURRENT,
        Ip=3.0e6,
    )
    grid = Grid(
        Nr=8,
        Nt=8,
        L_max=1,
        M_max=1,
        K_max=2,
        quadrature_scheme="legendre",
        calculus_scheme="spectral",
    )
    return Operator(grid, problem, fix_rho=0.0)


def _python_report() -> dict[str, Any]:
    operator = _fixed_case_operator()
    x_initial = np.zeros(operator.x_size, dtype=np.float64)
    initial_raw = operator.residual_var(x_initial)

    config = SolverConfig(
        method="hybr",
        max_residual=1.0e-6,
        max_evaluations=1000,
        initial_policy="auto",
        enable_fallback=False,
        enable_history=False,
    )
    x_scale = _build_x_block_scale_vector(operator, x_initial)
    residual_scale = _build_block_rms_scale(initial_raw, operator.residual_block_lengths())
    options = _root_options_for(config)
    effective_options = {**options, "factor": 1.0}

    initial = {
        "raw_residual": initial_raw.tolist(),
        "alpha": [operator.alpha1, operator.alpha2],
        "profile_psin": operator.profile_workspace.fields_for("psin")[0].tolist(),
        "profile_psin_r": operator.profile_workspace.fields_for("psin")[1].tolist(),
        "profile_psin_rr": operator.profile_workspace.fields_for("psin")[2].tolist(),
        "root_psin": operator.residual_workspace.root_fields[0].tolist(),
        "root_psin_r": operator.residual_workspace.root_fields[1].tolist(),
        "root_psin_rr": operator.residual_workspace.root_fields[2].tolist(),
        "FFn_psin": operator.residual_workspace.root_fields[3].tolist(),
        "Pn_psin": operator.residual_workspace.root_fields[4].tolist(),
        "source_target_psin": operator.source_workspace.target_root_fields[0].tolist(),
        "source_target_psin_r": operator.source_workspace.target_root_fields[1].tolist(),
        "source_target_psin_rr": operator.source_workspace.target_root_fields[2].tolist(),
        "source_psin_query": operator.source_workspace.psin_query.tolist(),
        "source_parameter_query": operator.source_workspace.parameter_query.tolist(),
        "materialized_heat_input": operator.source_workspace.materialized_heat_input.tolist(),
        "materialized_current_input": operator.source_workspace.materialized_current_input.tolist(),
        "geometry_S_r": operator.geometry_workspace.radial_fields[0].tolist(),
        "geometry_V_r": operator.geometry_workspace.radial_fields[1].tolist(),
        "geometry_Kn": operator.geometry_workspace.radial_fields[2].tolist(),
        "geometry_Kn_r": operator.geometry_workspace.radial_fields[3].tolist(),
        "geometry_Ln_r": operator.geometry_workspace.radial_fields[4].tolist(),
        "residual_surface_G": operator.residual_workspace.surface_fields[0].tolist(),
        "residual_surface_Gpsin_R": operator.residual_workspace.surface_fields[1].tolist(),
        "residual_surface_Gpsin_Z": operator.residual_workspace.surface_fields[2].tolist(),
        "residual_surface_Gpsin_R_sin_tb": operator.residual_workspace.surface_fields[3].tolist(),
    }

    solver = Solver(operator=operator, config=config)
    x_final = solver.solve()
    result = solver.result
    if result is None:
        raise RuntimeError("Solver did not populate a SolverResult")
    final_raw = operator.residual_var(x_final)

    return {
        "solver": {
            "method": config.method,
            "entrypoint": "scipy.optimize.root(method='hybr')",
            "tol": config.max_residual,
            "options": options,
            "effective_options": effective_options,
            "residual_normalization": config.residual_normalization,
            "x_scale": [] if x_scale is None else x_scale.tolist(),
            "residual_scale": [] if residual_scale is None else residual_scale.tolist(),
            "unknown_space": "z = x / x_scale",
        },
        "initial": initial,
        "final": {
            "x": x_final.tolist(),
            "raw_residual": final_raw.tolist(),
            "alpha": [operator.alpha1, operator.alpha2],
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.function_evaluations),
        },
    }


def _get(report: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = report
    for part in path:
        value = value[part]
    return value


def _max_abs(lhs: Any, rhs: Any) -> float:
    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if lhs_array.shape != rhs_array.shape:
        return float("inf")
    if lhs_array.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_array - rhs_array)))


def _comparison_rows(cxx: dict[str, Any], python: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        ("initial.raw_residual", ("initial", "raw_residual"), ("initial", "raw_residual")),
        ("initial.alpha", ("initial", "state", "alpha"), ("initial", "alpha")),
        (
            "initial.profile_psin",
            ("initial", "state", "profiles", "psin"),
            ("initial", "profile_psin"),
        ),
        (
            "initial.profile_psin_r",
            ("initial", "state", "profiles", "psin_r"),
            ("initial", "profile_psin_r"),
        ),
        (
            "initial.root_psin_r",
            ("initial", "state", "source", "profile_root_psin_r"),
            ("initial", "root_psin_r"),
        ),
        (
            "initial.materialized_heat_input",
            ("initial", "state", "source", "materialized_heat_input"),
            ("initial", "materialized_heat_input"),
        ),
        (
            "initial.materialized_current_input",
            ("initial", "state", "source", "materialized_current_input"),
            ("initial", "materialized_current_input"),
        ),
        (
            "initial.source_target_psin_r",
            ("initial", "state", "source", "source_target_psin_r"),
            ("initial", "source_target_psin_r"),
        ),
        ("initial.FFn_psin", ("initial", "state", "source", "FFn_psin"), ("initial", "FFn_psin")),
        ("initial.Pn_psin", ("initial", "state", "source", "Pn_psin"), ("initial", "Pn_psin")),
        (
            "initial.geometry_V_r",
            ("initial", "state", "geometry", "V_r"),
            ("initial", "geometry_V_r"),
        ),
        ("initial.geometry_Kn", ("initial", "state", "geometry", "Kn"), ("initial", "geometry_Kn")),
        (
            "initial.geometry_Ln_r",
            ("initial", "state", "geometry", "Ln_r"),
            ("initial", "geometry_Ln_r"),
        ),
        (
            "initial.residual_surface_G",
            ("initial", "state", "residual_surface", "G"),
            ("initial", "residual_surface_G"),
        ),
        ("final.x", ("final", "x"), ("final", "x")),
        ("final.raw_residual", ("final", "raw_residual"), ("final", "raw_residual")),
        ("final.alpha", ("final", "state", "alpha"), ("final", "alpha")),
    ]

    rows: list[dict[str, Any]] = []
    for name, cxx_path, python_path in fields:
        cxx_value = _get(cxx, cxx_path)
        python_value = _get(python, python_path)
        rows.append(
            {
                "field": name,
                "max_abs": _max_abs(cxx_value, python_value),
                "cxx_shape": list(np.asarray(cxx_value).shape),
                "python_shape": list(np.asarray(python_value).shape),
            }
        )
    return rows


def _solver_interface(cxx: dict[str, Any], python: dict[str, Any]) -> dict[str, Any]:
    cxx_solver = cxx["solver"]
    py_solver = python["solver"]
    return {
        "cxx_entrypoint": cxx_solver["entrypoint"],
        "python_entrypoint": py_solver["entrypoint"],
        "tol": {"cxx": cxx_solver["max_residual"], "python": py_solver["tol"]},
        "maxfev": {"cxx": cxx_solver["maxfev"], "python": py_solver["effective_options"]["maxfev"]},
        "eps": {"cxx": cxx_solver["eps"], "python": py_solver["effective_options"]["eps"]},
        "factor": {"cxx": cxx_solver["factor"], "python": py_solver["effective_options"]["factor"]},
        "x_scale": {"cxx": cxx["normalization"]["x_scale"], "python": py_solver["x_scale"]},
        "residual_scale": {
            "cxx": cxx["normalization"]["residual_scale"],
            "python": py_solver["residual_scale"],
        },
        "cxx_nfev": cxx["cminpack"]["nfev"],
        "python_nfev": python["final"]["nfev"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the fixed PF/psin/uniform/Ip VEQlib cminpack solve against VEQPy."
    )
    parser.add_argument("--cxx-exe", type=Path, default=_repo_relative_default_exe())
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args(argv)

    cxx = _run_cxx_report(args.cxx_exe.resolve())
    python = _python_report()
    rows = _comparison_rows(cxx, python)
    solver_interface = _solver_interface(cxx, python)
    max_abs = max(row["max_abs"] for row in rows)
    passed = bool(max_abs <= args.tolerance)

    report = {
        "case": "PF/psin/uniform/Ip active psin length=1, Grid(8,8,L=1,M=1,K=2)",
        "passed": passed,
        "tolerance": args.tolerance,
        "max_abs": max_abs,
        "solver_interface": solver_interface,
        "comparisons": rows,
        "final": {
            "cxx_x": cxx["final"]["x"],
            "python_x": python["final"]["x"],
            "cxx_raw_residual": cxx["final"]["raw_residual"],
            "python_raw_residual": python["final"]["raw_residual"],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
