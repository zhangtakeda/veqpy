#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "tests" / "benchmark.py"
DEFAULT_CXX_EXE = REPO_ROOT / "veqlib" / "build" / "debug" / "veqlib_pf_psin_uniform_benchmark"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "benchmark"
FIXED_CONSTRAINT = "Ip"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veqpy.operator import Operator  # noqa: E402
from veqpy.solver import Solver  # noqa: E402
from veqpy.solver.residual_scale import _build_block_rms_scale  # noqa: E402
from veqpy.solver.solver import _build_x_block_scale_vector  # noqa: E402


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("veqpy_route_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite_or_none(value: float) -> float | None:
    value_eval = float(value)
    return value_eval if np.isfinite(value_eval) else None


def _stats(samples: list[float]) -> dict[str, float | int | list[float]]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {
            "repeat_count": 0,
            "samples_ms": [],
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }
    return {
        "repeat_count": int(values.size),
        "samples_ms": values.tolist(),
        "avg_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": float(np.std(values)),
    }


def _max_abs(lhs: Any, rhs: Any) -> float:
    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if lhs_array.shape != rhs_array.shape:
        return float("inf")
    if lhs_array.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_array - rhs_array)))


def _relative_to_scale(value: float, *arrays: Any) -> float:
    scale = 1.0e-300
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        if values.size:
            scale = max(scale, float(np.max(np.abs(values))))
    return float(value / max(scale, 1.0e-12))


def _benchmark_spec(benchmark: ModuleType):
    return benchmark.BenchmarkCaseSpec(
        mode="PF",
        coordinate="psin",
        input_kind="uniform",
        constraint=FIXED_CONSTRAINT,
    )


def _x0_for_case(benchmark: ModuleType, operator: Operator, spec) -> np.ndarray:
    coefficients = benchmark._coefficients_from_coeffs(benchmark._case_profile_coeffs(spec))
    return operator.pack_coefficients(coefficients)


def _boundary_payload(problem) -> dict[str, float]:
    c_offsets = np.asarray(problem.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(problem.s_offsets, dtype=np.float64)
    return {
        "a": float(problem.a),
        "R0": float(problem.R0),
        "Z0": float(problem.Z0),
        "B0": float(problem.B0),
        "ka": float(problem.ka),
        "c0_offset": float(c_offsets[0]) if c_offsets.size else 0.0,
        "s1_offset": float(s_offsets[1]) if s_offsets.size > 1 else 0.0,
    }


def _python_case_inputs(
    benchmark: ModuleType,
    problem,
    spec,
) -> dict[str, Any]:
    operator = Operator(benchmark.TEST_GRID, problem)
    x0 = _x0_for_case(benchmark, operator, spec)
    initial_raw = operator.residual_var(x0)
    x_scale = _build_x_block_scale_vector(operator, x0)
    residual_scale = _build_block_rms_scale(initial_raw, operator.residual_block_lengths())
    if x_scale is None or residual_scale is None:
        raise RuntimeError(f"{spec.case_name} did not produce solver scaling vectors")

    source_plan = operator.plan.source_plan
    return {
        "case_name": spec.case_name,
        "boundary": _boundary_payload(problem),
        "scaled_heat": source_plan.scaled_heat.tolist(),
        "scaled_current": source_plan.scaled_current.tolist(),
        "scaled_Ip": _finite_or_none(source_plan.scaled_Ip),
        "beta": _finite_or_none(source_plan.beta),
        "fix_rho": float(operator.fix_rho),
        "x0": x0.tolist(),
        "x_scale": x_scale.tolist(),
        "residual_scale": residual_scale.tolist(),
        "initial_raw": initial_raw.tolist(),
        "initial_raw_norm": float(np.linalg.norm(initial_raw)),
        "residual_block_lengths": operator.residual_block_lengths().tolist(),
    }


def _solve_python_timed(
    benchmark: ModuleType,
    problem,
    x0: np.ndarray,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    operator = Operator(benchmark.TEST_GRID, problem)
    solver = Solver(operator=operator, config=benchmark.CONFIG)

    for _ in range(warmup):
        solver.solve(
            x0=x0,
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )

    samples_ms: list[float] = []
    for _ in range(repeat):
        solver.solve(
            x0=x0,
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        if solver.result is None:
            raise RuntimeError("Python solve produced no SolverResult")
        samples_ms.append(float(solver.result.elapsed) / 1000.0)

    if solver.result is None:
        solver.solve(
            x0=x0,
            method=benchmark.CONFIG.method,
            max_residual=benchmark.CONFIG.max_residual,
            max_evaluations=benchmark.CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
    result = solver.result
    if result is None:
        raise RuntimeError("Python solve produced no SolverResult")
    final_raw = operator.residual_var(result.x)

    return {
        "solver": {
            "method": benchmark.CONFIG.method,
            "entrypoint": "scipy.optimize.root(method='hybr')",
            "max_residual": float(benchmark.CONFIG.max_residual),
            "max_evaluations": int(benchmark.CONFIG.max_evaluations),
            "residual_normalization": benchmark.CONFIG.residual_normalization,
        },
        "timing": _stats(samples_ms),
        "final": {
            "success": bool(result.success),
            "message": result.message,
            "x": result.x.tolist(),
            "raw_residual": final_raw.tolist(),
            "raw_norm": float(np.linalg.norm(final_raw)),
            "alpha": [float(operator.alpha1), float(operator.alpha2)],
            "nfev": int(result.function_evaluations),
            "njev": int(result.jacobian_evaluations),
            "iterations": int(result.iterations),
        },
    }


def _run_cxx_case(executable: Path, *, repeat: int, warmup: int) -> dict[str, Any]:
    if not executable.exists():
        raise FileNotFoundError(
            f"C++ benchmark executable not found: {executable}. "
            "Build it with `cmake --build --preset clang-debug --target "
            "veqlib_pf_psin_uniform_benchmark`."
        )
    try:
        completed = subprocess.run(
            [str(executable), "--repeat", str(repeat), "--warmup", str(warmup)],
            check=True,
            cwd=executable.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"C++ benchmark failed:\n{exc.stderr}") from exc
    return json.loads(completed.stdout)


def _run_case(
    benchmark: ModuleType,
    executable: Path,
    reference,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    spec = _benchmark_spec(benchmark)
    problem = benchmark._make_benchmark_case(spec, reference)
    python_inputs = _python_case_inputs(benchmark, problem, spec)
    x0 = np.asarray(python_inputs["x0"], dtype=np.float64)
    python = _solve_python_timed(benchmark, problem, x0, repeat=repeat, warmup=warmup)
    cxx = _run_cxx_case(executable, repeat=repeat, warmup=warmup)

    x_diff = _max_abs(cxx["final"]["x"], python["final"]["x"])
    raw_diff = _max_abs(cxx["final"]["raw_residual"], python["final"]["raw_residual"])
    alpha_diff = _max_abs(cxx["final"]["alpha"], python["final"]["alpha"])
    initial_x_diff = _max_abs(cxx["initial"]["x"], python_inputs["x0"])
    initial_raw_diff = _max_abs(cxx["initial"]["raw_residual"], python_inputs["initial_raw"])
    source_heat_diff = _max_abs(cxx["source"]["scaled_heat"], python_inputs["scaled_heat"])
    source_current_diff = _max_abs(cxx["source"]["scaled_current"], python_inputs["scaled_current"])
    x_scale_diff = _max_abs(cxx["normalization"]["x_scale"], python_inputs["x_scale"])
    residual_scale_diff = _max_abs(
        cxx["normalization"]["residual_scale"],
        python_inputs["residual_scale"],
    )
    cxx_avg = float(cxx["timing"]["avg_ms"])
    python_avg = float(python["timing"]["avg_ms"])

    return {
        "case_name": spec.case_name,
        "constraint": FIXED_CONSTRAINT,
        "active_profiles": dict(problem.active_profiles),
        "x_size": int(x0.size),
        "initial_raw_norm": float(python_inputs["initial_raw_norm"]),
        "initial_residual_scale_max": float(np.max(python_inputs["residual_scale"])),
        "initial_x_max_abs_diff": initial_x_diff,
        "initial_raw_max_abs_diff": initial_raw_diff,
        "source_heat_max_abs_diff": source_heat_diff,
        "source_current_max_abs_diff": source_current_diff,
        "x_scale_max_abs_diff": x_scale_diff,
        "residual_scale_max_abs_diff": residual_scale_diff,
        "scaled_Ip_abs_diff": abs(
            float(cxx["constraints"]["scaled_Ip"]) - float(python_inputs["scaled_Ip"])
        ),
        "final_x_max_abs_diff": x_diff,
        "final_x_rel_diff": _relative_to_scale(x_diff, cxx["final"]["x"], python["final"]["x"]),
        "final_raw_max_abs_diff": raw_diff,
        "final_raw_rel_diff": _relative_to_scale(
            raw_diff,
            cxx["final"]["raw_residual"],
            python["final"]["raw_residual"],
        ),
        "final_alpha_max_abs_diff": alpha_diff,
        "cxx": {
            "success": bool(cxx["success"]),
            "accepted": bool(cxx["final"]["accepted_by_veqpy"]),
            "info": int(cxx["final"]["info"]),
            "nfev": int(cxx["final"]["nfev"]),
            "raw_norm": float(cxx["final"]["raw_norm"]),
            "avg_ms": cxx_avg,
            "median_ms": float(cxx["timing"]["median_ms"]),
            "p95_ms": float(cxx["timing"]["p95_ms"]),
            "std_ms": float(cxx["timing"]["std_ms"]),
        },
        "python": {
            "success": bool(python["final"]["success"]),
            "nfev": int(python["final"]["nfev"]),
            "raw_norm": float(python["final"]["raw_norm"]),
            "avg_ms": python_avg,
            "median_ms": float(python["timing"]["median_ms"]),
            "p95_ms": float(python["timing"]["p95_ms"]),
            "std_ms": float(python["timing"]["std_ms"]),
            "message": python["final"]["message"],
        },
        "speedup_python_over_cxx": python_avg / cxx_avg if cxx_avg > 0.0 else float("inf"),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "max_initial_x_abs_diff": max(float(row["initial_x_max_abs_diff"]) for row in rows),
        "max_source_heat_abs_diff": max(float(row["source_heat_max_abs_diff"]) for row in rows),
        "max_source_current_abs_diff": max(
            float(row["source_current_max_abs_diff"]) for row in rows
        ),
        "max_x_scale_abs_diff": max(float(row["x_scale_max_abs_diff"]) for row in rows),
        "max_residual_scale_abs_diff": max(
            float(row["residual_scale_max_abs_diff"]) for row in rows
        ),
        "max_scaled_Ip_abs_diff": max(float(row["scaled_Ip_abs_diff"]) for row in rows),
        "max_final_x_abs_diff": max(float(row["final_x_max_abs_diff"]) for row in rows),
        "max_final_raw_abs_diff": max(float(row["final_raw_max_abs_diff"]) for row in rows),
        "max_initial_raw_abs_diff": max(float(row["initial_raw_max_abs_diff"]) for row in rows),
        "max_alpha_abs_diff": max(float(row["final_alpha_max_abs_diff"]) for row in rows),
        "geomean_speedup_python_over_cxx": float(
            np.exp(np.mean(np.log([row["speedup_python_over_cxx"] for row in rows])))
        ),
    }


def _write_text_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    summary = payload["summary"]
    lines = [
        "PF/psin/uniform/Ip inline C++ cminpack vs VEQPy benchmark-style comparison",
        "",
        f"repeat_count                  : {payload['repeat']}",
        f"warmup_count                  : {payload['warmup']}",
        f"case_count                    : {summary['case_count']}",
        f"max_initial_x_abs_diff        : {summary['max_initial_x_abs_diff']:.6e}",
        f"max_source_heat_abs_diff      : {summary['max_source_heat_abs_diff']:.6e}",
        f"max_source_current_abs_diff   : {summary['max_source_current_abs_diff']:.6e}",
        f"max_x_scale_abs_diff          : {summary['max_x_scale_abs_diff']:.6e}",
        f"max_residual_scale_abs_diff   : {summary['max_residual_scale_abs_diff']:.6e}",
        f"max_scaled_Ip_abs_diff        : {summary['max_scaled_Ip_abs_diff']:.6e}",
        f"max_initial_raw_abs_diff      : {summary['max_initial_raw_abs_diff']:.6e}",
        f"max_final_x_abs_diff          : {summary['max_final_x_abs_diff']:.6e}",
        f"max_final_raw_abs_diff        : {summary['max_final_raw_abs_diff']:.6e}",
        f"max_alpha_abs_diff            : {summary['max_alpha_abs_diff']:.6e}",
        (
            "geomean_speedup_py_over_cxx  : "
            f"{summary['geomean_speedup_python_over_cxx']:.3f}x"
        ),
        "",
        "Case results",
        "",
        (
            "case".ljust(24)
            + " | "
            + "x_diff".rjust(10)
            + " | "
            + "raw_diff".rjust(10)
            + " | "
            + "cxx_ms".rjust(9)
            + " | "
            + "py_ms".rjust(9)
            + " | "
            + "py/cxx".rjust(8)
            + " | "
            + "nfev".rjust(9)
            + " | "
            + "residuals".rjust(23)
            + " | "
            + "ok".rjust(7)
        ),
    ]
    lines.append("-" * 135)
    for row in rows:
        ok = f"{'Y' if row['cxx']['success'] else 'N'}/{'Y' if row['python']['success'] else 'N'}"
        lines.append(
            f"{row['case_name']:<24} | "
            f"{row['final_x_max_abs_diff']:>10.3e} | "
            f"{row['final_raw_max_abs_diff']:>10.3e} | "
            f"{row['cxx']['avg_ms']:>9.3f} | "
            f"{row['python']['avg_ms']:>9.3f} | "
            f"{row['speedup_python_over_cxx']:>8.2f} | "
            f"{row['cxx']['nfev']:>4d}/{row['python']['nfev']:<4d} | "
            f"{row['cxx']['raw_norm']:>10.3e}/{row['python']['raw_norm']:<10.3e} | "
            f"{ok:>7}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare benchmark.py-style PF/psin/uniform cases in VEQlib and VEQPy."
    )
    parser.add_argument("--cxx-exe", type=Path, default=DEFAULT_CXX_EXE)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    benchmark = _load_benchmark_module()
    reference = benchmark._solve_reference(show_progress=not args.quiet)
    if not args.quiet:
        print("running PF_psin_uniform_Ip ...", flush=True)
    rows = [
        _run_case(
            benchmark,
            args.cxx_exe.resolve(),
            reference,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    ]
    payload = {
        "schema_version": 1,
        "source": "tests/benchmark.py fixed PF/psin/uniform/Ip case",
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "rows": rows,
        "summary": _summary(rows),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "cpp_python_pf_psin_uniform_compare.json"
        txt_path = args.output_dir / "cpp_python_pf_psin_uniform_compare.txt"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_text_report(txt_path, payload)
        if not args.quiet:
            print(f"wrote {json_path}")
            print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
