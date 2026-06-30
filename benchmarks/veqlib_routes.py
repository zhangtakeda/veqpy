#!/usr/bin/env python3
"""Route/topology benchmark for VEQlib against VEQPy.

The default scope enumerates the 12 ``*:rho/psin:uniform:Ip`` route cases, which
keeps one default run below the nanobind per-process cleanup-handler ceiling.
``--scope uniform`` restores the 46 historical uniform source cases from
``tests/benchmark.py``. ``--scope full`` adds grid-sampled variants for the 92
route/topology matrix. Topology planning is always performed; supported native
rows can be executed through the typed ``veqlib.facade`` runtime unless
``--no-run`` is passed. Native rows are isolated in one subprocess per row by
default so full matrices do not load dozens of nanobind domains into one
interpreter; ``--run-native-in-process`` keeps the single-process path for
debugging.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from benchmarks._common import (
    CORE_DIR,
    REPO_ROOT,
    cpu_affinity,
    family_counts,
    float_stats,
    int_stats,
    load_module,
    max_abs,
    measure_native_solver,
    profile_count,
    runtime_env,
    write_json,
)
from veqlib.facade import (
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_POWELL,
    KernelBoundary,
    KernelBuild,
    KernelConfig,
    KernelInput,
    KernelRegistry,
    KernelTopology,
    TopologyError,
    VEQlibSolver,
    build_kernel,
    default_kernel_cache_root,
)
from veqpy.operator import Operator
from veqpy.operator.packed_layout import build_profile_layout, build_profile_names
from veqpy.solver import Solver

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "veqlib_routes.json"
VALIDATION_ATOL = 1.0e-6
DEFAULT_SCOPE = "ip-uniform"
NATIVE_SOLVER_INITIAL_POLICY = "cold"
NATIVE_SOLVER_CONTINUATION_POLICY = "cold"
NATIVE_SOLVER_NORMALIZATION = "fast"
REPORT_TABLE_BOX = box.Box("    \n    \n ── \n    \n ── \n ── \n    \n ── \n")
Topology = KernelTopology


@dataclass(frozen=True, slots=True)
class RuntimeCase:
    spec: Any
    topology: Topology
    kernel_boundary: KernelBoundary
    kernel_input: KernelInput
    kernel_config: KernelConfig
    py_operator: Any
    py_measure: Any
    solver_method_code: int
    solver_engine_label: str
    x_size: int


_SOLVER_LABELS = {
    SOLVER_METHOD_POWELL: "veqlib-fastmath-powell",
    SOLVER_METHOD_LEVENBERG_MARQUARDT: "veqlib-fastmath-lm",
}


def _console() -> Console:
    return Console(highlight=False)


@lru_cache(maxsize=1)
def _benchmark_module() -> ModuleType:
    return load_module(
        "veqpy_route_benchmark_for_veqlib_routes",
        REPO_ROOT / "tests" / "benchmark.py",
    )


@lru_cache(maxsize=1)
def _benchmark_reference() -> Any:
    return _benchmark_module()._solve_reference(show_progress=False)


def _spec_label(spec: Any) -> str:
    return str(spec.case_name)


def _spec_selector(spec: Any) -> str:
    return f"{spec.mode}:{spec.coordinate}:{spec.input_kind}:{spec.constraint}"


def _cxx_solver_method_for_spec(spec: Any) -> int:
    if (
        str(spec.mode) == "PJ2"
        and str(spec.coordinate) == "psin"
        and str(spec.input_kind) == "grid"
        and str(spec.constraint) == "Ip"
    ):
        return SOLVER_METHOD_LEVENBERG_MARQUARDT
    return SOLVER_METHOD_POWELL


def _iter_route_specs(benchmark: ModuleType, *, scope: str) -> tuple[Any, ...]:
    if scope == DEFAULT_SCOPE:
        input_kinds = ("uniform",)
        constraints_by_mode = {mode: ("Ip",) for mode in benchmark.BENCHMARK_MODES}
    elif scope in {"uniform", "full"}:
        input_kinds = list(benchmark.BENCHMARK_INPUT_KINDS)
        if scope == "full" and "grid" not in input_kinds:
            input_kinds.append("grid")
        constraints_by_mode = benchmark.BENCHMARK_MODE_CONSTRAINTS
    else:
        raise ValueError(f"unknown route benchmark scope {scope!r}")
    return tuple(
        benchmark.BenchmarkCaseSpec(
            mode=mode,
            coordinate=coordinate,
            constraint=constraint,
            input_kind=input_kind,
        )
        for mode in benchmark.BENCHMARK_MODES
        for coordinate in ("rho", "psin")
        for input_kind in input_kinds
        for constraint in constraints_by_mode[mode]
    )


def _filter_specs(specs: tuple[Any, ...], selected: set[str] | None) -> tuple[Any, ...]:
    if selected is None:
        return specs
    selected_lower = {item.lower() for item in selected}
    retained = tuple(
        spec
        for spec in specs
        if _spec_label(spec).lower() in selected_lower
        or _spec_selector(spec).lower() in selected_lower
    )
    matched = {_spec_label(spec).lower() for spec in retained}
    matched.update(_spec_selector(spec).lower() for spec in retained)
    missing = selected_lower.difference(matched)
    if missing:
        raise ValueError(f"unknown case selector(s): {', '.join(sorted(missing))}")
    return retained


def _active_profiles_from_topology(topology: Topology) -> dict[str, int]:
    active: dict[str, int] = {}
    for name, count in (
        ("h", topology.h_count),
        ("v", topology.v_count),
        ("k", topology.kappa_count),
        ("psin", topology.psin_count),
        ("F", topology.F_count),
    ):
        if count > 0:
            active[name] = int(count)
    for order, count in enumerate(topology.c_counts):
        if count > 0:
            active[f"c{order}"] = int(count)
    for order, count in enumerate(topology.s_counts, start=1):
        if count > 0:
            active[f"s{order}"] = int(count)
    return active


def _coeff_index_for_layout(topology: Topology, *, layout: str) -> np.ndarray:
    _, coeff_index, _ = build_profile_layout(
        _active_profiles_from_topology(topology),
        profile_names=build_profile_names(topology.M_max),
        profile_first=layout == "family",
    )
    return coeff_index


def _packed_to_degree_layout(values: Any, topology: Topology) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if topology.layout == "degree":
        return values_arr.copy()
    source_index = _coeff_index_for_layout(topology, layout=topology.layout)
    degree_index = _coeff_index_for_layout(topology, layout="degree")
    out = np.empty_like(values_arr)
    for profile_row in range(source_index.shape[0]):
        for degree in range(source_index.shape[1]):
            source_pos = int(source_index[profile_row, degree])
            if source_pos >= 0:
                out[int(degree_index[profile_row, degree])] = values_arr[source_pos]
    return out


def _boundary_m_max(boundary: Any) -> int:
    c_offsets = np.asarray(boundary.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(boundary.s_offsets, dtype=np.float64)
    return max(
        int(c_offsets.size) - 1 if c_offsets.size else 0,
        int(s_offsets.size) - 1 if s_offsets.size else 0,
        1,
    )


def _sample_count_for_spec(benchmark: ModuleType, spec: Any) -> int:
    if str(spec.input_kind).lower() == "grid":
        return int(benchmark.TEST_GRID.Nr)
    return int(benchmark.TEST_SOURCE_SAMPLE_COUNT)


def _topology_from_spec(
    benchmark: ModuleType,
    spec: Any,
    *,
    build: str,
    layout: str = "degree",
    build_options: dict[str, object] | None = None,
) -> tuple[KernelTopology, tuple[str, ...]]:
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    m_max = _boundary_m_max(benchmark.BOUNDARY)
    topology_kwargs: dict[str, object] = {
        "h_count": profile_count(coeffs, "h"),
        "v_count": profile_count(coeffs, "v"),
        "kappa_count": profile_count(coeffs, "k"),
        "psin_count": profile_count(coeffs, "psin"),
        "F_count": profile_count(coeffs, "F"),
        "c_counts": family_counts(coeffs, "c", 0),
        "s_counts": family_counts(coeffs, "s", 1),
        "Nr": int(grid.Nr),
        "Nt": int(grid.Nt),
        "route": str(spec.mode),
        "coordinate": str(spec.coordinate),
        "constraint": str(spec.constraint),
        "nodes": str(spec.input_kind),
        "sample_count": _sample_count_for_spec(benchmark, spec),
        "M_max": m_max,
        "K_max": max(2, m_max),
    }
    build_kwargs: dict[str, object] = {"build": build, "layout": layout}
    if build_options:
        build_kwargs.update(build_options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        topology = KernelTopology(**topology_kwargs).with_build(KernelBuild(**build_kwargs))
    return topology, tuple(str(item.message) for item in caught)


def _kernel_boundary_from_case(case: Any) -> KernelBoundary:
    boundary = case.boundary
    return KernelBoundary(
        a=float(boundary.a),
        R0=float(boundary.R0),
        Z0=float(boundary.Z0),
        B0=float(boundary.B0),
        ka=float(boundary.ka),
        c_offsets=np.asarray(boundary.c_offsets, dtype=np.float64),
        s_offsets=np.asarray(boundary.s_offsets, dtype=np.float64),
    )


def _kernel_input_from_operator(case: Any, operator: Any, *, case_name: str) -> KernelInput:
    source_plan = operator.plan.source_plan
    return KernelInput(
        scaled_heat=np.asarray(source_plan.scaled_heat, dtype=np.float64),
        scaled_current=np.asarray(source_plan.scaled_current, dtype=np.float64),
        scaled_Ip=float(source_plan.scaled_Ip),
        beta=float(source_plan.beta),
        case_name=case_name,
    )


def _kernel_config_from_config(
    config: Any,
    *,
    method: int,
    x_size: int,
) -> KernelConfig:
    return KernelConfig(
        method=method,
        max_residual=float(config.max_residual),
        max_evaluations=int(x_size) ** 2,
        initial=NATIVE_SOLVER_INITIAL_POLICY,
        continuation=NATIVE_SOLVER_CONTINUATION_POLICY,
        norm=NATIVE_SOLVER_NORMALIZATION,
        residual_normalization_floor=float(config.residual_normalization_floor),
        residual_normalization_max_ratio=float(config.residual_normalization_max_ratio),
        residual_normalization_huber_tau=float(config.residual_normalization_huber_tau),
        residual_normalization_probe_count=int(config.residual_normalization_probe_count),
        residual_normalization_probe_step=float(config.residual_normalization_probe_step),
        residual_normalization_sensitivity_lambda=float(
            config.residual_normalization_sensitivity_lambda
        ),
    )


def _runtime_case(benchmark: ModuleType, spec: Any, topology: Topology) -> RuntimeCase:
    reference = _benchmark_reference()
    case = benchmark._make_benchmark_case(spec, reference)
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    operator = Operator(grid, case)
    x0 = operator.pack_coefficients(benchmark._coefficients_from_coeffs(coeffs))
    method = _cxx_solver_method_for_spec(spec)
    kernel_boundary = _kernel_boundary_from_case(case)
    kernel_input = _kernel_input_from_operator(
        case,
        operator,
        case_name=_spec_label(spec),
    )
    kernel_config = _kernel_config_from_config(
        benchmark.CONFIG,
        method=method,
        x_size=int(x0.size),
    )

    def measure_py(*, warmup: int, repeat: int) -> dict[str, Any]:
        for _ in range(warmup):
            solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
            solver.solve(
                x0=x0,
                method=benchmark.CONFIG.method,
                max_residual=benchmark.CONFIG.max_residual,
                max_evaluations=benchmark.CONFIG.max_evaluations,
                enable_verbose=False,
                enable_history=False,
            )
        wall_ms: list[float] = []
        nfev: list[int] = []
        success: list[bool] = []
        last_solver = None
        for _ in range(repeat):
            solver = Solver(operator=Operator(grid, case.copy()), config=benchmark.CONFIG)
            started = time.perf_counter_ns()
            solver.solve(
                x0=x0,
                method=benchmark.CONFIG.method,
                max_residual=benchmark.CONFIG.max_residual,
                max_evaluations=benchmark.CONFIG.max_evaluations,
                enable_verbose=False,
                enable_history=False,
            )
            wall_ms.append(float(time.perf_counter_ns() - started) / 1.0e6)
            if solver.result is None:
                raise RuntimeError(f"{_spec_label(spec)} VEQPy solve produced no result")
            nfev.append(int(solver.result.function_evaluations))
            success.append(bool(solver.result.success))
            last_solver = solver
        if last_solver is None or last_solver.result is None:
            raise RuntimeError(f"{_spec_label(spec)} VEQPy timing loop did not run")
        raw = np.asarray(last_solver.operator.residual_var(last_solver.result.x), dtype=np.float64)
        return {
            "success_all": all(success),
            "timing": float_stats(wall_ms),
            "nfev": int_stats(nfev),
            "x": np.asarray(last_solver.result.x, dtype=np.float64).copy(),
            "raw": raw,
            "raw_norm": float(np.linalg.norm(raw)),
            "message": str(last_solver.result.message),
        }

    return RuntimeCase(
        spec=spec,
        topology=topology,
        kernel_boundary=kernel_boundary,
        kernel_input=kernel_input,
        kernel_config=kernel_config,
        py_operator=operator,
        py_measure=measure_py,
        solver_method_code=method,
        solver_engine_label=_SOLVER_LABELS[method],
        x_size=int(x0.size),
    )


def _measure_veqlib(
    case: RuntimeCase,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    solver = VEQlibSolver(case.topology, registry=registry, solver=case.solver_method_code)
    build_start = time.perf_counter_ns()
    artifact = solver.build(force=False, dry_run=False)
    build_wall_ms = float(time.perf_counter_ns() - build_start) / 1.0e6

    def configure() -> None:
        solver.set_kernel_runtime(
            "" if case.kernel_input.case_name is None else case.kernel_input.case_name,
            *case.kernel_boundary.runtime_args(),
            *case.kernel_input.runtime_args(),
            *case.kernel_config.runtime_args(x_size=case.x_size),
        )

    timing = measure_native_solver(solver, configure, warmup=warmup, repeat=repeat)
    native_x = _packed_to_degree_layout(timing.result.x, case.topology)
    native_raw = _packed_to_degree_layout(timing.result.raw, case.topology)
    py_raw_at_native = np.asarray(case.py_operator.residual_var(native_x), dtype=np.float64)
    payload = timing.compact()
    payload.update(
        {
            "artifact": {
                "artifact_id": artifact.artifact_id,
                "reused": bool(artifact.reused),
                "build_wall_ms": float(build_wall_ms),
                "build_elapsed_ms": float(artifact.metadata["build"]["elapsed_ms"]),
            },
            "success_all": all(timing.success),
            "x": native_x.tolist(),
            "raw": native_raw.tolist(),
            "raw_norm": float(np.linalg.norm(native_raw)),
            "py_raw_at_veqlib_x_norm": float(np.linalg.norm(py_raw_at_native)),
        }
    )
    return payload


def _compact_py(engine: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_all": bool(engine["success_all"]),
        "timing": engine["timing"],
        "nfev": engine["nfev"],
        "raw_norm": float(engine["raw_norm"]),
        "message": str(engine["message"]),
        "x": np.asarray(engine["x"], dtype=np.float64).tolist(),
        "raw": np.asarray(engine["raw"], dtype=np.float64).tolist(),
    }


def _compare(cxx: dict[str, Any], py: dict[str, Any]) -> dict[str, Any]:
    x_diff = max_abs(cxx["x"], py["x"])
    raw_diff = max_abs(cxx["raw"], py["raw"])
    return {
        "x_max_abs": x_diff,
        "raw_max_abs": raw_diff,
        "within_atol": bool(x_diff <= VALIDATION_ATOL and raw_diff <= VALIDATION_ATOL),
    }


def _reference_dofs_payload(
    benchmark: ModuleType,
    case: RuntimeCase,
    cxx: dict[str, Any],
    py: dict[str, Any],
) -> dict[str, float]:
    reference = _benchmark_reference()
    active_profiles = case.py_operator.problem.active_profiles
    cxx_shape_x = benchmark._extract_shape_x(
        active_profiles,
        np.asarray(cxx["x"], dtype=np.float64),
    )
    py_shape_x = benchmark._extract_shape_x(
        active_profiles,
        np.asarray(py["x"], dtype=np.float64),
    )
    return {
        "veqlib_shape_error": float(
            benchmark._shape_error(reference.reference_shape_x, cxx_shape_x)
        ),
        "veqpy_shape_error": float(benchmark._shape_error(reference.reference_shape_x, py_shape_x)),
        "veqlib_vs_veqpy_shape_error": max_abs(cxx_shape_x, py_shape_x),
    }


def _topology_payload(topology: Topology, warnings_: tuple[str, ...]) -> dict[str, Any]:
    return {
        "status": "planned",
        "key": topology.key,
        "source": topology.source_policy_dict(),
        "layout": {
            "packed": topology.layout,
            "profile_first": topology.layout_profile_first,
            "code": topology.layout_code,
        },
        "profile_counts": {
            "h": topology.h_count,
            "v": topology.v_count,
            "kappa": topology.kappa_count,
            "psin": topology.psin_count,
            "F": topology.F_count,
            "c": list(topology.c_counts),
            "s": list(topology.s_counts),
        },
        "grid": {"Nr": topology.Nr, "Nt": topology.Nt},
        "sample_count": topology.sample_count,
        "warnings": list(warnings_),
    }


def _plan_row(
    benchmark: ModuleType,
    spec: Any,
    *,
    build: str,
    layout: str,
    build_options: dict[str, object] | None,
    cache_root: Path,
    source_dir: Path,
    skip_artifact_dry_run: bool,
) -> tuple[dict[str, Any], Topology | None]:
    try:
        topology, warning_messages = _topology_from_spec(
            benchmark,
            spec,
            build=build,
            layout=layout,
            build_options=build_options,
        )
    except Exception as exc:
        return (
            {
                "case": _spec_label(spec),
                "selector": _spec_selector(spec),
                "topology": {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"},
                "runtime": {"status": "skipped_invalid_topology"},
            },
            None,
        )

    topology_payload = _topology_payload(topology, warning_messages)
    if not skip_artifact_dry_run:
        artifact = build_kernel(
            topology,
            cache_root=cache_root,
            source_dir=source_dir,
            dry_run=True,
        )
        topology_payload["artifact"] = {
            "status": artifact.metadata["artifact"]["status"],
            "artifact_id": artifact.artifact_id,
            "metadata_path": str(artifact.metadata_path),
            "reused": bool(artifact.reused),
        }
    try:
        topology.validate_supported_for_veqlib_mvp()
    except TopologyError as exc:
        runtime = {"status": "blocked_unsupported_native_kernel", "reason": str(exc)}
    else:
        runtime = {"status": "ready_supported_native_kernel"}
    return (
        {
            "case": _spec_label(spec),
            "selector": _spec_selector(spec),
            "route": str(spec.mode),
            "coordinate": str(spec.coordinate),
            "nodes": str(spec.input_kind),
            "constraint": str(spec.constraint),
            "topology": topology_payload,
            "runtime": runtime,
        },
        topology,
    )


def _run_supported_row(
    benchmark: ModuleType,
    spec: Any,
    topology: Topology,
    *,
    registry: KernelRegistry,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    case = _runtime_case(benchmark, spec, topology)
    py = case.py_measure(warmup=warmup, repeat=repeat)
    cxx = _measure_veqlib(case, registry=registry, warmup=warmup, repeat=repeat)
    compare = _compare(cxx, py)
    reference_dofs = _reference_dofs_payload(benchmark, case, cxx, py)
    passed = cxx["success_all"] and py["success_all"] and compare["within_atol"]
    failure_reason = None
    if not cxx["success_all"]:
        failure_reason = "veqlib_solve_failed"
    elif not py["success_all"]:
        failure_reason = "veqpy_solve_failed"
    elif not compare["within_atol"]:
        failure_reason = "validation_mismatch"
    runtime = {
        "status": "passed" if passed else "failed",
        "x_size": case.x_size,
        "solver_policy": {
            "initial": NATIVE_SOLVER_INITIAL_POLICY,
            "continue": NATIVE_SOLVER_CONTINUATION_POLICY,
            "norm": NATIVE_SOLVER_NORMALIZATION,
        },
        "engines": {
            case.solver_engine_label: cxx,
            "veqpy-numba-hybr": _compact_py(py),
        },
        "closeness_to_numba": compare,
        "reference_dofs": reference_dofs,
    }
    if failure_reason is not None:
        runtime["failure_reason"] = failure_reason
    return runtime


def _run_supported_row_subprocess(
    spec: Any,
    *,
    args: argparse.Namespace,
    cache_root: Path,
    source_dir: Path,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix="veqlib-route-row-",
        suffix=".json",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scope",
        str(args.scope),
        "--case",
        _spec_selector(spec),
        "--build",
        str(args.build),
        "--layout",
        str(args.layout),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--output",
        str(output_path),
        "--cache-root",
        str(cache_root),
        "--source-dir",
        str(source_dir),
        "--skip-artifact-dry-run",
        "--run-native-in-process",
        "--quiet-progress",
    ]
    _append_optional_arg(command, "--cmake-build-type", args.cmake_build_type)
    _append_optional_arg(command, "--fp-mode", args.fp_mode)
    _append_optional_arg(command, "--enzyme-jacobian-batch-width", args.enzyme_jacobian_batch_width)
    _append_bool_override_arg(command, "--enable-enzyme", "--disable-enzyme", args.enable_enzyme)
    _append_bool_override_arg(
        command,
        "--enable-native-optimizations",
        "--disable-native-optimizations",
        args.enable_native_optimizations,
    )
    _append_bool_override_arg(
        command,
        "--enable-thin-lto",
        "--disable-thin-lto",
        args.enable_thin_lto,
    )
    _append_bool_override_arg(command, "--analysis", "--no-analysis", args.analysis)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return {
                "status": "failed",
                "error": "subprocess_failed",
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            return {
                "status": "failed",
                "error": "subprocess_returned_unexpected_row_count",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        return dict(rows[0]["runtime"])
    finally:
        output_path.unlink(missing_ok=True)


def _append_optional_arg(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _append_bool_override_arg(
    command: list[str],
    positive: str,
    negative: str,
    value: bool | None,
) -> None:
    if value is True:
        command.append(positive)
    elif value is False:
        command.append(negative)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "topology_planned": 0,
        "topology_invalid": 0,
        "native_ready": 0,
        "native_blocked": 0,
        "runtime_passed": 0,
        "runtime_failed": 0,
        "runtime_not_requested": 0,
    }
    for row in rows:
        if row["topology"]["status"] == "planned":
            summary["topology_planned"] += 1
        else:
            summary["topology_invalid"] += 1
        status = row["runtime"]["status"]
        if status == "blocked_unsupported_native_kernel":
            summary["native_blocked"] += 1
        elif status in {"ready_supported_native_kernel", "not_requested", "passed", "failed"}:
            summary["native_ready"] += 1
            if status == "not_requested":
                summary["runtime_not_requested"] += 1
            elif status == "passed":
                summary["runtime_passed"] += 1
            elif status == "failed":
                summary["runtime_failed"] += 1
    return summary


def _build_option_overrides(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    for name in (
        "cmake_build_type",
        "fp_mode",
        "enable_enzyme",
        "enable_native_optimizations",
        "enable_thin_lto",
        "analysis",
        "enzyme_jacobian_batch_width",
    ):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    return options


def _add_bool_override(
    parser: argparse.ArgumentParser,
    *,
    positive: str,
    negative: str,
    dest: str,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(positive, dest=dest, action="store_true", default=None, help=help_text)
    group.add_argument(negative, dest=dest, action="store_false", help=argparse.SUPPRESS)


def _native_engine_payload(runtime: dict[str, Any]) -> dict[str, Any] | None:
    engines = runtime.get("engines")
    if not isinstance(engines, dict):
        return None
    for name, payload in engines.items():
        if str(name).startswith("veqlib-") and isinstance(payload, dict):
            return payload
    return None


def _veqpy_engine_payload(runtime: dict[str, Any]) -> dict[str, Any] | None:
    engines = runtime.get("engines")
    if not isinstance(engines, dict):
        return None
    payload = engines.get("veqpy-numba-hybr")
    return payload if isinstance(payload, dict) else None


def _timing_median_ms(engine: dict[str, Any] | None) -> float:
    if engine is None:
        return float("nan")
    timing = engine.get("timing")
    if not isinstance(timing, dict):
        return float("nan")
    return float(timing.get("median_ms", float("nan")))


def _format_optional_float(value: float, *, decimals: int = 6) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{decimals}f}"


def _format_optional_sci(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2e}"


def _format_optional_speedup(py_ms: float, cxx_ms: float) -> str:
    if not np.isfinite(py_ms) or not np.isfinite(cxx_ms) or cxx_ms <= 0.0:
        return "n/a"
    return f"{py_ms / cxx_ms:.3f}x"


def _progress_context(console: Console, *, quiet: bool) -> Any:
    if quiet:
        return nullcontext(None)
    return Progress(
        TextColumn("[dim]{task.fields[current]:<24.24}[/]"),
        BarColumn(
            bar_width=48,
            complete_style="cyan",
            finished_style="green",
            pulse_style="cyan",
        ),
        MofNCompleteColumn(),
        TextColumn("{task.fields[phase]:>8}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _print_config_tree(
    console: Console,
    *,
    scope: str,
    specs: tuple[Any, ...],
    build: str,
    layout: str,
    repeat: int,
    warmup: int,
) -> None:
    console.print(Text("[config]", style="bold cyan"))
    lines = (
        f"scope: [green]{scope}[/]",
        f"cases: [green]{len(specs)}[/]",
        f"build: [green]{build}/{layout}[/]",
        f"initial: [green]{NATIVE_SOLVER_INITIAL_POLICY}[/]",
        f"continue: [green]{NATIVE_SOLVER_CONTINUATION_POLICY}[/]",
        f"norm: [green]{NATIVE_SOLVER_NORMALIZATION}[/]",
        f"warmup: [green]{warmup}[/]",
        f"repeat: [green]{repeat}[/]",
    )
    for index, line in enumerate(lines):
        branch = "└──" if index == len(lines) - 1 else "├──"
        console.print(f"  {branch} {line}")


def _print_outputs_tree(console: Console, outputs: dict[str, Path]) -> None:
    if not outputs:
        return
    console.print(Text("[outputs]", style="bold cyan"))
    paths: list[Path] = []
    for path in outputs.values():
        try:
            display_path = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        paths.append(display_path)
    for index, path in enumerate(paths):
        branch = "└──" if index == len(paths) - 1 else "├──"
        console.print(f"  {branch} [green]{path}[/]")


def _status_cell(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]not requested[/]"
    if text.startswith("blocked"):
        return "[yellow]blocked[/]"
    if text.startswith("skipped") or text == "invalid":
        return "[yellow]skipped[/]"
    return text


def _progress_phase(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]skip[/]"
    if text.startswith("blocked"):
        return "[yellow]blocked[/]"
    if text.startswith("skipped"):
        return "[yellow]skipped[/]"
    return "[dim]done[/]"


def _failure_detail(runtime: dict[str, Any]) -> str:
    if runtime.get("status") == "not_requested":
        return "n/a"
    reason = runtime.get("failure_reason")
    if reason in {"veqlib_solve_failed", "veqpy_solve_failed"}:
        return str(reason).replace("_", " ")
    closeness = runtime.get("closeness_to_numba")
    if not isinstance(closeness, dict):
        return "n/a"
    if bool(closeness.get("within_atol", False)):
        return "[green]ok[/]"
    dx = float(closeness.get("x_max_abs", float("nan")))
    draw = float(closeness.get("raw_max_abs", float("nan")))
    return f"Δx={_format_optional_sci(dx)} Δr={_format_optional_sci(draw)}"


def _print_summary(
    console: Console,
    summary: dict[str, int],
) -> None:
    counts = Table(
        box=REPORT_TABLE_BOX,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    counts.add_column("summary")
    counts.add_column("count", justify="right")
    for key in (
        "total",
        "topology_planned",
        "native_ready",
        "native_blocked",
        "runtime_passed",
        "runtime_failed",
        "runtime_not_requested",
    ):
        counts.add_row(key.replace("_", " "), str(summary[key]))
    console.print(counts)


def _print_failures(console: Console, rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row["runtime"]["status"] == "failed"]
    if not failed:
        return
    console.print()
    tree = Tree(Text("[failures]", style="bold red"))
    for row in failed:
        tree.add(f"{row.get('case', 'n/a')}: {_failure_detail(row['runtime'])}")
    console.print(tree)
    console.print()


def _print_timing_table(
    console: Console,
    rows: list[dict[str, Any]],
) -> None:
    table = Table(
        box=REPORT_TABLE_BOX,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("case", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("x", justify="right")
    table.add_column(Text("Cxx (ms)"), justify="right")
    table.add_column(Text("Numba (ms)"), justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("diff", justify="right")
    for row in rows:
        runtime = row["runtime"]
        native = _native_engine_payload(runtime)
        py = _veqpy_engine_payload(runtime)
        cxx_ms = _timing_median_ms(native)
        py_ms = _timing_median_ms(py)
        closeness = runtime.get("closeness_to_numba")
        closeness = closeness if isinstance(closeness, dict) else {}
        table.add_row(
            str(row.get("case", "n/a")),
            _status_cell(runtime["status"]),
            str(runtime.get("x_size", "n/a")),
            _format_optional_float(cxx_ms),
            _format_optional_float(py_ms),
            _format_optional_speedup(py_ms, cxx_ms),
            _format_optional_sci(float(closeness.get("x_max_abs", float("nan")))),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(DEFAULT_SCOPE, "uniform", "full"),
        default=DEFAULT_SCOPE,
    )
    parser.add_argument("--case", action="append", help="Case name or route:coord:nodes:constraint")
    parser.add_argument("--build", default="fastmath")
    parser.add_argument("--layout", default="degree")
    parser.add_argument("--cmake-build-type", default=None)
    parser.add_argument("--fp-mode", default=None)
    parser.add_argument("--enzyme-jacobian-batch-width", type=int, default=None)
    _add_bool_override(
        parser,
        positive="--enable-enzyme",
        negative="--disable-enzyme",
        dest="enable_enzyme",
        help_text="Override Enzyme.",
    )
    _add_bool_override(
        parser,
        positive="--enable-native-optimizations",
        negative="--disable-native-optimizations",
        dest="enable_native_optimizations",
        help_text="Override native CPU flags.",
    )
    _add_bool_override(
        parser,
        positive="--enable-thin-lto",
        negative="--disable-thin-lto",
        dest="enable_thin_lto",
        help_text="Override ThinLTO.",
    )
    _add_bool_override(
        parser,
        positive="--analysis",
        negative="--no-analysis",
        dest="analysis",
        help_text="Override analysis build.",
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--run-native-in-process", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-artifact-dry-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    console = _console()
    benchmark = _benchmark_module()
    specs = _filter_specs(
        _iter_route_specs(benchmark, scope=args.scope),
        set(args.case) if args.case else None,
    )
    cache_root = args.cache_root or default_kernel_cache_root()
    source_dir = args.source_dir.resolve()
    registry = KernelRegistry(cache_root=cache_root, source_dir=source_dir)
    build_options = _build_option_overrides(args)

    rows: list[dict[str, Any]] = []
    if not args.quiet_progress:
        _print_config_tree(
            console,
            scope=str(args.scope),
            specs=specs,
            build=str(args.build),
            layout=str(args.layout),
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))
    with _progress_context(console, quiet=args.quiet_progress) as progress:
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                "routes",
                total=len(specs),
                current="-",
                phase="[dim]plan[/]",
            )
        for spec in specs:
            selector = _spec_selector(spec)
            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    current=selector,
                    phase="[dim]plan[/]",
                )
            row, topology = _plan_row(
                benchmark,
                spec,
                build=args.build,
                layout=args.layout,
                build_options=build_options,
                cache_root=cache_root,
                source_dir=source_dir,
                skip_artifact_dry_run=args.skip_artifact_dry_run,
            )
            if row["runtime"]["status"] == "ready_supported_native_kernel":
                if args.no_run:
                    row["runtime"] = {"status": "not_requested"}
                elif topology is not None:
                    if progress is not None and task_id is not None:
                        progress.update(task_id, phase="[cyan]run[/]")
                    try:
                        if args.run_native_in_process:
                            row["runtime"] = _run_supported_row(
                                benchmark,
                                spec,
                                topology,
                                registry=registry,
                                warmup=args.warmup,
                                repeat=args.repeat,
                            )
                        else:
                            row["runtime"] = _run_supported_row_subprocess(
                                spec,
                                args=args,
                                cache_root=cache_root,
                                source_dir=source_dir,
                            )
                    except Exception as exc:
                        row["runtime"] = {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            rows.append(row)
            if progress is not None and task_id is not None:
                progress.update(task_id, phase=_progress_phase(row["runtime"]["status"]))
                progress.advance(task_id)

    summary = _summarize(rows)
    payload = {
        "schema": "veqlib.routes.v2",
        "scope": str(args.scope),
        "case_count": len(rows),
        "build": str(args.build),
        "build_option_overrides": build_options,
        "layout": str(args.layout),
        "solver_policy": {
            "initial": NATIVE_SOLVER_INITIAL_POLICY,
            "continue": NATIVE_SOLVER_CONTINUATION_POLICY,
            "norm": NATIVE_SOLVER_NORMALIZATION,
        },
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "native_isolation": "in_process" if args.run_native_in_process else "subprocess_per_row",
        "validation_atol": VALIDATION_ATOL,
        "cache_root": str(cache_root),
        "source_dir": str(source_dir),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        console.print()
        _print_outputs_tree(console, {"json": args.output})
    _print_summary(console, summary)
    _print_failures(console, rows)
    _print_timing_table(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
