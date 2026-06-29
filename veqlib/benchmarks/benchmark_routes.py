#!/usr/bin/env python3
"""Full route/topology benchmark for VEQlib against VEQPy.

The default scope enumerates the 46 historical uniform source cases from
``tests/benchmark.py``.  ``--scope full`` adds grid-sampled variants for the 92
route/topology matrix.  Topology planning is always performed; supported native
rows can be executed through the typed ``veqlib.facade`` runtime unless
``--no-run`` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from veqlib.benchmarks._common import (
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
    temp_cache,
    write_json,
)
from veqlib.facade import (
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_POWELL,
    KernelBoundary,
    KernelBuild,
    KernelInput,
    KernelRegistry,
    KernelSolve,
    KernelTopology,
    TopologyError,
    VEQlibSolver,
    build_kernel,
)
from veqpy.operator import Operator
from veqpy.operator.packed_layout import build_profile_layout, build_profile_names
from veqpy.solver import Solver

DEFAULT_OUTPUT = Path("/tmp/veqlib_routes.json")
VALIDATION_ATOL = 1.0e-6
Topology = KernelTopology


@dataclass(frozen=True, slots=True)
class RuntimeCase:
    spec: Any
    topology: Topology
    kernel_input: KernelInput
    kernel_solve: KernelSolve
    py_operator: Any
    py_measure: Any
    solver_method_code: int
    solver_engine_label: str
    x_size: int


_SOLVER_LABELS = {
    SOLVER_METHOD_POWELL: "veqlib-fastmath-powell",
    SOLVER_METHOD_LEVENBERG_MARQUARDT: "veqlib-fastmath-lm",
}


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


def _iter_route_specs(benchmark: ModuleType, *, include_grid: bool) -> tuple[Any, ...]:
    input_kinds = list(benchmark.BENCHMARK_INPUT_KINDS)
    if include_grid and "grid" not in input_kinds:
        input_kinds.append("grid")
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
        for constraint in benchmark.BENCHMARK_MODE_CONSTRAINTS[mode]
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
        boundary=_kernel_boundary_from_case(case),
        scaled_heat=np.asarray(source_plan.scaled_heat, dtype=np.float64),
        scaled_current=np.asarray(source_plan.scaled_current, dtype=np.float64),
        scaled_Ip=float(source_plan.scaled_Ip),
        beta=float(source_plan.beta),
        fix_rho=float(operator.fix_rho),
        case_name=case_name,
    )


def _kernel_solve_from_config(config: Any, *, method: int, x_size: int) -> KernelSolve:
    return KernelSolve(
        method=method,
        max_residual=float(config.max_residual),
        max_evaluations=int(x_size) ** 2,
        initial="cold",
        norm="fast",
        residual_normalization_floor=float(config.residual_normalization_floor),
        residual_normalization_max_ratio=float(config.residual_normalization_max_ratio),
        residual_normalization_huber_tau=float(config.residual_normalization_huber_tau),
        residual_normalization_probe_count=int(config.residual_normalization_probe_count),
        residual_normalization_probe_step=float(config.residual_normalization_probe_step),
        residual_normalization_sensitivity_lambda=float(config.residual_normalization_sensitivity_lambda),
    )


def _runtime_case(benchmark: ModuleType, spec: Any, topology: Topology) -> RuntimeCase:
    reference = _benchmark_reference()
    case = benchmark._make_benchmark_case(spec, reference)
    coeffs = benchmark._case_profile_coeffs(spec)
    grid = benchmark.TEST_GRID
    operator = Operator(grid, case)
    x0 = operator.pack_coefficients(benchmark._coefficients_from_coeffs(coeffs))
    method = _cxx_solver_method_for_spec(spec)
    kernel_input = _kernel_input_from_operator(
        case,
        operator,
        case_name=_spec_label(spec),
    )
    kernel_solve = _kernel_solve_from_config(benchmark.CONFIG, method=method, x_size=int(x0.size))

    def measure_py(*, warmup: int, repeat: int) -> dict[str, Any]:
        for _ in range(max(1, warmup)):
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
        kernel_input=kernel_input,
        kernel_solve=kernel_solve,
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
            *case.kernel_input.runtime_args(),
            *case.kernel_solve.runtime_args(x_size=case.x_size),
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
    passed = cxx["success_all"] and py["success_all"] and compare["within_atol"]
    return {
        "status": "passed" if passed else "failed",
        "x_size": case.x_size,
        "engines": {
            case.solver_engine_label: cxx,
            "veqpy-numba-hybr": _compact_py(py),
        },
        "closeness_to_numba": compare,
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("uniform", "full"), default="uniform")
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
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--skip-artifact-dry-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    benchmark = _benchmark_module()
    specs = _filter_specs(
        _iter_route_specs(benchmark, include_grid=args.scope == "full"),
        set(args.case) if args.case else None,
    )
    cache_root = args.cache_root or temp_cache("veqlib-routes-")
    source_dir = args.source_dir.resolve()
    registry = KernelRegistry(cache_root=cache_root, source_dir=source_dir)
    build_options = _build_option_overrides(args)

    rows: list[dict[str, Any]] = []
    for spec in specs:
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
                print(f"[routes] running {_spec_label(spec)}", flush=True)
                try:
                    row["runtime"] = _run_supported_row(
                        benchmark,
                        spec,
                        topology,
                        registry=registry,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                except Exception as exc:
                    row["runtime"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    payload = {
        "schema": "veqlib.routes.v2",
        "scope": str(args.scope),
        "case_count": len(rows),
        "build": str(args.build),
        "build_option_overrides": build_options,
        "layout": str(args.layout),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "validation_atol": VALIDATION_ATOL,
        "cache_root": str(cache_root),
        "source_dir": str(source_dir),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "summary": _summarize(rows),
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        print(f"json: {args.output}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
