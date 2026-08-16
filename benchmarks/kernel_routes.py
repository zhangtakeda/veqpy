#!/usr/bin/env python3
"""Benchmark the four production Kernels over all VEQ route/coordinate pairs.

The matrix is deliberately fixed at four backends, seven source routes, and
three source coordinates.  Every case is manufactured from one converged
reference equilibrium and uses physical, unconstrained source profiles.  This
keeps Ip/beta normalization policy out of the timing dimension while checking
that every route reconstructs the same equilibrium family.

Artifact builds, imports, the reference solve, and Numba JIT warmup are outside
the formal timing samples.  The timed path is the prepared Kernel solve after
one Adapter fill.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    benchmark_result_path,
    cpu_affinity,
    default_kernel_cache_root,
    integer_statistics,
    monotonic_interleave,
    prepare_metadata,
    runtime_env,
    statistics_payload,
    time_call,
    write_json,
)
from veqpy import build
from veqpy.config import normalize_topology
from veqpy.kernels.abi.enums import source_driver_for

BACKENDS = ("numba", "cxx-strict", "cxx-relaxed", "cxx-enzyme")
ROUTES = ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ")
COORDINATES = ("r", "rho", "psin")

DEFAULT_OUTPUT = benchmark_result_path("kernel_routes")
DEFAULT_SOURCE_COUNT = 51
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 100
DEFAULT_MAX_EVALUATIONS = 1000

MU0 = 4.0e-7 * np.pi
SOLUTION_ATOL = 1.0e-6
STRICT_RESIDUAL_ATOL = 1.0e-10
RELAXED_RESIDUAL_ATOL = 1.0e-8
PSIN_FIXED_POINT_RESIDUAL_ATOL = 1.0e-9
REFERENCE_PSIN_ATOL = 1.0e-2


@dataclass(frozen=True, slots=True)
class RouteReference:
    """Frozen physical profiles shared by all route benchmark cases."""

    boundary: dict[str, Any]
    axes: dict[str, np.ndarray]
    pressure: dict[str, np.ndarray]
    drivers: dict[str, np.ndarray | dict[str, np.ndarray]]
    p0: float
    r: np.ndarray
    psin: np.ndarray
    Ip: float
    beta: float
    solve: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RouteCase:
    """One route/coordinate case expressed through the public dictionary API."""

    route: str
    coordinate: str
    topology: dict[str, Any]
    solver: dict[str, Any]
    boundary: dict[str, Any]
    source: dict[str, Any]
    targets: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.route}:{self.coordinate}"


def _profile_counts(route: str, coordinate: str) -> dict[str, Any]:
    sampled_psin = coordinate == "psin" and route not in {"PJ2", "PJ3"}
    optimized_f = coordinate == "psin" and route in {"PJ2", "PJ3"}
    return {
        "h_count": 3,
        "v_count": 0,
        "kappa_count": 6,
        "psin_count": 6 if sampled_psin else 0,
        "F_count": 6 if optimized_f else 0,
        "c_counts": (),
        "s_counts": (3,),
    }


def _topology(route: str, coordinate: str, *, constraint: str) -> dict[str, Any]:
    return {
        **_profile_counts(route, coordinate),
        "Nr": 32,
        "Nt": 16,
        "route": route,
        "coordinate": coordinate,
        "constraint": constraint,
        "quadrature": "legendre",
        "calculus": "spectral",
        "L_max": 20,
        "M_max": 7,
        "K_max": 20,
    }


def _solver(max_evaluations: int) -> dict[str, Any]:
    return {
        "method": "powell",
        "max_residual": 1.0e-6,
        "max_evaluations": int(max_evaluations),
        "initial": "cold",
        "continuation": "cold",
        "norm": "fast",
    }


def _boundary() -> dict[str, Any]:
    return {
        "a": 1.05 / 1.85,
        "R0": 1.05,
        "Z0": 0.0,
        "B0": 3.0,
        "kappa_lcfs": 2.2,
        "c_lcfs": np.zeros(1, dtype=np.float64),
        "s_lcfs": np.asarray([np.arcsin(0.5)], dtype=np.float64),
    }


def _reference_source() -> dict[str, Any]:
    r = np.linspace(0.0, 1.0, DEFAULT_SOURCE_COUNT, dtype=np.float64)
    psin = r * r
    beta_fraction = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_p = np.exp(alpha_p)
    exp_f = np.exp(alpha_f)
    normalized_ff_psin = (
        (1.0 - beta_fraction)
        * alpha_f
        * (np.exp(alpha_f * psin) - exp_f)
        / (1.0 + exp_f * (alpha_f - 1.0))
    )
    normalized_p_psin = (
        beta_fraction
        * alpha_p
        * (np.exp(alpha_p * psin) - exp_p)
        / (1.0 + exp_p * (alpha_p - 1.0))
    )
    return {
        "r": r,
        "P_r": normalized_p_psin * (2.0 * r) / MU0,
        "FF_r": normalized_ff_psin * (2.0 * r),
        "P0": 0.0,
    }


def _safe_divisor(values: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    signs = np.where(values < 0.0, -1.0, 1.0)
    return np.where(np.abs(values) < floor, signs * floor, values)


def _interpolate(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    profile = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    order = np.argsort(source)
    unique_axis, indices = np.unique(source[order], return_index=True)
    unique_profile = profile[order][indices]
    if unique_axis.size < 2:
        raise ValueError("reference profile does not define a usable coordinate axis")
    return np.interp(
        target,
        unique_axis,
        unique_profile,
        left=float(unique_profile[0]),
        right=float(unique_profile[-1]),
    )


def _make_reference(*, max_evaluations: int, artifact_dir: Path) -> RouteReference:
    boundary = _boundary()
    module = build(
        topology=_topology("PF", "r", constraint="ip"),
        solver=_solver(max_evaluations),
        backend="numba",
        artifact_dir=artifact_dir,
        materialize=True,
        verbose=False,
        report=False,
    )
    try:
        record = module.solve(
            boundary=boundary,
            source=_reference_source(),
            targets={"Ip": 3.0e6},
            materialize=True,
            verbose=False,
            report=False,
        )
        if not record.accepted or record.equilibrium is None:
            raise RuntimeError(
                "reference PF/r solve failed: "
                f"residual={record.residual_norm:.6e}, evaluations={record.evaluations}"
            )
        equilibrium = record.equilibrium
    finally:
        module.close()

    rho_r = _safe_divisor(np.asarray(equilibrium.rho_r, dtype=np.float64))
    return RouteReference(
        boundary=boundary,
        axes={
            "r": np.asarray(equilibrium.r, dtype=np.float64),
            "rho": np.asarray(equilibrium.rho, dtype=np.float64),
            "psin": np.asarray(equilibrium.psin, dtype=np.float64),
        },
        pressure={
            "r": np.asarray(equilibrium.P_r, dtype=np.float64),
            "rho": np.asarray(equilibrium.P_r, dtype=np.float64) / rho_r,
            "psin": np.asarray(equilibrium.P_psin, dtype=np.float64),
        },
        drivers={
            "PF": {
                "r": np.asarray(equilibrium.FF_r, dtype=np.float64),
                "rho": np.asarray(equilibrium.FF_r, dtype=np.float64) / rho_r,
                "psin": np.asarray(equilibrium.FF_psin, dtype=np.float64),
            },
            "PP": np.asarray(equilibrium.psi_r, dtype=np.float64),
            "PI": np.asarray(equilibrium.Itor, dtype=np.float64),
            "PJ1": np.asarray(equilibrium.jtor, dtype=np.float64),
            "PJ2": np.asarray(equilibrium.jpara, dtype=np.float64),
            "PJ3": np.asarray(equilibrium.jtotal, dtype=np.float64),
            "PQ": np.asarray(equilibrium.q, dtype=np.float64),
        },
        p0=float(equilibrium.P0),
        r=np.asarray(equilibrium.r, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        Ip=float(equilibrium.Ip),
        beta=float(equilibrium.betat),
        solve={
            "residual_norm": float(record.residual_norm),
            "scaled_residual_norm": float(record.scaled_residual_norm),
            "evaluations": int(record.evaluations),
        },
    )


def _make_case(
    reference: RouteReference,
    route: str,
    coordinate: str,
    *,
    max_evaluations: int,
) -> RouteCase:
    nodes = np.linspace(0.0, 1.0, DEFAULT_SOURCE_COUNT, dtype=np.float64)
    axis = reference.axes[coordinate]
    driver_value = reference.drivers[route]
    driver = driver_value[coordinate] if isinstance(driver_value, dict) else driver_value
    source = {
        coordinate: nodes,
        f"P_{coordinate}": _interpolate(axis, reference.pressure[coordinate], nodes),
        source_driver_for(route, coordinate): _interpolate(axis, driver, nodes),
        "P0": reference.p0,
    }
    return RouteCase(
        route=route,
        coordinate=coordinate,
        topology=_topology(route, coordinate, constraint="none"),
        solver=_solver(max_evaluations),
        boundary=reference.boundary,
        source=source,
        targets={},
    )


def _snapshot_output(output: Any) -> dict[str, Any]:
    return {
        "success": bool(output.success),
        "info": int(output.info),
        "nfev": int(output.nfev),
        "njev": int(output.njev),
        "raw_norm": float(output.raw_norm),
        "scaled_norm": float(output.scaled_norm),
        "elapsed_ms": float(output.elapsed_ms),
        "solve_ms": float(output.solve_ms),
        "x": np.asarray(output.x, dtype=np.float64).copy(),
        "raw": np.asarray(output.raw, dtype=np.float64).copy(),
    }


def _physics_diagnostics(equilibrium: Any, reference: RouteReference) -> dict[str, float]:
    target_psin = _interpolate(reference.r, reference.psin, equilibrium.r)
    return {
        "psin_max_abs": float(
            np.max(np.abs(np.asarray(equilibrium.psin, dtype=np.float64) - target_psin))
        ),
        "Ip_relative_error": float(
            abs(float(equilibrium.Ip) - reference.Ip) / max(abs(reference.Ip), 1.0)
        ),
        "beta_relative_error": float(
            abs(float(equilibrium.betat) - reference.beta)
            / max(abs(reference.beta), 1.0e-12)
        ),
    }


def _empty_engine(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason}


def _measure_case(
    case: RouteCase,
    reference: RouteReference,
    *,
    backends: tuple[str, ...],
    artifact_dir: Path,
    rebuild: bool,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    engines: dict[str, dict[str, Any]] = {
        backend: _empty_engine("not-selected", "backend filter")
        for backend in BACKENDS
        if backend not in backends
    }
    measurements: dict[str, dict[str, list[Any]]] = {}

    for backend in backends:
        try:
            module = build(
                topology=case.topology,
                solver=case.solver,
                backend=backend,
                artifact_dir=artifact_dir,
                rebuild=rebuild,
                materialize=False,
                verbose=False,
                report=False,
            )
            module._adapter.fill(case.boundary, case.source, case.targets)
            modules[backend] = module
            engines[backend] = {
                "status": "prepared",
                "metadata": prepare_metadata(module, backend),
            }
        except Exception as error:
            engines[backend] = _empty_engine(
                "failed",
                f"{type(error).__name__}: {error}",
            )

    active = tuple(backend for backend in backends if backend in modules)
    try:
        for backend in active:
            for _ in range(warmup):
                output = modules[backend]._kernel.solve()
                if not output.success:
                    raise RuntimeError(f"{backend} warmup failed for {case.label}")

        measurements = {
            backend: {
                "wall_ms": [],
                "elapsed_ms": [],
                "solve_ms": [],
                "nfev": [],
                "njev": [],
                "accepted": [],
                "last": [],
            }
            for backend in active
        }
        for iteration in range(repeat):
            for backend in monotonic_interleave(active, iteration):
                output, wall_ms = time_call(modules[backend]._kernel.solve)
                snapshot = _snapshot_output(output)
                measured = measurements[backend]
                measured["wall_ms"].append(wall_ms)
                measured["elapsed_ms"].append(snapshot["elapsed_ms"])
                measured["solve_ms"].append(snapshot["solve_ms"])
                measured["nfev"].append(snapshot["nfev"])
                measured["njev"].append(snapshot["njev"])
                measured["accepted"].append(
                    snapshot["success"] and np.isfinite(snapshot["raw_norm"])
                )
                measured["last"] = [snapshot]

        for backend in active:
            measured = measurements[backend]
            last = measured["last"][0]
            equilibrium = modules[backend]._kernel.build_equilibrium()
            engines[backend].update(
                {
                    "status": "measured",
                    "timing_ms": statistics_payload(measured["elapsed_ms"]),
                    "wall_timing_ms": statistics_payload(measured["wall_ms"]),
                    "solve_timing_ms": statistics_payload(measured["solve_ms"]),
                    "nfev": integer_statistics(measured["nfev"]),
                    "njev": integer_statistics(measured["njev"]),
                    "accepted_all": all(measured["accepted"]),
                    "last": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in last.items()
                    },
                    "physics": _physics_diagnostics(equilibrium, reference),
                }
            )

        numba_last = measurements.get("numba", {}).get("last", [])
        if numba_last:
            reference_x = np.asarray(numba_last[0]["x"], dtype=np.float64)
            reference_raw = np.asarray(numba_last[0]["raw"], dtype=np.float64)
            for backend in active:
                last = measurements[backend]["last"][0]
                current_x = np.asarray(last["x"], dtype=np.float64)
                current_raw = np.asarray(last["raw"], dtype=np.float64)
                same_input_raw = np.asarray(
                    modules[backend]._kernel.residual(reference_x),
                    dtype=np.float64,
                )
                engines[backend]["parity_to_numba"] = {
                    "x_max_abs": float(np.max(np.abs(current_x - reference_x))),
                    "raw_max_abs": float(np.max(np.abs(current_raw - reference_raw))),
                    "same_input_raw_max_abs": float(
                        np.max(np.abs(same_input_raw - reference_raw))
                    ),
                }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        for backend in active:
            if engines[backend].get("status") in {"prepared", "measured"}:
                engines[backend] = _empty_engine("failed", message)
    finally:
        for module in modules.values():
            module.close()

    checks: dict[str, dict[str, Any]] = {}
    for backend in backends:
        engine = engines.get(backend, {})
        parity = engine.get("parity_to_numba", {})
        physics = engine.get("physics", {})
        residual_atol = STRICT_RESIDUAL_ATOL
        if case.route in {"PJ2", "PJ3"} and case.coordinate == "psin":
            # This route owns a source-local fixed point with a 1e-10 stopping
            # threshold. Re-evaluating the same x can therefore differ by a
            # few termination-scale units even within the Numba baseline.
            residual_atol = PSIN_FIXED_POINT_RESIDUAL_ATOL
        if backend in {"cxx-relaxed", "cxx-enzyme"}:
            residual_atol = max(residual_atol, RELAXED_RESIDUAL_ATOL)
        backend_checks = {
            "measured": engine.get("status") == "measured",
            "accepted_all": engine.get("accepted_all") is True,
            "solution_parity": float(parity.get("x_max_abs", float("inf")))
            <= SOLUTION_ATOL,
            "solution_residual_parity": float(
                parity.get("raw_max_abs", float("inf"))
            )
            <= SOLUTION_ATOL,
            "same_input_residual_parity": float(
                parity.get("same_input_raw_max_abs", float("inf"))
            )
            <= residual_atol,
            "reference_psin": float(physics.get("psin_max_abs", float("inf")))
            <= REFERENCE_PSIN_ATOL,
        }
        checks[backend] = {
            "status": "passed" if all(backend_checks.values()) else "failed",
            "checks": backend_checks,
            "same_input_residual_atol": residual_atol,
        }
        if engine.get("status") == "measured":
            engine["status"] = checks[backend]["status"]

    passed = bool(checks) and all(item["status"] == "passed" for item in checks.values())
    return {
        "label": case.label,
        "route": case.route,
        "coordinate": case.coordinate,
        "constraint": "none",
        "source_count": int(np.asarray(case.source[case.coordinate]).size),
        "topology": case.topology,
        "solver": case.solver,
        "status": "passed" if passed else "failed",
        "backends": engines,
        "correctness": {
            "status": "passed" if passed else "failed",
            "solution_atol": SOLUTION_ATOL,
            "reference_psin_atol": REFERENCE_PSIN_ATOL,
            "backends": checks,
        },
    }


def _planned_row(route: str, coordinate: str, backends: tuple[str, ...]) -> dict[str, Any]:
    topology = _topology(route, coordinate, constraint="none")
    normalized = normalize_topology(topology)
    return {
        "label": f"{route}:{coordinate}",
        "route": route,
        "coordinate": coordinate,
        "constraint": "none",
        "source_count": DEFAULT_SOURCE_COUNT,
        "topology": topology,
        "x_size": int(normalized.x_size),
        "status": "planned",
        "backends": {
            backend: _empty_engine(
                "planned" if backend in backends else "not-selected",
                "--no-run",
            )
            for backend in BACKENDS
        },
    }


def _summary(rows: list[dict[str, Any]], backends: tuple[str, ...]) -> dict[str, Any]:
    cells = [
        row["backends"][backend]
        for row in rows
        for backend in backends
    ]
    return {
        "route_coordinate_rows": len(rows),
        "selected_backend_count": len(backends),
        "matrix_cells": len(cells),
        "passed_cells": sum(item.get("status") == "passed" for item in cells),
        "failed_cells": sum(item.get("status") == "failed" for item in cells),
        "planned_cells": sum(item.get("status") == "planned" for item in cells),
        "passed_rows": sum(row.get("status") == "passed" for row in rows),
        "failed_rows": sum(row.get("status") == "failed" for row in rows),
    }


def _print_table(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(title="VEQPy Kernel route matrix", show_lines=False)
    table.add_column("route/coordinate", no_wrap=True)
    for backend in BACKENDS:
        table.add_column(backend, justify="right", no_wrap=True)
    for row in rows:
        values = [row["label"]]
        for backend in BACKENDS:
            engine = row.get("backends", {}).get(backend, {})
            median = engine.get("timing_ms", {}).get("median_ms")
            status = engine.get("status", "n/a")
            values.append(status if median is None else f"{status} {median:.3g} ms")
        table.add_row(*values)
    console.print(table)


def _selected(values: list[str] | None, choices: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return choices
    requested = set(values)
    return tuple(value for value in choices if value in requested)


def _worker_command(
    *,
    route: str,
    coordinate: str,
    backends: tuple[str, ...],
    artifact_dir: Path,
    rebuild: bool,
    warmup: int,
    repeat: int,
    max_evaluations: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-row",
        route,
        coordinate,
        "--artifact-dir",
        str(artifact_dir),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--max-evaluations",
        str(max_evaluations),
    ]
    for backend in backends:
        command.extend(("--backend", backend))
    if rebuild:
        command.append("--rebuild")
    return command


def _measure_row_in_subprocess(
    *,
    route: str,
    coordinate: str,
    backends: tuple[str, ...],
    artifact_dir: Path,
    rebuild: bool,
    warmup: int,
    repeat: int,
    max_evaluations: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        _worker_command(
            route=route,
            coordinate=coordinate,
            backends=backends,
            artifact_dir=artifact_dir,
            rebuild=rebuild,
            warmup=warmup,
            repeat=repeat,
            max_evaluations=max_evaluations,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker failed"
        raise RuntimeError(f"{route}:{coordinate} worker exited {completed.returncode}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{route}:{coordinate} worker returned invalid JSON: {completed.stdout!r}"
        ) from error
    if completed.stderr.strip():
        payload["row"]["worker_stderr"] = completed.stderr.strip()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--route", action="append", choices=ROUTES)
    parser.add_argument("--coordinate", action="append", choices=COORDINATES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--artifact-dir", type=Path, default=default_kernel_cache_root())
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild selected native artifacts before formal timing",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--worker-row", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.max_evaluations <= 0:
        parser.error("--max-evaluations must be positive")

    backends = _selected(args.backend, BACKENDS)
    routes = _selected(args.route, ROUTES)
    coordinates = _selected(args.coordinate, COORDINATES)
    if not args.no_run and "numba" not in backends:
        parser.error("timed route matrices require --backend numba as the parity baseline")

    artifact_dir = args.artifact_dir.expanduser().resolve()
    if args.worker_row is not None:
        route, coordinate = args.worker_row
        if route not in ROUTES or coordinate not in COORDINATES:
            parser.error("invalid internal worker route/coordinate")
        reference = _make_reference(
            max_evaluations=args.max_evaluations,
            artifact_dir=artifact_dir,
        )
        case = _make_case(
            reference,
            route,
            coordinate,
            max_evaluations=args.max_evaluations,
        )
        row = _measure_case(
            case,
            reference,
            backends=backends,
            artifact_dir=artifact_dir,
            rebuild=args.rebuild,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(json.dumps({"reference": reference.solve, "row": row}, sort_keys=True))
        return 0

    console = Console()
    rows: list[dict[str, Any]] = []
    reference_solve: dict[str, Any] | None = None

    for route in routes:
        for coordinate in coordinates:
            label = f"{route}:{coordinate}"
            if not args.quiet:
                console.print(f"[cyan]running[/] {label}")
            if args.no_run:
                rows.append(_planned_row(route, coordinate, backends))
                continue
            worker = _measure_row_in_subprocess(
                route=route,
                coordinate=coordinate,
                backends=backends,
                artifact_dir=artifact_dir,
                rebuild=args.rebuild,
                warmup=args.warmup,
                repeat=args.repeat,
                max_evaluations=args.max_evaluations,
            )
            if reference_solve is None:
                reference_solve = worker["reference"]
            rows.append(worker["row"])

    summary = _summary(rows, backends)
    payload = {
        "schema": "veqpy.kernel_routes.v1",
        "matrix": {
            "backends": list(backends),
            "routes": list(routes),
            "coordinates": list(coordinates),
            "constraint": "none",
            "source_count": DEFAULT_SOURCE_COUNT,
        },
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "timed_path": "prepared Kernel solve after one Adapter fill",
        "artifact_dir": str(artifact_dir),
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "reference": reference_solve,
        "summary": summary,
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
    _print_table(console, rows)
    if not args.no_write:
        console.print(f"JSON: {args.output.resolve()}")
    console.print(f"summary: {summary}")
    return 1 if summary["failed_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
