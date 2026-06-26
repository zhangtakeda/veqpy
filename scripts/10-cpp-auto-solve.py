"""Future-shaped VEQlib solve call, implemented with today's VEQPy bridge."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

from veqpy.cpp import (
    INITIAL_POLICY_COLD,
    RESIDUAL_NORMALIZATION_FAST,
    SOLVER_METHOD_POWELL,
    VEQlibSolver,
)
from veqpy.model import Boundary, Grid, Topology
from veqpy.model.problem import Problem
from veqpy.operator import Operator

MU0 = 4.0e-7 * np.pi


class Spinner:
    """Tiny terminal spinner that also reports elapsed wall time."""

    _FRAMES = "|/-\\"

    def __init__(self, message: str, *, min_visible_s: float = 0.6) -> None:
        self.message = message
        self.min_visible_s = min_visible_s
        self.elapsed_ms = 0.0
        self._started = 0.0
        self._finished_elapsed_s: float | None = None
        self._last_width = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Spinner:
        self._started = perf_counter()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        elapsed_s = perf_counter() - self._started
        self._finished_elapsed_s = elapsed_s
        self.elapsed_ms = elapsed_s * 1000.0
        if elapsed_s < self.min_visible_s:
            sleep(self.min_visible_s - elapsed_s)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if exc_type is not None:
            self.replace(f"{self.message} failed {self.elapsed_ms / 1000.0:.3f}s")

    def _spin(self) -> None:
        index = 0
        while not self._stop.is_set():
            elapsed = self._finished_elapsed_s
            if elapsed is None:
                elapsed = perf_counter() - self._started
            frame = self._FRAMES[index % len(self._FRAMES)]
            line = f"{self.message} {frame} {elapsed:.1f}s"
            self._last_width = max(self._last_width, len(line))
            self._write(f"\r{line}")
            index += 1
            sleep(0.08)

    def replace(self, line: str) -> None:
        padding = " " * max(0, self._last_width - len(line))
        self._write(f"\r{line}{padding}\n")

    @staticmethod
    def _write(text: str) -> None:
        sys.stderr.write(text)
        sys.stderr.flush()


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Small summary for the artifact build/cache step."""

    artifact_id: str
    elapsed_ms: float
    reused: bool


@dataclass(frozen=True, slots=True)
class PinReport:
    """Describe best-effort CPU affinity applied by this demo."""

    requested: int
    actual: int


@dataclass(frozen=True, slots=True)
class Result:
    """Small Python owner for the C++ solve output and VEQPy snapshot rebuild."""

    operator: Operator
    x: np.ndarray
    build: BuildReport
    success: bool
    elapsed_ms: float
    nfev: int
    raw_norm: float

    def build_equilibrium(self):
        return self.operator.build_equilibrium(self.x)


class Solver:
    """Thin adapter showing the intended high-level C++ solve shape."""

    def solve(
        self,
        topo: dict[str, Any],
        bdry: Boundary,
        profiles: dict[str, Any],
        Ip: float,
    ) -> Result:
        active_profiles = _active_profiles(topo)
        grid = Grid(
            Nr=int(topo["Nr"]),
            Nt=int(topo["Nt"]),
            L_max=_l_max(active_profiles),
            M_max=_m_max(active_profiles),
            K_max=max(2, _m_max(active_profiles)),
            quadrature_scheme=str(topo.get("quadrature", "legendre")),
        )
        problem = Problem(
            route=str(topo.get("route", "PF")),
            coordinate=str(topo.get("coordinate", "psin")),
            nodes=str(topo.get("nodes", "uniform")),
            active_profiles=active_profiles,
            boundary=bdry,
            heat_input=_array(profiles, "heat_input", "heat"),
            current_input=_array(profiles, "current_input", "current"),
            Ip=Ip,
        )
        operator = Operator(grid, problem)
        kernel = VEQlibSolver(_kernel_topology(topo, problem, grid, active_profiles))
        build = _build_kernel(kernel)
        kernel.set_case_json(_payload_json(problem, operator))
        solve = kernel.solve_direct()

        return Result(
            operator=operator,
            x=np.asarray(solve[11], dtype=np.float64).copy(),
            build=build,
            success=bool(solve[1]),
            elapsed_ms=float(solve[0]),
            nfev=int(solve[3]),
            raw_norm=float(solve[9]),
        )


def _active_profiles(topo: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("h", "v", "k", "c0", "psin", "F"):
        count = int(topo.get(name, 0))
        if count > 0:
            out[name] = count
    for prefix in ("c", "s"):
        for key, value in topo.items():
            if isinstance(key, str) and key.startswith(prefix) and key[1:].isdigit():
                count = int(value)
                if count > 0:
                    out[key] = count
    if not out:
        raise ValueError("topo must declare at least one active profile count")
    return out


def _family_counts(active: dict[str, int], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, count in active.items()
        if name.startswith(prefix) and name[1:].isdigit() and int(name[1:]) >= first and count > 0
    ]
    if prefix == "c" and active.get("c0", 0) > 0:
        orders.append(0)
    if not orders:
        return ()
    counts = []
    for order in range(first, max(orders) + 1):
        key = "c0" if prefix == "c" and order == 0 else f"{prefix}{order}"
        counts.append(int(active.get(key, 0)))
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def _l_max(active: dict[str, int]) -> int:
    return max(1, max(active.values()) - 1)


def _m_max(active: dict[str, int]) -> int:
    orders = [1]
    orders.extend(int(name[1:]) for name in active if name[:1] in {"c", "s"} and name[1:].isdigit())
    return max(orders)


def _array(profiles: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in profiles:
            return np.asarray(profiles[name], dtype=np.float64)
    raise KeyError(f"profiles must include one of {names!r}")


def _kernel_topology(
    topo: dict[str, Any],
    problem: Problem,
    grid: Grid,
    active: dict[str, int],
) -> Topology:
    return Topology(
        h_count=int(active.get("h", 0)),
        v_count=int(active.get("v", 0)),
        kappa_count=int(active.get("k", 0)),
        psin_count=int(active.get("psin", 0)),
        F_count=int(active.get("F", 0)),
        c_counts=_family_counts(active, "c", 0),
        s_counts=_family_counts(active, "s", 1),
        Nr=int(grid.Nr),
        Nt=int(grid.Nt),
        route=problem.route,
        coordinate=problem.coordinate,
        constraint=str(topo.get("constraint", "Ip")),
        nodes=problem.nodes,
        sample_count=int(problem.heat_input.size),
        M_max=int(grid.M_max),
        K_max=int(grid.K_max or max(2, grid.M_max)),
    )


def _payload_json(problem: Problem, operator: Operator) -> str:
    source = operator.plan.source_plan
    payload = {
        "case_name": "future_shaped_pf_psin_uniform_ip",
        "boundary": {
            "a": problem.a,
            "R0": problem.R0,
            "Z0": problem.Z0,
            "B0": problem.B0,
            "ka": problem.ka,
            "c_offsets": problem.c_offsets.tolist(),
            "s_offsets": problem.s_offsets.tolist(),
        },
        "source": {
            "scaled_heat": source.scaled_heat.tolist(),
            "scaled_current": source.scaled_current.tolist(),
        },
        "constraints": {
            "scaled_Ip": float(source.scaled_Ip),
            "fix_rho": float(operator.fix_rho),
        },
        "solver": {
            "method_code": SOLVER_METHOD_POWELL,
            "max_residual": 1.0e-6,
            "max_evaluations": operator.x_size**2,
            "accepted_residual_factor": 10.0,
            "accepted_residual_floor": 1.0e-5,
            "initial_policy_code": INITIAL_POLICY_COLD,
            "residual_normalization_code": RESIDUAL_NORMALIZATION_FAST,
            "residual_normalization_floor": 1.0,
            "residual_normalization_max_ratio": 1.0e6,
            "residual_normalization_huber_tau": 3.0,
            "residual_normalization_probe_count": 4,
            "residual_normalization_probe_step": 1.0e-6,
            "residual_normalization_sensitivity_lambda": 0.5,
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _build_kernel(kernel: VEQlibSolver) -> BuildReport:
    with Spinner("kernel   : building") as spinner:
        artifact = kernel.build()
    elapsed_ms = spinner.elapsed_ms
    state = "reused" if artifact.reused else "built"
    spinner.replace(f"kernel   : {state} {artifact.artifact_id}")
    return BuildReport(
        artifact_id=artifact.artifact_id,
        elapsed_ms=elapsed_ms,
        reused=bool(artifact.reused),
    )


def demo_source_profiles() -> dict[str, np.ndarray]:
    psin = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    beta0, alpha_p, alpha_f = 0.75, 5.0, 3.32
    den_p = 1.0 + np.exp(alpha_p) * (alpha_p - 1.0)
    den_f = 1.0 + np.exp(alpha_f) * (alpha_f - 1.0)
    current = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - np.exp(alpha_f)) / den_f
    pressure = beta0 * alpha_p * (np.exp(alpha_p * psin) - np.exp(alpha_p)) / den_p
    return {"heat_input": pressure / MU0, "current_input": current}


def output_path() -> Path:
    outdir = Path(os.environ.get("VEQPY_OUTPUT_DIR", Path("outputs") / "cpp_auto_solve"))
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "cpp_equilibrium.png"


def pin_cpu() -> PinReport | None:
    token = os.environ.get("VEQPY_PIN_CPU", "2").strip().lower()
    if token in {"", "off", "false", "no", "none"}:
        return None
    try:
        requested = int(token)
    except ValueError as exc:
        raise ValueError("VEQPY_PIN_CPU must be an integer CPU id or 'off'") from exc
    if not hasattr(os, "sched_setaffinity"):
        return None

    allowed = os.sched_getaffinity(0)
    actual = requested if requested in allowed else min(allowed)
    os.sched_setaffinity(0, {actual})
    return PinReport(requested=requested, actual=actual)


def main() -> None:
    pin = pin_cpu()
    if pin is not None:
        suffix = "" if pin.actual == pin.requested else f" (requested {pin.requested})"
        print(f"cpu      : pinned {pin.actual}{suffix}", flush=True)

    topo = {
        "route": "PF",
        "coordinate": "psin",
        "nodes": "uniform",
        "constraint": "Ip",
        "h": 3,
        "k": 6,
        "s1": 3,
        "psin": 6,
        "Nr": 32,
        "Nt": 16,
    }
    bdry = Boundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=np.array([0.0, float(np.arcsin(0.5))], dtype=np.float64),
    )
    profiles = demo_source_profiles()

    solver = Solver()
    results = solver.solve(topo, bdry, profiles, Ip=3.0e6)
    equilibrium = results.build_equilibrium()
    png = output_path()
    equilibrium.plot(str(png))

    print(f"success  : {results.success}", flush=True)
    print(f"build    : {results.build.elapsed_ms:.3f} [ms]", flush=True)
    print(f"solve    : {results.elapsed_ms:.3f} [ms]", flush=True)
    print(f"nfev     : {results.nfev}", flush=True)
    print(f"raw_norm : {results.raw_norm:.6e}", flush=True)
    print(f"png      : {png}", flush=True)


if __name__ == "__main__":
    main()
