from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/veqpy-mpl")))

import numpy as np

from veqlib.facade import KernelResult, pinned_cpu
from veqpy.engine.numba_source import source_parameterization_for_route_key
from veqpy.model import Boundary, Geqdsk, Grid, Problem
from veqpy.operator import (
    Operator,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
)
from veqpy.solver import Solver, SolverConfig

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
VEQLIB_ROOT = REPO_ROOT / "veqlib"
CORE_DIR = VEQLIB_ROOT / "core"
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MU0 = 4.0e-7 * np.pi
EPS = 1.0e-14

CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")

CASE_REFERENCE_GFILES = {
    "solovev": str(REPO_ROOT / "data" / "SOLOVEV.geqdsk"),
    "chease": str(REPO_ROOT / "data" / "CHEASE.geqdsk"),
    "efit": str(REPO_ROOT / "data" / "EFIT.geqdsk"),
}
CASE_REFERENCE_EQUILIBRIUM_JSONS = {
    "solovev": str(REPO_ROOT / "data" / "solovev-equilibrium.json"),
    "chease": str(REPO_ROOT / "data" / "chease-equilibrium.json"),
    "efit": str(REPO_ROOT / "data" / "efit-equilibrium.json"),
}
REFERENCE_EQUILIBRIUM_MANIFEST_PATH = str(REPO_ROOT / "data" / "reference_equilibria.json")
REDUCED_EQUILIBRIUM_JSON_TEMPLATE = str(
    REPO_ROOT / "data" / "pareto_reduced_{case_key}_{config_label}.json"
)
REDUCED_EQUILIBRIUM_MANIFEST_PATH = str(REPO_ROOT / "data" / "pareto_reduced_equilibria.json")

CASE_REFERENCE_PROFILE_LENGTHS = {
    "solovev": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "chease": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "efit": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
}

REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
SOLVER_INITIAL_POLICY = "auto"
BOUNDARY_MAXTOL = 1.0
CASE_BOUNDARY_FIT_M = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}
CASE_BOUNDARY_FIT_N = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}

ROUTE_BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
ROUTE_BENCHMARK_COORDINATES = ("rho", "psin")
ROUTE_BENCHMARK_MODE_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "PF": ("null", "Ip", "beta"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}
ROUTE_TEST_SOURCE_SAMPLE_COUNT = 51
ROUTE_SHAPE_MATCH_TOL = 1.0e-2
ROUTE_DIAGNOSTIC_SIGN_CHANGE_WINDOW = 12
ROUTE_TEST_GRID = Grid(Nr=32, Nt=16, quadrature_scheme="legendre")


@dataclass(frozen=True, slots=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True, slots=True)
class PfReferenceCase:
    case_key: str
    boundary: object
    geqdsk: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    psin_interp_axis: PreparedInterpAxis


@dataclass(frozen=True, slots=True)
class RouteBenchmarkSpec:
    mode: str
    coordinate: str
    constraint: str
    input_kind: str = "uniform"

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.input_kind}_{self.constraint}"


@dataclass(frozen=True, slots=True)
class RouteReferenceBundle:
    result: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    reference_shape_x: np.ndarray
    rho_axis: np.ndarray
    psin_axis: np.ndarray
    rho_interp_axis: PreparedInterpAxis
    psin_interp_axis: PreparedInterpAxis
    boundary: Boundary
    profile_coeffs: dict[str, np.ndarray]


@lru_cache(maxsize=1)
def load_veqpy_components() -> dict[str, object]:
    from veqpy.model import Boundary, Equilibrium, Grid, Problem
    from veqpy.model.boundary import _fit_boundary_params
    from veqpy.operator import Operator
    from veqpy.solver import Solver, SolverConfig

    return {
        "Boundary": Boundary,
        "Equilibrium": Equilibrium,
        "Geqdsk": Geqdsk,
        "Grid": Grid,
        "Problem": Problem,
        "fit_boundary_params": _fit_boundary_params,
        "Operator": Operator,
        "Solver": Solver,
        "SolverConfig": SolverConfig,
    }


def temp_cache(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return None


def runtime_env() -> dict[str, str | None]:
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def quantile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    return float(values_sorted[int((len(values_sorted) - 1) * q)])


def float_stats(values: list[float], *, samples: bool = True, prefix: str = "") -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}median_ms": float("nan"),
            f"{prefix}mean_ms": float("nan"),
            f"{prefix}min_ms": float("nan"),
            f"{prefix}max_ms": float("nan"),
            f"{prefix}p05_ms": float("nan"),
            f"{prefix}p95_ms": float("nan"),
            "count": 0,
            **({"samples_ms": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        f"{prefix}median_ms": float(statistics.median(values)),
        f"{prefix}mean_ms": float(statistics.mean(values)),
        f"{prefix}min_ms": float(min(values)),
        f"{prefix}max_ms": float(max(values)),
        f"{prefix}p05_ms": quantile(values, 0.05),
        f"{prefix}p95_ms": quantile(values, 0.95),
        "count": len(values),
    }
    if samples:
        payload["samples_ms"] = [float(value) for value in values]
    return payload


def int_stats(values: list[int], *, samples: bool = True) -> dict[str, Any]:
    if not values:
        return {
            "median": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            **({"samples": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }
    if samples:
        payload["samples"] = [int(value) for value in values]
    return payload


def max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    if lhs_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def profile_count(profile_coeffs: dict[str, Any], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


def family_counts(profile_coeffs: dict[str, Any], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, values in profile_coeffs.items()
        if values is not None
        and len(name) > 1
        and name[0] == prefix
        and name[1:].isdigit()
        and profile_count(profile_coeffs, name) > 0
    ]
    if not orders:
        return ()
    counts = [
        profile_count(profile_coeffs, f"{prefix}{order}")
        for order in range(first, max(orders) + 1)
    ]
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def finite_or_nan(value: Any) -> float:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else float("nan")


def as_float64_array(values: Any, *, copy: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr.copy() if copy else arr


def prepare_interp_axis(axis: np.ndarray) -> PreparedInterpAxis:
    axis_f64 = as_float64_array(axis)
    order = np.argsort(axis_f64)
    axis_sorted = axis_f64[order]
    unique_axis, unique_index = np.unique(axis_sorted, return_index=True)
    return PreparedInterpAxis(unique_axis=unique_axis, order=order, unique_index=unique_index)


def prepare_interp_values(values: np.ndarray, prepared_axis: PreparedInterpAxis) -> np.ndarray:
    values_f64 = as_float64_array(values)
    return values_f64[prepared_axis.order][prepared_axis.unique_index]


def profile_interp(
    axis: np.ndarray | PreparedInterpAxis, values: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = prepare_interp_values(values, prepared_axis)
    x_new = as_float64_array(x_new)
    if unique_axis.size < 2:
        fill_value = float(unique_values[0] if unique_values.size else 0.0)
        return np.full_like(x_new, fill_value, dtype=np.float64)
    if unique_axis.size < 3:
        return np.interp(x_new, unique_axis, unique_values).astype(np.float64, copy=False)
    return as_float64_array(PchipInterpolator(unique_axis, unique_values, extrapolate=True)(x_new))


def active_profiles_from_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, int]:
    active_profiles: dict[str, int] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            length = int(coeff)
        else:
            length = int(np.asarray(coeff, dtype=np.float64).size)
        if length > 0:
            active_profiles[str(name)] = length
    return active_profiles


def safe_divisor(values: Sequence[float], *, floor: float = EPS) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sign = np.where(arr < 0.0, -1.0, 1.0)
    return np.where(np.abs(arr) > floor, arr, sign * floor)


def coefficients_from_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, np.ndarray]:
    coefficients: dict[str, np.ndarray] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            if int(coeff) <= 0:
                continue
            values = np.zeros(int(coeff), dtype=np.float64)
        else:
            values = np.asarray(coeff, dtype=np.float64)
            if values.size <= 0:
                continue
        coefficients[str(name)] = values
    return coefficients


def profile_order_from_active(active_profiles: Mapping[str, int]) -> int:
    order = 1
    for name in active_profiles:
        if len(name) > 1 and name[0] in {"c", "s"} and name[1:].isdigit():
            order = max(order, int(name[1:]))
    return order


def extract_shape_x(
    active_profiles: dict[str, int],
    x: np.ndarray,
    *,
    m_max: int | None = None,
) -> np.ndarray:
    resolved_m_max = max(ROUTE_TEST_GRID.M_max, profile_order_from_active(active_profiles))
    if m_max is not None:
        resolved_m_max = max(resolved_m_max, int(m_max))
    profile_names = build_profile_names(resolved_m_max)
    shape_profile_names = build_shape_profile_names(resolved_m_max)
    profile_index = build_profile_index(profile_names)
    _, coeff_index, _ = build_profile_layout(active_profiles, profile_names=profile_names)
    shape_values: list[float] = []
    for degree in range(coeff_index.shape[1]):
        for name in shape_profile_names:
            idx = int(coeff_index[profile_index[name], degree])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def build_route_reference_profiles(equilibrium: Any) -> dict[str, np.ndarray | float]:
    psin_r = as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = safe_divisor(psin_r)
    psi_r = as_float64_array(equilibrium.alpha2 * psin_r)
    psi_r_safe = safe_divisor(psi_r)

    ffn_r = as_float64_array(equilibrium.FFn_r, copy=True)
    pn_r = as_float64_array(equilibrium.Pn_r, copy=True)
    ff_r = as_float64_array(equilibrium.FF_r, copy=True)
    p_r = as_float64_array(equilibrium.P_r, copy=True)
    itor = as_float64_array(equilibrium.Itor, copy=True)
    jtor = as_float64_array(equilibrium.jtor, copy=True)
    jpara = as_float64_array(equilibrium.jpara, copy=True)
    q = as_float64_array(equilibrium.q, copy=True)
    mu0_p_r = MU0 * p_r
    return {
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_r": ffn_r,
        "Pn_r": pn_r,
        "FFn_psin": ffn_r / psin_r_safe,
        "Pn_psin": pn_r / psin_r_safe,
        "setup_Pn_r": pn_r / MU0,
        "setup_Pn_psin": (pn_r / psin_r_safe) / MU0,
        "FF_r": ff_r,
        "P_r": p_r,
        "mu0_P_r": mu0_p_r,
        "FF_psi": ff_r / psi_r_safe,
        "P_psi": p_r / psi_r_safe,
        "mu0_P_psi": mu0_p_r / psi_r_safe,
        "Itorn": MU0 * itor,
        "Itor": itor,
        "mu0_Itor": MU0 * itor,
        "jtorn": MU0 * jtor,
        "jtor": jtor,
        "mu0_jtor": MU0 * jtor,
        "jparan": MU0 * jpara,
        "jpara": jpara,
        "mu0_jpara": MU0 * jpara,
        "qn": q * 0.1,
        "q": q,
        "scaled_Ip": float(MU0 * equilibrium.Ip),
        "Ip_constraint": float(equilibrium.Ip),
        "beta_constraint": float(equilibrium.beta_t),
    }


def solve_route_reference(
    problem: Problem,
    grid: Grid,
    config: SolverConfig,
    coeffs: Mapping[str, object],
) -> RouteReferenceBundle:
    solver = Solver(operator=Operator(grid, problem.copy()), config=config)
    kwargs: dict[str, Any] = {
        "method": config.method,
        "max_residual": config.max_residual,
        "max_evaluations": config.max_evaluations,
        "initial_policy": config.initial_policy,
        "enable_verbose": False,
        "enable_history": False,
    }
    if config.initial_policy is None:
        kwargs["x0"] = solver.operator.pack_coefficients(coefficients_from_coeffs(coeffs))
    solver.solve(**kwargs)
    if solver.result is None:
        raise RuntimeError("reference solve produced no SolverResult")
    equilibrium = solver.build_equilibrium()
    rho_axis = as_float64_array(equilibrium.rho)
    psin_axis = as_float64_array(equilibrium.psin)
    return RouteReferenceBundle(
        result=solver.result,
        equilibrium=equilibrium,
        ref_profiles=build_route_reference_profiles(equilibrium),
        reference_shape_x=extract_shape_x(solver.operator.problem.active_profiles, solver.result.x),
        rho_axis=rho_axis,
        psin_axis=psin_axis,
        rho_interp_axis=prepare_interp_axis(rho_axis),
        psin_interp_axis=prepare_interp_axis(psin_axis),
        boundary=problem.boundary,
        profile_coeffs=solver.operator.build_coeffs(solver.result.x, include_none=False),
    )


def constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    if constraint == "null":
        return "physical", "physical"
    raise ValueError(f"unsupported constraint: {constraint!r}")


def pressure_keys_for_coordinate(coordinate: str) -> tuple[str, str]:
    if coordinate == "rho":
        return "setup_Pn_r", "P_r"
    if coordinate == "psin":
        return "setup_Pn_psin", "P_psi"
    raise ValueError(f"unsupported coordinate: {coordinate!r}")


def pick_ref_profile(
    ref: dict[str, np.ndarray | float],
    normalized_key: str,
    physical_key: str,
    normalized: bool,
) -> np.ndarray:
    key = normalized_key if normalized else physical_key
    return np.asarray(ref[key], dtype=np.float64)


def build_route_mode_inputs(
    mode: str,
    coordinate: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    pressure_keys = pressure_keys_for_coordinate(coordinate)
    if mode == "PF":
        use_normalized = constraint in {"Ip", "beta"}
        driver_keys = ("FFn_r", "FF_r") if coordinate == "rho" else ("FFn_psin", "FF_psi")
        current_input = pick_ref_profile(ref, driver_keys[0], driver_keys[1], use_normalized)
        heat_input = pick_ref_profile(ref, pressure_keys[0], pressure_keys[1], use_normalized)
        return heat_input, current_input
    if mode == "PP":
        driver_normalized = constraint in {"Ip_beta", "Ip"}
        pressure_normalized = constraint in {"Ip_beta", "beta"}
        current_input = pick_ref_profile(ref, "psin_r", "psi_r", driver_normalized)
        heat_input = pick_ref_profile(ref, pressure_keys[0], pressure_keys[1], pressure_normalized)
        return heat_input, current_input

    driver_domain, pressure_domain = constraint_route_domains(constraint)
    driver_keys = {
        "PI": ("Itor", "Itor"),
        "PJ1": ("jtor", "jtor"),
        "PJ2": ("jpara", "jpara"),
        "PQ": ("qn", "q"),
    }[mode]
    current_input = pick_ref_profile(
        ref,
        driver_keys[0],
        driver_keys[1],
        driver_domain == "normalized",
    )
    heat_input = pick_ref_profile(
        ref,
        pressure_keys[0],
        pressure_keys[1],
        pressure_domain == "normalized",
    )
    return heat_input, current_input


def uniform_route_source_axis(spec: RouteBenchmarkSpec, sample_count: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    route_key = (str(spec.mode).upper(), str(spec.coordinate).lower(), str(spec.input_kind).lower())
    if source_parameterization_for_route_key(route_key) == "sqrt_psin":
        return axis * axis
    return axis


def resample_route_input(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    spec: RouteBenchmarkSpec,
    *,
    sample_count: int,
) -> np.ndarray:
    return profile_interp(source_axis, values, uniform_route_source_axis(spec, sample_count))


def sample_route_input_on_grid(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    grid_axis: np.ndarray,
) -> np.ndarray:
    return profile_interp(source_axis, values, grid_axis)


def make_route_problem(
    spec: RouteBenchmarkSpec,
    reference: RouteReferenceBundle,
    coeffs: Mapping[str, object],
    *,
    grid: Grid,
    sample_count: int,
) -> Problem:
    heat_profile, current_profile = build_route_mode_inputs(
        spec.mode,
        spec.coordinate,
        spec.constraint,
        reference.ref_profiles,
    )
    if spec.input_kind == "grid":
        if spec.coordinate == "rho":
            grid_axis = np.asarray(grid.rho, dtype=np.float64)
            source_axis = reference.rho_interp_axis
        else:
            grid_axis = profile_interp(
                reference.rho_interp_axis,
                reference.psin_axis,
                np.asarray(grid.rho, dtype=np.float64),
            )
            source_axis = reference.psin_interp_axis
        heat_input = sample_route_input_on_grid(heat_profile, source_axis, grid_axis)
        current_input = sample_route_input_on_grid(current_profile, source_axis, grid_axis)
        nodes = "grid"
    else:
        source_axis = (
            reference.rho_interp_axis if spec.coordinate == "rho" else reference.psin_interp_axis
        )
        heat_input = resample_route_input(
            heat_profile,
            source_axis,
            spec,
            sample_count=sample_count,
        )
        current_input = resample_route_input(
            current_profile,
            source_axis,
            spec,
            sample_count=sample_count,
        )
        nodes = "uniform"
    ip = (
        float(reference.ref_profiles["Ip_constraint"])
        if spec.constraint in {"Ip", "Ip_beta"}
        else None
    )
    beta = (
        float(reference.ref_profiles["beta_constraint"])
        if spec.constraint in {"beta", "Ip_beta"}
        else None
    )
    return Problem(
        route=spec.mode,
        active_profiles=active_profiles_from_coeffs(coeffs),
        boundary=reference.boundary,
        heat_input=heat_input,
        current_input=current_input,
        coordinate=spec.coordinate,
        nodes=nodes,
        Ip=ip,
        beta=beta,
    )


def solve_route_case(
    problem: Problem,
    grid: Grid,
    config: SolverConfig,
    coeffs: Mapping[str, object],
    *,
    initial_policy: str | None | object = ...,  # ellipsis means use config/default solve behavior
) -> Solver:
    solver = Solver(operator=Operator(grid, problem.copy()), config=config)
    x0 = solver.operator.pack_coefficients(coefficients_from_coeffs(coeffs))
    kwargs: dict[str, Any] = {
        "x0": x0,
        "method": config.method,
        "max_residual": config.max_residual,
        "max_evaluations": config.max_evaluations,
        "enable_verbose": False,
        "enable_history": False,
    }
    if initial_policy is not ...:
        kwargs["initial_policy"] = initial_policy
        if initial_policy is not None:
            kwargs.pop("x0", None)
    solver.solve(**kwargs)
    if solver.result is None:
        raise RuntimeError("route solve produced no SolverResult")
    return solver


def shape_error(reference_x: np.ndarray, current_x: np.ndarray) -> float:
    n = min(reference_x.shape[0], current_x.shape[0])
    if n == 0:
        return 0.0
    return float(np.max(np.abs(current_x[:n] - reference_x[:n])))


def relative_profile_errors(
    reference_values: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float]:
    reference_values = as_float64_array(reference_values)
    current_values = as_float64_array(current_values)
    n = min(reference_values.shape[0], current_values.shape[0])
    if n == 0:
        return 0.0, 0.0
    reference_values = reference_values[:n]
    current_values = current_values[:n]
    diff = current_values - reference_values
    scale = max(float(np.max(np.abs(reference_values))), 1.0e-12)
    return float(np.sqrt(np.mean(diff * diff)) / scale), float(np.max(np.abs(diff)) / scale)


def window_derivative_sign_changes(
    values: np.ndarray,
    *,
    side: str,
    window: int = ROUTE_DIAGNOSTIC_SIGN_CHANGE_WINDOW,
) -> int:
    values = as_float64_array(values)
    count = min(int(window), values.shape[0])
    if side == "head":
        sample = values[:count]
    elif side == "tail":
        sample = values[-count:]
    else:
        raise ValueError(f"unsupported side {side!r}")
    delta = np.diff(sample)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def diagnostic_profile_metrics(
    reference_axis: PreparedInterpAxis,
    reference_values: np.ndarray,
    current_axis: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float, int, int]:
    current_axis = as_float64_array(current_axis)
    current_values = as_float64_array(current_values)
    reference_on_current = profile_interp(reference_axis, reference_values, current_axis)
    rel_rms, rel_max = relative_profile_errors(reference_on_current, current_values)
    return (
        rel_rms,
        rel_max,
        window_derivative_sign_changes(current_values, side="head"),
        window_derivative_sign_changes(current_values, side="tail"),
    )


def benchmark_route_case_diagnostics(
    reference: RouteReferenceBundle,
    equilibrium: object,
    shape_x: np.ndarray,
) -> dict[str, float | int]:
    psi_r_rel_rms_error, psi_r_rel_max_error, psi_r_head_sign_changes, psi_r_tail_sign_changes = (
        diagnostic_profile_metrics(
            reference.rho_interp_axis,
            np.asarray(reference.ref_profiles["psi_r"], dtype=np.float64),
            equilibrium.rho,
            equilibrium.alpha2 * equilibrium.psin_r,
        )
    )
    (
        ff_psi_rel_rms_error,
        ff_psi_rel_max_error,
        ff_psi_head_sign_changes,
        ff_psi_tail_sign_changes,
    ) = (
        diagnostic_profile_metrics(
            reference.rho_interp_axis,
            np.asarray(reference.ref_profiles["FF_psi"], dtype=np.float64),
            equilibrium.rho,
            equilibrium.alpha1 * equilibrium.FFn_psin,
        )
    )
    (
        mu0_p_psi_rel_rms_error,
        mu0_p_psi_rel_max_error,
        mu0_p_psi_head_sign_changes,
        mu0_p_psi_tail_sign_changes,
    ) = diagnostic_profile_metrics(
        reference.rho_interp_axis,
        np.asarray(reference.ref_profiles["mu0_P_psi"], dtype=np.float64),
        equilibrium.rho,
        equilibrium.alpha1 * equilibrium.Pn_psin,
    )
    return {
        "shape_error": shape_error(reference.reference_shape_x, shape_x),
        "psi_r_rel_rms_error": psi_r_rel_rms_error,
        "psi_r_rel_max_error": psi_r_rel_max_error,
        "psi_r_head_sign_changes": psi_r_head_sign_changes,
        "psi_r_tail_sign_changes": psi_r_tail_sign_changes,
        "ff_psi_rel_rms_error": ff_psi_rel_rms_error,
        "ff_psi_rel_max_error": ff_psi_rel_max_error,
        "ff_psi_head_sign_changes": ff_psi_head_sign_changes,
        "ff_psi_tail_sign_changes": ff_psi_tail_sign_changes,
        "mu0_p_psi_rel_rms_error": mu0_p_psi_rel_rms_error,
        "mu0_p_psi_rel_max_error": mu0_p_psi_rel_max_error,
        "mu0_p_psi_head_sign_changes": mu0_p_psi_head_sign_changes,
        "mu0_p_psi_tail_sign_changes": mu0_p_psi_tail_sign_changes,
    }


def iter_route_specs(
    *,
    scope: str,
    default_scope: str = "ip-uniform",
    allow_grid: bool = True,
) -> tuple[RouteBenchmarkSpec, ...]:
    if scope == default_scope:
        input_kinds = ("uniform",)
        constraints_by_mode = {mode: ("Ip",) for mode in ROUTE_BENCHMARK_MODES}
    elif scope in {"uniform", "full"}:
        input_kinds = ["uniform"]
        if scope == "full" and allow_grid:
            input_kinds.append("grid")
        constraints_by_mode = ROUTE_BENCHMARK_MODE_CONSTRAINTS
    else:
        raise ValueError(f"unknown route benchmark scope {scope!r}")
    return tuple(
        RouteBenchmarkSpec(
            mode=mode,
            coordinate=coordinate,
            constraint=constraint,
            input_kind=input_kind,
        )
        for mode in ROUTE_BENCHMARK_MODES
        for coordinate in ROUTE_BENCHMARK_COORDINATES
        for input_kind in input_kinds
        for constraint in constraints_by_mode[mode]
    )


def route_spec_label(spec: RouteBenchmarkSpec) -> str:
    return spec.case_name


def route_spec_selector(spec: RouteBenchmarkSpec) -> str:
    return f"{spec.mode}:{spec.coordinate}:{spec.input_kind}:{spec.constraint}"


def filter_route_specs(
    specs: tuple[RouteBenchmarkSpec, ...], selected: set[str] | None
) -> tuple[RouteBenchmarkSpec, ...]:
    if selected is None:
        return specs
    selected_lower = {item.lower() for item in selected}
    retained = tuple(
        spec
        for spec in specs
        if route_spec_label(spec).lower() in selected_lower
        or route_spec_selector(spec).lower() in selected_lower
    )
    matched = {route_spec_label(spec).lower() for spec in retained}
    matched.update(route_spec_selector(spec).lower() for spec in retained)
    missing = selected_lower.difference(matched)
    if missing:
        raise ValueError(f"unknown case selector(s): {', '.join(sorted(missing))}")
    return retained


def summarize_runtime_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "runtime_passed": 0,
        "runtime_failed": 0,
        "runtime_not_requested": 0,
    }
    for row in rows:
        status = row["runtime"]["status"]
        if status == "passed":
            summary["runtime_passed"] += 1
        elif status == "failed":
            summary["runtime_failed"] += 1
        elif status == "not_requested":
            summary["runtime_not_requested"] += 1
    return summary


def runtime_engine_payload(runtime: dict[str, Any], engine_label: str) -> dict[str, Any] | None:
    engines = runtime.get("engines")
    if not isinstance(engines, dict):
        return None
    payload = engines.get(engine_label)
    return payload if isinstance(payload, dict) else None


def timing_median_ms(engine: dict[str, Any] | None) -> float:
    if engine is None:
        return float("nan")
    timing = engine.get("timing")
    if not isinstance(timing, dict):
        return float("nan")
    return float(timing.get("median_ms", float("nan")))


def nfev_median(engine: dict[str, Any] | None) -> str:
    if engine is None:
        return "n/a"
    nfev = engine.get("nfev")
    if not isinstance(nfev, dict):
        return "n/a"
    return str(nfev.get("median", "n/a"))


def format_optional_float(value: float, *, decimals: int = 6) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{decimals}f}"


def format_optional_sci(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(parsed) else f"{parsed:.2e}"


def runtime_status_cell(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]not requested[/]"
    return text


def runtime_progress_phase(status: object) -> str:
    text = str(status)
    if text == "passed":
        return "[green]passed[/]"
    if text == "failed":
        return "[red]failed[/]"
    if text == "not_requested":
        return "[blue]skip[/]"
    return "[dim]done[/]"


def grid_payload(grid: Grid) -> dict[str, Any]:
    return {
        "Nr": int(grid.Nr),
        "Nt": int(grid.Nt),
        "L_max": int(grid.L_max),
        "M_max": int(grid.M_max),
        "K_max": None if grid.K_max is None else int(grid.K_max),
        "quadrature_scheme": str(grid.quadrature_scheme),
        "calculus_scheme": str(grid.calculus_scheme),
    }


def read_geqdsk(path: str):
    geqdsk = load_veqpy_components()["Geqdsk"]()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def load_equilibrium_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing equilibrium JSON: {path}")
    return load_veqpy_components()["Equilibrium"].load(path)


def build_pf_reference_profiles(equilibrium: Any) -> dict[str, np.ndarray | float]:
    psin_r = as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1.0e-14, psin_r, 1.0e-14)
    pn_psin = as_float64_array(equilibrium.Pn_r, copy=True) / psin_r_safe
    return {
        "psin": as_float64_array(equilibrium.psin, copy=True),
        "FFn_psin": as_float64_array(equilibrium.FFn_r, copy=True) / psin_r_safe,
        "Pn_psin": pn_psin,
        "setup_Pn_psin": pn_psin / MU0,
    }


def build_geqdsk_boundary(geqdsk: Any, *, fit_m: int, fit_n: int, return_fit: bool = False):
    components = load_veqpy_components()
    fit = components["fit_boundary_params"](
        geqdsk,
        M=int(fit_m),
        N=int(fit_n),
        maxtol=BOUNDARY_MAXTOL,
        R0=None,
        Z0=None,
        a=None,
        ka=None,
    )
    boundary = components["Boundary"](
        a=float(fit["a"]),
        R0=float(fit["R0"]),
        Z0=float(fit["Z0"]),
        B0=float(geqdsk.Bt0),
        ka=float(fit["ka"]),
        c_offsets=np.asarray(fit["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(fit["s_offsets"], dtype=np.float64),
    )
    return (boundary, fit) if return_fit else boundary


def load_pf_benchmark(backend: str):
    os.environ["VEQPY_BACKEND"] = str(backend)
    components = load_veqpy_components()
    reference_grid = components["Grid"](
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        quadrature_scheme="legendre",
    )
    config = components["SolverConfig"](
        method="hybr",
        max_evaluations=REFERENCE_SOLVER_MAXFEV,
        initial_policy=SOLVER_INITIAL_POLICY,
        enable_verbose=False,
        enable_fallback=False,
        enable_history=False,
    )
    return SimpleNamespace(
        Grid=components["Grid"],
        Operator=components["Operator"],
        Problem=components["Problem"],
        Solver=components["Solver"],
        CONFIG=config,
        REFERENCE_GRID=reference_grid,
    )


def build_pf_reference_case(case_key: str) -> PfReferenceCase:
    equilibrium = load_equilibrium_json(CASE_REFERENCE_EQUILIBRIUM_JSONS[case_key])
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    boundary = build_geqdsk_boundary(
        geqdsk,
        fit_m=CASE_BOUNDARY_FIT_M[case_key],
        fit_n=CASE_BOUNDARY_FIT_N[case_key],
    )
    return PfReferenceCase(
        case_key=case_key,
        boundary=boundary,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        ref_profiles=build_pf_reference_profiles(equilibrium),
        psin_interp_axis=prepare_interp_axis(np.asarray(equilibrium.psin, dtype=np.float64)),
    )


def make_profile_coeffs(
    signature: dict[str, int],
    *,
    max_lengths: dict[str, int],
) -> dict[str, list[float] | None]:
    profile_coeffs: dict[str, list[float] | None] = {name: None for name in max_lengths}
    for name, length in signature.items():
        coeff_length = int(length)
        if coeff_length > 0:
            profile_coeffs[name] = [0.0] * coeff_length
    return profile_coeffs


def build_pf_case(benchmark: Any, reference: PfReferenceCase, signature: dict[str, int]):
    return benchmark.Problem(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        active_profiles=active_profiles_from_coeffs(
            make_profile_coeffs(
                signature,
                max_lengths=CASE_REFERENCE_PROFILE_LENGTHS[reference.case_key],
            )
        ),
        boundary=reference.boundary,
        heat_input=np.asarray(reference.geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(reference.geqdsk.FF_psi, dtype=np.float64),
        Ip=float(reference.geqdsk.Ip),
        beta=None,
    )


def reduced_equilibrium_json_path(case_key: str, config_label: str) -> str:
    return REDUCED_EQUILIBRIUM_JSON_TEMPLATE.format(
        case_key=str(case_key),
        config_label=str(config_label).lower(),
    )


def load_reduced_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REDUCED_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reduced-equilibrium manifest: {path}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def load_reference_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REFERENCE_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reference-equilibrium manifest: {path}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]],
    case_key: str,
    config_label: str,
) -> dict[str, object]:
    entry = manifest.get((case_key, config_label))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} {config_label} "
            f"entry in {REDUCED_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    return entry


def reference_manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]], case_key: str
) -> dict[str, object]:
    entry = manifest.get((case_key, "Ref"))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} Ref entry in "
            f"{REFERENCE_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    return entry


def normalize_signature(signature: dict[str, int]) -> dict[str, int]:
    return {str(name): int(length) for name, length in sorted(signature.items()) if int(length) > 0}


def signature_from_metadata(entry: dict[str, object]) -> dict[str, int]:
    signature = entry.get("signature", {})
    return normalize_signature(signature) if isinstance(signature, dict) else {}


@dataclass(frozen=True, slots=True)
class NativeTiming:
    result: KernelResult
    wall_ms: list[float]
    inner_ms: list[float]
    success: list[bool]
    nfev: list[int]
    njev: list[int]
    jacobian_component_evaluations: list[int]

    def compact(self) -> dict[str, Any]:
        return {
            "success_all": all(self.success),
            "info": int(self.result.info),
            "timing": float_stats(self.wall_ms),
            "inner_timing": float_stats(self.inner_ms),
            "nfev": int_stats(self.nfev),
            "njev": int_stats(self.njev),
            "jacobian_component_evaluations": int_stats(self.jacobian_component_evaluations),
            "raw_norm": float(self.result.raw_norm),
            "scaled_norm": float(self.result.scaled_norm),
            "x": self.result.x.tolist(),
            "raw": self.result.raw.tolist(),
            "alpha": self.result.alpha.tolist(),
        }


def measure_native_solver(
    solver: Any,
    configure_runtime: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
) -> NativeTiming:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    with pinned_cpu():
        configure_runtime()
        for _ in range(warmup):
            solver.solve_direct()

        wall_ms: list[float] = []
        inner_ms: list[float] = []
        success: list[bool] = []
        nfev: list[int] = []
        njev: list[int] = []
        jaccomp: list[int] = []
        final_result: Any | None = None
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            final_result = solver.solve_direct()
            wall_ms.append(float(time.perf_counter_ns() - start_ns) / 1.0e6)
            inner_ms.append(float(final_result[0]))
            success.append(bool(final_result[1]))
            nfev.append(int(final_result[3]))
            njev.append(int(final_result[4]))
            jaccomp.append(int(final_result[6]))

    if final_result is None:
        raise RuntimeError("native timing loop did not run")
    return NativeTiming(
        result=KernelResult.from_solve_direct(final_result),
        wall_ms=wall_ms,
        inner_ms=inner_ms,
        success=success,
        nfev=nfev,
        njev=njev,
        jacobian_component_evaluations=jaccomp,
    )
