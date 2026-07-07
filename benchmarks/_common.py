from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/veqpy-mpl")))

from veqpy import (  # noqa: E402
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)
from veqpy.kernels.numba_kernel.packed_layout import (  # noqa: E402
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
)
from veqpy.model import Geqdsk, Grid  # noqa: E402

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
CORE_DIR = REPO_ROOT / "veqpy" / "kernels" / "cxx_kernel" / "core"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

MU0 = 4.0e-7 * np.pi

ROUTE_BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
ROUTE_BENCHMARK_COORDINATES = ("rho", "psin")
ROUTE_BENCHMARK_NODES = ("uniform", "grid")
ROUTE_BENCHMARK_CONSTRAINTS = {
    "PF": ("Ip", "beta", "null"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}
DEFAULT_ROUTE_SCOPE = "ip-uniform"
DEFAULT_ROUTE_SAMPLE_COUNT = 51
DEFAULT_ROUTE_NR = 32
DEFAULT_ROUTE_NT = 16
DEFAULT_ROUTE_L_MAX = 20
DEFAULT_ROUTE_M_MAX = 20
DEFAULT_ROUTE_K_MAX = 20
SYNTHETIC_SOLVER_METHOD = "powell"
SYNTHETIC_SOLVER_LABEL = "numba-hybr"
SYNTHETIC_SOLVER_MAX_RESIDUAL = 1.0e-6
SYNTHETIC_SOLVER_MAX_EVALUATIONS = 1000
SYNTHETIC_ROUTE_SIGNATURE = {"h": 3, "k": 6, "s1": 3}
SYNTHETIC_PSIN_ROUTE_SIGNATURE = {**SYNTHETIC_ROUTE_SIGNATURE, "psin": 6}
ROUTE_SHAPE_MATCH_TOL = 1.0e-2
ROUTE_DIAGNOSTIC_SIGN_CHANGE_WINDOW = 8

CASE_KEYS = ("solovev", "chease", "efit")
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
CASE_REFERENCE_GFILES = {
    "solovev": REPO_ROOT / "data" / "SOLOVEV.geqdsk",
    "chease": REPO_ROOT / "data" / "CHEASE.geqdsk",
    "efit": REPO_ROOT / "data" / "EFIT.geqdsk",
}
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
REFERENCE_EQUILIBRIUM_MANIFEST_PATH = REPO_ROOT / "data" / "reference_equilibria.json"
REDUCED_EQUILIBRIUM_MANIFEST_PATH = REPO_ROOT / "data" / "pareto_reduced_equilibria.json"
REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
GEQDSK_ROUTE_PROFILE_SIGNATURE = {
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
}


def default_kernel_cache_root() -> Path:
    override = os.environ.get("VEQPY_KERNEL_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.cwd() / ".veqpy-kernel-cache"


@dataclass(frozen=True, slots=True)
class RouteBenchmarkSpec:
    mode: str
    coordinate: str
    nodes: str
    constraint: str

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.nodes}_{self.constraint}"

    @property
    def input_kind(self) -> str:
        return self.nodes


@dataclass(frozen=True, slots=True)
class KernelCase:
    name: str
    topology: KernelTopology
    boundary: KernelBoundary
    source: KernelSource
    config: KernelConfig


@dataclass(frozen=True, slots=True)
class RouteReference:
    ref_profiles: dict[str, np.ndarray | float]
    rho_axis: np.ndarray
    psin_axis: np.ndarray
    reference_shape_x: np.ndarray


def runtime_env() -> dict[str, Any]:
    return {
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def runtime_platform_payload() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def cpu_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(cpu) for cpu in getter(0))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def float_stats(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples_ms": [],
            "count": 0,
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p05_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    sorted_values = sorted(samples)
    return {
        "samples_ms": samples,
        "count": len(samples),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "p05_ms": float(np.percentile(sorted_values, 5)),
        "p95_ms": float(np.percentile(sorted_values, 95)),
    }


def int_stats(values: Sequence[int]) -> dict[str, Any]:
    samples = [int(value) for value in values]
    if not samples:
        return {"samples": [], "count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0}
    return {
        "samples": samples,
        "count": len(samples),
        "min": int(min(samples)),
        "max": int(max(samples)),
        "mean": float(statistics.fmean(samples)),
        "median": int(statistics.median(samples)),
    }


def format_float(value: float, *, precision: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.{precision}g}"


def constraint_flags(label: str) -> tuple[bool, bool]:
    normalized = str(label)
    if normalized == "null":
        return False, False
    if normalized == "Ip":
        return True, False
    if normalized == "beta":
        return False, True
    if normalized == "Ip_beta":
        return True, True
    raise ValueError(f"unknown source constraint {label!r}")


def iter_route_specs(
    scope: str = DEFAULT_ROUTE_SCOPE,
    *,
    default_scope: str = DEFAULT_ROUTE_SCOPE,
    allow_grid: bool = True,
) -> tuple[RouteBenchmarkSpec, ...]:
    if scope not in {default_scope, "uniform", "full"}:
        raise ValueError(f"scope must be {default_scope}, uniform, or full")
    node_choices = ("uniform",)
    if allow_grid and scope == "full":
        node_choices = ("uniform", "grid")
    specs: list[RouteBenchmarkSpec] = []
    for mode in ROUTE_BENCHMARK_MODES:
        constraints = ROUTE_BENCHMARK_CONSTRAINTS[mode]
        for coordinate in ROUTE_BENCHMARK_COORDINATES:
            for nodes in node_choices:
                for constraint in constraints:
                    if scope == default_scope and constraint != "Ip":
                        continue
                    specs.append(RouteBenchmarkSpec(mode, coordinate, nodes, constraint))
    return tuple(specs)


def _route_spec_predicate(value: str) -> Callable[[RouteBenchmarkSpec], bool]:
    raw = str(value)
    parts = raw.split(":")
    if len(parts) == 1:
        token = parts[0].lower()
        return lambda spec: (
            spec.case_name.lower() == token
            or route_spec_selector(spec).lower() == token
            or spec.mode.lower() == token
        )
    if len(parts) != 4:
        raise ValueError("case selectors must be name or route:coordinate:nodes:constraint")
    route, coordinate, nodes, constraint = parts
    return lambda spec: (
        spec.mode == route.upper()
        and spec.coordinate == coordinate.lower()
        and spec.nodes == nodes.lower()
        and spec.constraint == constraint
    )


def filter_route_specs(
    specs: Iterable[RouteBenchmarkSpec],
    selectors: Sequence[str] | None,
) -> tuple[RouteBenchmarkSpec, ...]:
    specs_tuple = tuple(specs)
    if not selectors:
        return specs_tuple
    predicates = tuple(_route_spec_predicate(selector) for selector in selectors)
    retained = tuple(
        spec for spec in specs_tuple if any(predicate(spec) for predicate in predicates)
    )
    if not retained:
        raise ValueError(f"unknown case selector(s): {', '.join(selectors)}")
    return retained


def route_spec_label(spec: RouteBenchmarkSpec) -> str:
    return spec.case_name


def route_spec_selector(spec: RouteBenchmarkSpec) -> str:
    return f"{spec.mode}:{spec.coordinate}:{spec.nodes}:{spec.constraint}"


def profile_counts_for_route(
    route: str,
    coordinate: str,
    nodes: str,
    *,
    h_count: int = 3,
    v_count: int = 0,
    kappa_count: int = 6,
    c_counts: tuple[int, ...] = (),
    s_counts: tuple[int, ...] = (3,),
    active_count: int = 6,
) -> dict[str, Any]:
    psin_count = 0
    f_count = 0
    if route == "PJ2":
        f_count = active_count
    elif coordinate == "psin" and nodes == "uniform":
        psin_count = active_count
    return {
        "h_count": h_count,
        "v_count": v_count,
        "kappa_count": kappa_count,
        "psin_count": psin_count,
        "F_count": f_count,
        "c_counts": c_counts,
        "s_counts": s_counts,
    }


def profile_counts_from_signature(
    signature: dict[str, int],
    *,
    route: str,
    coordinate: str,
    nodes: str,
    pj2_f_count: int = 6,
) -> dict[str, Any]:
    psin_count = int(signature.get("psin", 0))
    f_count = int(signature.get("F", 0))
    if route == "PJ2" and f_count <= 0:
        f_count = int(pj2_f_count)
    if route == "PJ2":
        psin_count = 0
    elif not (coordinate == "psin" and nodes == "uniform"):
        psin_count = 0
    c_counts = _family_counts(signature, "c", start=0)
    s_counts = _family_counts(signature, "s", start=1)
    return {
        "h_count": int(signature.get("h", 0)),
        "v_count": int(signature.get("v", 0)),
        "kappa_count": int(signature.get("k", 0)),
        "psin_count": psin_count,
        "F_count": f_count,
        "c_counts": c_counts,
        "s_counts": s_counts,
    }


def _family_counts(signature: dict[str, int], prefix: str, *, start: int) -> tuple[int, ...]:
    values: list[int] = []
    order = start
    while True:
        key = f"{prefix}{order}"
        if key not in signature:
            break
        values.append(int(signature[key]))
        order += 1
    while values and values[-1] == 0:
        values.pop()
    return tuple(values)


def grid_payload(*, nr: int, nt: int, l_max: int, m_max: int, k_max: int | None) -> dict[str, Any]:
    return {
        "Nr": int(nr),
        "Nt": int(nt),
        "L_max": int(l_max),
        "M_max": int(m_max),
        "K_max": None if k_max is None else int(k_max),
        "quadrature_scheme": "legendre",
        "calculus_scheme": "spectral",
    }


def synthetic_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=(float(np.arcsin(0.5)),),
    )


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current.astype(np.float64), heat.astype(np.float64)


@lru_cache(maxsize=1)
def synthetic_route_reference() -> RouteReference:
    rho_src = np.linspace(0.0, 1.0, DEFAULT_ROUTE_SAMPLE_COUNT, dtype=np.float64)
    psin_src = rho_src * rho_src
    ffn_psin_src, pn_psin_src = pf_reference_profiles(psin_src)
    ffn_r_src = ffn_psin_src * (2.0 * rho_src)
    pn_r_src = pn_psin_src * (2.0 * rho_src)
    topology = KernelTopology(
        **profile_counts_from_signature(
            SYNTHETIC_ROUTE_SIGNATURE,
            route="PF",
            coordinate="rho",
            nodes="uniform",
        ),
        Nr=64,
        Nt=32,
        route="PF",
        coordinate="rho",
        nodes="uniform",
        ip_constraint=True,
        sample_count=DEFAULT_ROUTE_SAMPLE_COUNT,
        M_max=DEFAULT_ROUTE_M_MAX,
        K_max=DEFAULT_ROUTE_K_MAX,
    )
    source = KernelSource(
        heat_profile=pn_r_src / MU0,
        current_profile=ffn_r_src,
        Ip=3.0e6,
        beta=np.nan,
        case_name="synthetic-reference",
    )
    config = KernelConfig(
        method=SYNTHETIC_SOLVER_METHOD,
        max_residual=SYNTHETIC_SOLVER_MAX_RESIDUAL,
        max_evaluations=SYNTHETIC_SOLVER_MAX_EVALUATIONS,
        initial="cold",
        continuation="cold",
        norm="fast",
    )
    result, kernel = solve_numba_case(
        KernelCase("synthetic_reference", topology, synthetic_boundary(), source, config)
    )
    if not result.success:
        kernel.close()
        raise RuntimeError("synthetic route reference solve failed")
    try:
        equilibrium = kernel.build_equilibrium()
        rho_axis = np.asarray(equilibrium.rho, dtype=np.float64)
        psin_axis = np.asarray(equilibrium.psin, dtype=np.float64)
        return RouteReference(
            ref_profiles=build_route_reference_profiles(equilibrium),
            rho_axis=rho_axis,
            psin_axis=psin_axis,
            reference_shape_x=extract_shape_x(topology, result.x),
        )
    finally:
        kernel.close()


def build_route_reference_profiles(equilibrium: Any) -> dict[str, np.ndarray | float]:
    psin_r = np.asarray(equilibrium.psin_r, dtype=np.float64)
    psin_r_safe = safe_divisor(psin_r)
    psi_r = np.asarray(equilibrium.alpha2 * psin_r, dtype=np.float64)
    psi_r_safe = safe_divisor(psi_r)

    ffn_r = np.asarray(equilibrium.FFn_r, dtype=np.float64)
    pn_r = np.asarray(equilibrium.Pn_r, dtype=np.float64)
    ff_r = np.asarray(equilibrium.FF_r, dtype=np.float64)
    p_r = np.asarray(equilibrium.P_r, dtype=np.float64)
    itor = np.asarray(equilibrium.Itor, dtype=np.float64)
    jtor = np.asarray(equilibrium.jtor, dtype=np.float64)
    jpara = np.asarray(equilibrium.jpara, dtype=np.float64)
    q = np.asarray(equilibrium.q, dtype=np.float64)
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
        "Ip_constraint": float(equilibrium.Ip),
        "beta_constraint": float(equilibrium.beta_t),
    }


def safe_divisor(values: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    signs = np.where(arr < 0.0, -1.0, 1.0)
    return np.where(np.abs(arr) < floor, signs * floor, arr)


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


def source_profiles_for_route(
    spec: RouteBenchmarkSpec,
    *,
    nr: int,
    nt: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = synthetic_route_reference()
    heat_profile, current_profile = build_route_mode_inputs(
        spec.mode,
        spec.coordinate,
        spec.constraint,
        reference.ref_profiles,
    )
    if spec.nodes == "grid":
        grid = Grid(Nr=int(nr), Nt=int(nt), quadrature_scheme="legendre")
        if spec.coordinate == "rho":
            target_axis = np.asarray(grid.rho, dtype=np.float64)
            source_axis = reference.rho_axis
        else:
            target_axis = profile_interp(reference.rho_axis, reference.psin_axis, grid.rho)
            source_axis = reference.psin_axis
    else:
        target_axis = uniform_route_source_axis(spec, sample_count)
        source_axis = reference.rho_axis if spec.coordinate == "rho" else reference.psin_axis
    return (
        profile_interp(source_axis, heat_profile, target_axis).astype(np.float64),
        profile_interp(source_axis, current_profile, target_axis).astype(np.float64),
    )


def uniform_route_source_axis(spec: RouteBenchmarkSpec, sample_count: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    if spec.mode == "PP" and spec.coordinate == "psin" and spec.nodes == "uniform":
        return axis * axis
    return axis


def profile_interp(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    source = np.asarray(source_axis, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    order = np.argsort(source)
    sorted_axis, unique_index = np.unique(source[order], return_index=True)
    sorted_values = vals[order][unique_index]
    if sorted_axis.size < 2:
        fill_value = float(sorted_values[0] if sorted_values.size else 0.0)
        return np.full_like(target, fill_value, dtype=np.float64)
    if sorted_axis.size < 3:
        return np.interp(target, sorted_axis, sorted_values).astype(np.float64, copy=False)
    return np.asarray(PchipInterpolator(sorted_axis, sorted_values, extrapolate=True)(target))


def active_profiles_from_topology(topology: KernelTopology) -> dict[str, int]:
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


def extract_shape_x(topology: KernelTopology, x: np.ndarray) -> np.ndarray:
    active_profiles = active_profiles_from_topology(topology)
    profile_names = build_profile_names(topology.M_max)
    shape_profile_names = build_shape_profile_names(topology.M_max)
    profile_index = build_profile_index(profile_names)
    _, coeff_index, _ = build_profile_layout(active_profiles, profile_names=profile_names)
    shape_values: list[float] = []
    for degree in range(coeff_index.shape[1]):
        for name in shape_profile_names:
            idx = int(coeff_index[profile_index[name], degree])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def shape_error(reference_x: np.ndarray, current_x: np.ndarray) -> float:
    n = min(reference_x.shape[0], current_x.shape[0])
    if n == 0:
        return 0.0
    return float(np.max(np.abs(reference_x[:n] - current_x[:n])))


def relative_profile_errors(
    reference_values: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float]:
    ref = np.asarray(reference_values, dtype=np.float64)
    cur = np.asarray(current_values, dtype=np.float64)
    n = min(ref.shape[0], cur.shape[0])
    if n == 0:
        return 0.0, 0.0
    ref = ref[:n]
    cur = cur[:n]
    diff = cur - ref
    scale = max(float(np.max(np.abs(ref))), 1.0e-12)
    return float(np.sqrt(np.mean(diff * diff)) / scale), float(np.max(np.abs(diff)) / scale)


def window_derivative_sign_changes(
    values: np.ndarray,
    *,
    side: str,
    window: int = ROUTE_DIAGNOSTIC_SIGN_CHANGE_WINDOW,
) -> int:
    arr = np.asarray(values, dtype=np.float64)
    count = min(int(window), arr.shape[0])
    if side == "head":
        sample = arr[:count]
    elif side == "tail":
        sample = arr[-count:]
    else:
        raise ValueError(f"unsupported side {side!r}")
    signs = np.sign(np.diff(sample))
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def diagnostic_profile_metrics(
    reference_axis: np.ndarray,
    reference_values: np.ndarray,
    current_axis: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float, int, int]:
    reference_on_current = profile_interp(reference_axis, reference_values, current_axis)
    rel_rms, rel_max = relative_profile_errors(reference_on_current, current_values)
    return (
        rel_rms,
        rel_max,
        window_derivative_sign_changes(current_values, side="head"),
        window_derivative_sign_changes(current_values, side="tail"),
    )


def benchmark_route_case_diagnostics(
    reference: RouteReference,
    equilibrium: Any,
    shape_x: np.ndarray,
) -> dict[str, float | int]:
    psi_r_rel_rms_error, psi_r_rel_max_error, psi_r_head_changes, psi_r_tail_changes = (
        diagnostic_profile_metrics(
            reference.rho_axis,
            np.asarray(reference.ref_profiles["psi_r"], dtype=np.float64),
            equilibrium.rho,
            equilibrium.alpha2 * equilibrium.psin_r,
        )
    )
    ff_rel_rms_error, ff_rel_max_error, ff_head_changes, ff_tail_changes = (
        diagnostic_profile_metrics(
            reference.rho_axis,
            np.asarray(reference.ref_profiles["FF_psi"], dtype=np.float64),
            equilibrium.rho,
            equilibrium.alpha1 * equilibrium.FFn_psin,
        )
    )
    mu0_p_rel_rms_error, mu0_p_rel_max_error, mu0_p_head_changes, mu0_p_tail_changes = (
        diagnostic_profile_metrics(
            reference.rho_axis,
            np.asarray(reference.ref_profiles["mu0_P_psi"], dtype=np.float64),
            equilibrium.rho,
            equilibrium.alpha1 * equilibrium.Pn_psin,
        )
    )
    return {
        "shape_error": shape_error(reference.reference_shape_x, shape_x),
        "shape_match_tol": ROUTE_SHAPE_MATCH_TOL,
        "psi_r_rel_rms_error": psi_r_rel_rms_error,
        "psi_r_rel_max_error": psi_r_rel_max_error,
        "psi_r_head_sign_changes": psi_r_head_changes,
        "psi_r_tail_sign_changes": psi_r_tail_changes,
        "ff_psi_rel_rms_error": ff_rel_rms_error,
        "ff_psi_rel_max_error": ff_rel_max_error,
        "ff_psi_head_sign_changes": ff_head_changes,
        "ff_psi_tail_sign_changes": ff_tail_changes,
        "mu0_p_psi_rel_rms_error": mu0_p_rel_rms_error,
        "mu0_p_psi_rel_max_error": mu0_p_rel_max_error,
        "mu0_p_psi_head_sign_changes": mu0_p_head_changes,
        "mu0_p_psi_tail_sign_changes": mu0_p_tail_changes,
    }


def source_profiles_from_geqdsk(
    geqdsk: Geqdsk,
    *,
    route: str,
    coordinate: str,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    source_axis = axis if coordinate == "psin" else axis * axis
    geqdsk_axis = np.linspace(0.0, 1.0, max(int(geqdsk.P_psi.size), 2), dtype=np.float64)
    p_psi = _finite_or_default_profile(geqdsk.P_psi, geqdsk_axis.size, scale=1.0e6)
    ff_psi = _finite_or_default_profile(geqdsk.FF_psi, geqdsk_axis.size, scale=1.0)
    heat_profile = np.interp(source_axis, geqdsk_axis, p_psi)
    current_profile = np.interp(source_axis, geqdsk_axis, ff_psi)
    if coordinate == "rho":
        heat_profile = heat_profile * (2.0 * np.maximum(axis, 1.0e-12))
        current_profile = current_profile * (2.0 * np.maximum(axis, 1.0e-12))
    if route in {"PI", "PJ1", "PJ2"}:
        current_profile = current_profile / MU0
    return heat_profile.astype(np.float64), current_profile.astype(np.float64)


def _finite_or_default_profile(values: np.ndarray, size: int, *, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and arr.size == size and np.all(np.isfinite(arr)):
        return arr
    axis = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    return scale * (1.0 - axis)


def route_kernel_case(
    spec: RouteBenchmarkSpec,
    *,
    nr: int = DEFAULT_ROUTE_NR,
    nt: int = DEFAULT_ROUTE_NT,
    sample_count: int | None = None,
    method: str = SYNTHETIC_SOLVER_METHOD,
    max_residual: float = SYNTHETIC_SOLVER_MAX_RESIDUAL,
    max_evaluations: int | None = SYNTHETIC_SOLVER_MAX_EVALUATIONS,
    pj2_f_count: int = 6,
    initial: str = "cold",
    norm: str = "fast",
) -> KernelCase:
    count = int(nr if spec.nodes == "grid" else (sample_count or DEFAULT_ROUTE_SAMPLE_COUNT))
    ip_constraint, beta_constraint = constraint_flags(spec.constraint)
    heat_profile, current_profile = source_profiles_for_route(
        spec,
        nr=nr,
        nt=nt,
        sample_count=count,
    )
    signature = (
        SYNTHETIC_PSIN_ROUTE_SIGNATURE
        if spec.coordinate == "psin" and spec.nodes == "uniform"
        else SYNTHETIC_ROUTE_SIGNATURE
    )
    topology = KernelTopology(
        **profile_counts_from_signature(
            signature,
            route=spec.mode,
            coordinate=spec.coordinate,
            nodes=spec.nodes,
            pj2_f_count=pj2_f_count,
        ),
        Nr=int(nr),
        Nt=int(nt),
        route=spec.mode,
        coordinate=spec.coordinate,
        nodes=spec.nodes,
        ip_constraint=ip_constraint,
        beta_constraint=beta_constraint,
        sample_count=count,
        L_max=None,
        M_max=DEFAULT_ROUTE_M_MAX,
        K_max=DEFAULT_ROUTE_K_MAX,
    )
    source = KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=3.0e6 if ip_constraint else np.nan,
        beta=0.02 if beta_constraint else np.nan,
        case_name=spec.case_name,
    )
    config = KernelConfig(
        method=method,
        max_residual=float(max_residual),
        max_evaluations=max_evaluations,
        initial=initial,
        continuation="cold",
        norm=norm,
    )
    return KernelCase(spec.case_name, topology, synthetic_boundary(), source, config)


def geqdsk_kernel_case(
    case_key: str,
    config_label: str,
    *,
    geqdsk_path: Path | None = None,
    route_spec: RouteBenchmarkSpec | None = None,
    signature: dict[str, int] | None = None,
    nr: int = REFERENCE_LAYOUT_NR,
    nt: int = REFERENCE_LAYOUT_NT,
    sample_count: int | None = None,
    method: str = "levenberg-marquardt",
    max_residual: float = SYNTHETIC_SOLVER_MAX_RESIDUAL,
    max_evaluations: int | None = 400,
    initial: str = "cold",
    norm: str = "none",
    boundary_fit_m: int = 10,
    boundary_fit_n: int = 10,
    boundary_maxtol: float = 1.0,
) -> KernelCase:
    geqdsk = Geqdsk(CASE_REFERENCE_GFILES[case_key] if geqdsk_path is None else geqdsk_path)
    boundary = KernelBoundary(
        B0=float(geqdsk.Bt0),
        R_boundary=np.asarray(geqdsk.boundary[:, 0], dtype=np.float64),
        Z_boundary=np.asarray(geqdsk.boundary[:, 1], dtype=np.float64),
        c_order=int(boundary_fit_m),
        s_order=int(boundary_fit_n),
        fit_maxtol=float(boundary_maxtol),
    )
    spec = route_spec or RouteBenchmarkSpec("PF", "psin", "uniform", "Ip")
    effective_signature = (
        dict(signature) if signature is not None else geqdsk_signature(case_key, config_label)
    )
    count = int(
        nr
        if spec.nodes == "grid"
        else (sample_count or max(int(geqdsk.P_psi.size), int(geqdsk.FF_psi.size), 9))
    )
    m_max = max(
        int(np.asarray(boundary.c_offsets, dtype=np.float64).size) - 1,
        int(np.asarray(boundary.s_offsets, dtype=np.float64).size),
        1,
    )
    ip_constraint, beta_constraint = constraint_flags(spec.constraint)
    heat_profile, current_profile = source_profiles_from_geqdsk(
        geqdsk,
        route=spec.mode,
        coordinate=spec.coordinate,
        sample_count=count,
    )
    topology = KernelTopology(
        **profile_counts_from_signature(
            effective_signature,
            route=spec.mode,
            coordinate=spec.coordinate,
            nodes=spec.nodes,
            pj2_f_count=5,
        ),
        Nr=int(nr),
        Nt=int(nt),
        route=spec.mode,
        coordinate=spec.coordinate,
        nodes=spec.nodes,
        ip_constraint=ip_constraint,
        beta_constraint=beta_constraint,
        sample_count=count,
        L_max=None,
        M_max=m_max,
        K_max=max(2, m_max),
    )
    source = KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=abs(float(geqdsk.Ip)) if ip_constraint else np.nan,
        beta=0.02 if beta_constraint else np.nan,
        case_name=f"{case_key}-{config_label}-{spec.case_name}",
    )
    config = KernelConfig(
        method=method,
        max_residual=float(max_residual),
        max_evaluations=max_evaluations,
        initial=initial,
        continuation="cold",
        norm=norm,
    )
    return KernelCase(
        f"{case_key}_{config_label}_{spec.case_name}",
        topology,
        boundary,
        source,
        config,
    )


@lru_cache(maxsize=1)
def load_reduced_equilibrium_manifest() -> dict[tuple[str, str], dict[str, object]]:
    return _load_manifest(REDUCED_EQUILIBRIUM_MANIFEST_PATH)


@lru_cache(maxsize=1)
def load_reference_equilibrium_manifest() -> dict[tuple[str, str], dict[str, object]]:
    return _load_manifest(REFERENCE_EQUILIBRIUM_MANIFEST_PATH)


def _load_manifest(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def geqdsk_signature(case_key: str, config_label: str) -> dict[str, int]:
    if config_label == "Ref":
        entry = load_reference_equilibrium_manifest().get((case_key, "Ref"))
    else:
        entry = load_reduced_equilibrium_manifest().get((case_key, config_label))
    if entry is None:
        raise FileNotFoundError(f"missing benchmark manifest entry for {case_key} {config_label}")
    signature = entry.get("signature", {})
    if not isinstance(signature, dict):
        return {}
    return {str(name): int(value) for name, value in signature.items() if int(value) > 0}


def topology_profile_counts(topology: KernelTopology) -> dict[str, Any]:
    return {
        "h": int(topology.h_count),
        "v": int(topology.v_count),
        "kappa": int(topology.kappa_count),
        "psin": int(topology.psin_count),
        "F": int(topology.F_count),
        "c": [int(value) for value in topology.c_counts],
        "s": [int(value) for value in topology.s_counts],
    }


def route_topology_payload(
    topology: KernelTopology,
    recipe: KernelRecipe,
    *,
    status: str = "planned",
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "key": topology.key,
        "source": {
            "route": topology.route,
            "route_key": [topology.route, topology.coordinate, topology.nodes],
            "coordinate": topology.coordinate,
            "nodes": topology.nodes,
            "constraint": topology.constraint_label,
            "uses_Ip": bool(topology.ip_constraint),
            "uses_beta": bool(topology.beta_constraint),
            "sample_count": int(topology.sample_count),
        },
        "layout": {"packed": recipe.layout},
        "recipe": {
            "backend": recipe.backend,
            "preset": recipe.build,
            "layout": {"packed": recipe.layout},
        },
        "profile_counts": topology_profile_counts(topology),
        "grid": {"Nr": int(topology.Nr), "Nt": int(topology.Nt)},
        "sample_count": int(topology.sample_count),
        "warnings": [str(item) for item in warnings],
    }


def solve_numba_case(case: KernelCase) -> tuple[SolveResult, Any]:
    kernel = Kernel(
        topology=case.topology,
        recipe=KernelRecipe(backend="numba", layout="degree"),
        config=case.config,
    )
    result = kernel.solve(case.boundary, case.source)
    return result, kernel


def solve_native_case(
    case: KernelCase, *, recipe: KernelRecipe | None = None
) -> tuple[SolveResult, Any]:
    kernel = Kernel(
        topology=case.topology,
        recipe=recipe or KernelRecipe(backend="cxx", layout="degree"),
        config=case.config,
    )
    result = kernel.solve(case.boundary, case.source)
    return result, kernel


def measure_solver(
    solve_once: Callable[[], tuple[SolveResult, Any]],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(max(0, int(warmup))):
        result, kernel = solve_once()
        close = getattr(kernel, "close", None)
        if close is not None:
            close()
        if not isinstance(result, SolveResult):
            raise RuntimeError("warmup solve did not return SolveResult")

    timings: list[float] = []
    wall_timings: list[float] = []
    results: list[SolveResult] = []
    nfev: list[int] = []
    njev: list[int] = []
    iterations: list[int] = []
    callbacks: list[int] = []
    jacobian_component_evaluations: list[int] = []
    jvp_evaluations: list[int] = []
    last_kernel = None
    for _ in range(max(1, int(repeat))):
        started = time.perf_counter_ns()
        result, kernel = solve_once()
        wall_timings.append(float(time.perf_counter_ns() - started) / 1.0e6)
        timings.append(float(result.elapsed_ms))
        results.append(result)
        nfev.append(int(result.nfev))
        njev.append(int(result.njev))
        iterations.append(int(result.linear_iterations))
        callbacks.append(int(result.callbacks))
        jacobian_component_evaluations.append(int(result.jacobian_component_evaluations))
        jvp_evaluations.append(int(result.jvp_evaluations))
        if last_kernel is not None:
            close = getattr(last_kernel, "close", None)
            if close is not None:
                close()
        last_kernel = kernel
    return {
        "timings_ms": timings,
        "wall_timings_ms": wall_timings,
        "median_ms": median(timings),
        "result": results[-1],
        "kernel": last_kernel,
        "success": all(bool(result.success) for result in results),
        "nfev": nfev,
        "njev": njev,
        "iterations": iterations,
        "callbacks": callbacks,
        "jacobian_component_evaluations": jacobian_component_evaluations,
        "jvp_evaluations": jvp_evaluations,
        "nfev_median": median([float(value) for value in nfev]),
    }


def result_payload(result: SolveResult) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "info": int(result.info),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "callbacks": int(result.callbacks),
        "linear_iterations": int(result.linear_iterations),
        "raw_norm": float(result.raw_norm),
        "scaled_norm": float(result.scaled_norm),
        "alpha": [float(value) for value in np.asarray(result.alpha, dtype=np.float64)],
        "x_size": int(result.x.shape[0]),
    }


def row_payload(name: str, case: KernelCase, measure: dict[str, Any]) -> dict[str, Any]:
    result = measure["result"]
    return {
        "case": name,
        "topology_key": case.topology.key,
        "route": case.topology.route,
        "coordinate": case.topology.coordinate,
        "nodes": case.topology.nodes,
        "constraint": case.topology.constraint_label,
        "x_size": case.topology.x_size,
        "median_ms": float(measure["median_ms"]),
        "nfev_median": float(measure["nfev_median"]),
        **result_payload(result),
    }


def engine_payload(measure: dict[str, Any]) -> dict[str, Any]:
    result = measure["result"]
    return {
        "success_all": bool(measure["success"]),
        "info": int(result.info),
        "timing": float_stats(measure["timings_ms"]),
        "wall_timing": float_stats(measure.get("wall_timings_ms", [])),
        "inner_timing": float_stats(measure["timings_ms"]),
        "nfev": int_stats(measure["nfev"]),
        "njev": int_stats(measure["njev"]),
        "iterations": int_stats(measure["iterations"]),
        "callbacks": int_stats(measure["callbacks"]),
        "jacobian_component_evaluations": int_stats(measure["jacobian_component_evaluations"]),
        "jvp_evaluations": int_stats(measure["jvp_evaluations"]),
        "raw_norm": float(result.raw_norm),
        "scaled_norm": float(result.scaled_norm),
        "x": result.x.tolist(),
        "raw": result.raw.tolist(),
        "alpha": result.alpha.tolist(),
        "message": "",
    }


def runtime_payload(
    *,
    status: str,
    x_size: int,
    engine: str,
    measure: dict[str, Any] | None,
    failure_reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "x_size": int(x_size),
    }
    if measure is not None:
        payload["engine"] = engine
        payload["engines"] = {engine: engine_payload(measure)}
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    return payload


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


def timing_median_ms(engine: dict[str, Any] | None) -> float:
    if engine is None:
        return float("nan")
    timing = engine.get("timing")
    if not isinstance(timing, dict):
        return float("nan")
    return float(timing.get("median_ms", float("nan")))


def max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    return float(np.max(np.abs(lhs_arr - rhs_arr))) if lhs_arr.size else 0.0


def continuation_points(
    case: KernelCase,
    *,
    update: str,
    span: float,
    points: int,
) -> tuple[KernelCase, ...]:
    offsets = _scan_offsets(points=points, span=span)
    return tuple(
        _updated_case(case, update=update, offset=offset, index=index)
        for index, offset in enumerate(offsets)
    )


def _scan_offsets(*, points: int, span: float) -> tuple[float, ...]:
    if points <= 0:
        raise ValueError("points must be positive")
    if points == 1:
        return (0.0,)
    lower = -0.5 * float(span)
    step = float(span) / float(points - 1)
    return tuple(float(lower + index * step) for index in range(points))


def _updated_case(case: KernelCase, *, update: str, offset: float, index: int) -> KernelCase:
    suffix = f"{update}-{index:03d}"
    boundary = case.boundary
    source = case.source
    if update in {"boundary", "mixed"}:
        boundary = replace(
            boundary,
            c_offsets=_scale_array(boundary.c_offsets, offset, strength=0.5, keep_first=False),
            s_offsets=_scale_array(boundary.s_offsets, offset, strength=0.5, keep_first=False),
        )
    if update in {"ip", "mixed"} and np.isfinite(source.Ip):
        source = replace(source, Ip=float(source.Ip) * (1.0 + offset))
    if update in {"source", "mixed"}:
        source = replace(
            source,
            heat_profile=_scale_profile(source.heat_profile, offset, sign=1.0),
            current_profile=_scale_profile(source.current_profile, offset, sign=-1.0),
        )
    source = replace(source, case_name=f"{source.case_name or case.name}-{suffix}")
    return replace(case, name=f"{case.name}_{suffix}", boundary=boundary, source=source)


def _scale_array(
    values: np.ndarray | tuple[float, ...], offset: float, *, strength: float, keep_first: bool
) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    for index in range(arr.size):
        if keep_first and index == 0:
            continue
        direction = 1.0 if index % 2 == 0 else -1.0
        arr[index] *= 1.0 + strength * offset * direction / float(index + 1)
    return arr


def _scale_profile(values: np.ndarray, offset: float, *, sign: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return np.array(arr, dtype=np.float64, copy=True)
    axis = np.linspace(-1.0, 1.0, arr.size, dtype=np.float64)
    return np.array(arr * (1.0 + sign * offset * axis), dtype=np.float64, copy=True)


def selected_cases(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else CASE_KEYS


def selected_configs(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else CONFIG_LABELS
