"""Repository benchmark and regression-driving script.

This file is intentionally script-oriented rather than pytest-oriented. It
builds one high-resolution PF reference solve, projects that reference into a
matrix of route/constraint cases, and writes comparison artifacts under
``tests/benchmark/``.

Note: The first run may be slower due to JIT compilation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "veqpy-mpl"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

import veqpy.engine.backend_abi as backend_abi
from veqpy.engine.numba_profile import update_profile
from veqpy.engine.numba_source import source_parameterization_for_route_key
from veqpy.model import Boundary, Grid
from veqpy.operator import (
    Operator,
    OperatorCase,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
)
from veqpy.solver import Solver, SolverConfig

PLOT = False
SHOW_PROGRESS = True
COMPARE_SHAPE_KEYS = ("h", "k", "s1")
COMPARE_SOURCE_KEYS = (
    ("psi_r", r"$\psi_\rho$"),
    ("FF_psi", r"$FF_\psi$"),
    ("mu0_P_psi", r"$\mu_0 P_\psi$"),
)
COMPARE_SURFACE_LEVELS = tuple(np.linspace(0.1, 1.0, 10, dtype=np.float64))

# Reference solve: high-resolution baseline used to derive downstream cases.
REFERENCE_SOURCE_SAMPLE_COUNT = 51
TEST_SOURCE_SAMPLE_COUNT = 51
BENCHMARK_REPEAT_COUNT = 100
SHAPE_MATCH_TOL = 1e-2
REFERENCE_CACHE_VERSION = 6
BENCHMARK_BASELINE_SCHEMA_VERSION = 1
DIAGNOSTIC_SIGN_CHANGE_WINDOW = 12
MU0 = 4.0e-7 * np.pi

REFERENCE_GRID = Grid(
    Nr=64,
    Nt=32,
    quadrature_scheme="legendre",
)

TEST_GRID = Grid(
    Nr=32,
    Nt=16,
    quadrature_scheme="legendre",
)

REFERENCE_SUMMARY_GRID = Grid(
    Nr=64,
    Nt=128,
    quadrature_scheme="uniform",
)

CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=False,
    enable_history=False,
)

# Minimal robust coefficient seeds for benchmark cases.
BASE_COEFFS = {
    "h": [0.0] * 3,
    "k": [0.0] * 6,
    "s1": [0.0] * 3,
}

PSIN_ROBUST_COEFFS = {
    **BASE_COEFFS,
    "psin": [0.0] * 6,
}

F_ROBUST_COEFFS = {
    **BASE_COEFFS,
    "F": [0.0] * 6,
}

BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)

REFERENCE_IP = 3.0e6
REFERENCE_MU0_IP = MU0 * REFERENCE_IP
SHAPE_PROFILE_NAMES = build_shape_profile_names(REFERENCE_GRID.M_max)
BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
BENCHMARK_INPUT_KINDS = ("uniform",)
BENCHMARK_MODE_CONSTRAINTS = {
    "PF": ("null", "Ip", "beta"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}
BENCHMARK_BASELINE_PATH = (
    Path(__file__).resolve().parent / "baselines" / "benchmark_non_timing.json"
)


@dataclass(frozen=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True)
class ReferenceBundle:
    result: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    reference_shape_x: np.ndarray
    rho_axis: np.ndarray
    psin_axis: np.ndarray
    rho_interp_axis: PreparedInterpAxis
    psin_interp_axis: PreparedInterpAxis


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    coordinate: str
    constraint: str
    input_kind: str

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.input_kind}_{self.constraint}"


@dataclass(frozen=True)
class BenchmarkCaseResult:
    spec: BenchmarkCaseSpec
    result: object
    equilibrium: object
    avg_ms: float
    std_ms: float
    shape_error: float
    psi_r_rel_rms_error: float
    psi_r_rel_max_error: float
    psi_r_head_sign_changes: int
    psi_r_tail_sign_changes: int
    ff_psi_rel_rms_error: float
    ff_psi_rel_max_error: float
    ff_psi_head_sign_changes: int
    ff_psi_tail_sign_changes: int
    mu0_p_psi_rel_rms_error: float
    mu0_p_psi_rel_max_error: float
    mu0_p_psi_head_sign_changes: int
    mu0_p_psi_tail_sign_changes: int

    @property
    def case_name(self) -> str:
        return self.spec.case_name


_UNIFORM_SOURCE_AXIS = np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT, dtype=np.float64)
_UNIFORM_SOURCE_AXIS_SQRT_PSIN = _UNIFORM_SOURCE_AXIS**2
_TEST_GRID_RHO_AXIS = np.asarray(TEST_GRID.rho, dtype=np.float64)
_REFERENCE_SUMMARY_RHO_AXIS = np.asarray(REFERENCE_SUMMARY_GRID.rho, dtype=np.float64)


def _sort_rows_desc(rows: list[BenchmarkCaseResult], key_fn) -> list[BenchmarkCaseResult]:
    return sorted(rows, key=lambda row: (-float(key_fn(row)), row.case_name))


def _render_ranking_section(
    title: str,
    rows: list[BenchmarkCaseResult],
    *,
    columns,
) -> list[str]:
    lines = ["", title, ""]
    header_parts = []
    for align, label, width, _ in columns:
        if align == "left":
            header_parts.append(label.ljust(width))
        else:
            header_parts.append(label.rjust(width))
    header = " | ".join(header_parts)
    lines.append(header)
    lines.append("-" * len(header))
    for index, row in enumerate(rows, start=1):
        value_parts = []
        for align, _, width, formatter in columns:
            value = str(formatter(index, row))
            if align == "left":
                value_parts.append(value.ljust(width))
            else:
                value_parts.append(value.rjust(width))
        lines.append(" | ".join(value_parts))
    return lines


def _artifact_dir() -> Path:
    """Keep generated benchmark artifacts in one ignored location."""
    outdir = Path(__file__).resolve().parent / "benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _plot_dir() -> Path:
    outdir = _artifact_dir() / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _reference_summary_json_path() -> Path:
    return _artifact_dir() / "reference_summary.json"


def _reference_cache_path() -> Path:
    return _artifact_dir() / "reference_bundle.pkl"


def _render_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    if not pairs:
        return []
    key_width = max(len(key) for key, _ in pairs)
    return [f"{key:<{key_width}} : {value}" for key, value in pairs]


def _as_float64_array(values, *, copy: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if copy:
        return arr.copy()
    return arr


def _extract_shape_x(profile_coeffs: dict[str, list[float] | None], x: np.ndarray) -> np.ndarray:
    profile_names = build_profile_names(REFERENCE_GRID.M_max)
    profile_index = build_profile_index(profile_names)
    _, coeff_index, _ = build_profile_layout(profile_coeffs, profile_names=profile_names)
    shape_values: list[float] = []
    for k in range(coeff_index.shape[1]):
        for name in SHAPE_PROFILE_NAMES:
            idx = int(coeff_index[profile_index[name], k])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def _prepare_interp_axis(axis: np.ndarray) -> PreparedInterpAxis:
    axis_f64 = _as_float64_array(axis)
    order = np.argsort(axis_f64)
    axis_sorted = axis_f64[order]
    unique_axis, unique_index = np.unique(axis_sorted, return_index=True)
    return PreparedInterpAxis(unique_axis=unique_axis, order=order, unique_index=unique_index)


def _prepare_interp_values(values: np.ndarray, prepared_axis: PreparedInterpAxis) -> np.ndarray:
    values_f64 = _as_float64_array(values)
    return values_f64[prepared_axis.order][prepared_axis.unique_index]


def _unique_interp(
    axis: np.ndarray | PreparedInterpAxis,
    values: np.ndarray,
    x_new: np.ndarray,
    *,
    kind: str = "cubic",
) -> np.ndarray:
    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else _prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = _prepare_interp_values(values, prepared_axis)
    interp_kind = kind if unique_axis.size >= 4 else "linear"
    fn = interp1d(
        unique_axis, unique_values, kind=interp_kind, fill_value="extrapolate", assume_sorted=True
    )
    return _as_float64_array(fn(_as_float64_array(x_new)))


def _profile_interp(
    axis: np.ndarray | PreparedInterpAxis, values: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else _prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = _prepare_interp_values(values, prepared_axis)
    x_new = _as_float64_array(x_new)
    if unique_axis.size < 2:
        return np.full_like(
            x_new, float(unique_values[0] if unique_values.size else 0.0), dtype=np.float64
        )
    if unique_axis.size < 3:
        return np.interp(x_new, unique_axis, unique_values).astype(np.float64, copy=False)
    return _as_float64_array(PchipInterpolator(unique_axis, unique_values, extrapolate=True)(x_new))


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic PF reference profiles used to build the baseline solve."""
    beta0 = 0.75

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = _as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1e-14, psin_r, 1e-14)

    psi_r = _as_float64_array(equilibrium.alpha2 * psin_r)
    psi_r_safe = np.where(np.abs(psi_r) > 1e-14, psi_r, 1e-14)

    FFn_r = _as_float64_array(equilibrium.FFn_r, copy=True)
    Pn_r = _as_float64_array(equilibrium.Pn_r, copy=True)
    FF_r = _as_float64_array(equilibrium.FF_r, copy=True)
    P_r = _as_float64_array(equilibrium.P_r, copy=True)
    Itor = _as_float64_array(equilibrium.Itor, copy=True)
    jtor = _as_float64_array(equilibrium.jtor, copy=True)
    jpara = _as_float64_array(equilibrium.jpara, copy=True)
    q = _as_float64_array(equilibrium.q, copy=True)
    mu0_P_r = MU0 * P_r
    mu0_P_psi = mu0_P_r / psi_r_safe
    mu0_Itor = MU0 * Itor
    mu0_jtor = MU0 * jtor
    mu0_jpara = MU0 * jpara

    return {
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_r": FFn_r,
        "Pn_r": Pn_r,
        "FFn_psin": FFn_r / psin_r_safe,
        "Pn_psin": Pn_r / psin_r_safe,
        "setup_Pn_r": Pn_r / MU0,
        "setup_Pn_psin": (Pn_r / psin_r_safe) / MU0,
        "FF_r": FF_r,
        "P_r": P_r,
        "mu0_P_r": mu0_P_r,
        "FF_psi": FF_r / psi_r_safe,
        "P_psi": P_r / psi_r_safe,
        "mu0_P_psi": mu0_P_psi,
        "Itorn": mu0_Itor,
        "Itor": Itor,
        "mu0_Itor": mu0_Itor,
        "jtorn": mu0_jtor,
        "jtor": jtor,
        "mu0_jtor": mu0_jtor,
        "jparan": mu0_jpara,
        "jpara": jpara,
        "mu0_jpara": mu0_jpara,
        "qn": q * 0.1,
        "q": q,
        "mu0_Ip": float(MU0 * equilibrium.Ip),
        "beta_constraint": float(equilibrium.beta_t),
    }


def _reference_pf_case() -> OperatorCase:
    # Start from a stable PF/rho/uniform case, then reuse its solved profiles
    # to build the wider route/constraint benchmark matrix.
    rho_src = np.linspace(0.0, 1.0, REFERENCE_SOURCE_SAMPLE_COUNT)
    psin_src = rho_src * rho_src
    FFn_psin_src, Pn_psin_src = pf_reference_profiles(psin_src)
    FFn_r_src = FFn_psin_src * (2.0 * rho_src)
    Pn_r_src = Pn_psin_src * (2.0 * rho_src)
    return OperatorCase(
        route="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=BASE_COEFFS,
        boundary=BOUNDARY,
        heat_input=Pn_r_src / MU0,
        current_input=FFn_r_src,
        Ip=REFERENCE_IP,
    )


def _reference_cache_signature() -> dict[str, object]:
    return {
        "version": REFERENCE_CACHE_VERSION,
        "reference_source_sample_count": int(REFERENCE_SOURCE_SAMPLE_COUNT),
        "reference_ip": float(REFERENCE_IP),
        "reference_mu0_ip": float(REFERENCE_MU0_IP),
        "reference_grid": {
            "Nr": int(REFERENCE_GRID.Nr),
            "Nt": int(REFERENCE_GRID.Nt),
            "quadrature_scheme": REFERENCE_GRID.quadrature_scheme,
            "calculus": REFERENCE_GRID.calculus_scheme,
            "L_max": int(REFERENCE_GRID.L_max),
            "M_max": int(REFERENCE_GRID.M_max),
        },
        "boundary": {
            "a": float(BOUNDARY.a),
            "R0": float(BOUNDARY.R0),
            "Z0": float(BOUNDARY.Z0),
            "B0": float(BOUNDARY.B0),
            "ka": float(BOUNDARY.ka),
            "s_offsets": _as_float64_array(BOUNDARY.s_offsets).tolist(),
        },
        "config": {
            "method": CONFIG.method,
            "max_residual": float(CONFIG.max_residual),
            "max_evaluations": int(CONFIG.max_evaluations),
        },
    }


def _is_reference_equilibrium_cache_compatible(equilibrium: object) -> bool:
    grid = getattr(equilibrium, "grid", None)
    if grid is None:
        return False
    if not isinstance(getattr(grid, "L_max", None), int):
        return False
    if not isinstance(getattr(grid, "M_max", None), int):
        return False
    K_max = getattr(grid, "K_max", None)
    if K_max is not None and not isinstance(K_max, int):
        return False

    rho = np.asarray(getattr(grid, "rho", None), dtype=np.float64)
    psin = np.asarray(getattr(equilibrium, "psin", None), dtype=np.float64)
    psin_r = np.asarray(getattr(equilibrium, "psin_r", None), dtype=np.float64)
    ffn_psin = np.asarray(getattr(equilibrium, "FFn_psin", None), dtype=np.float64)
    pn_psin = np.asarray(getattr(equilibrium, "Pn_psin", None), dtype=np.float64)

    if rho.ndim != 1:
        return False
    try:
        R = np.asarray(getattr(equilibrium, "R"), dtype=np.float64)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if R.shape != (grid.Nr, grid.Nt):
        return False

    expected_shape = rho.shape
    for profile in (psin, psin_r, ffn_psin, pn_psin):
        if profile.ndim != 1 or profile.shape != expected_shape:
            return False

    return True


def _load_reference_cache() -> ReferenceBundle | None:
    path = _reference_cache_path()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict) or payload.get("signature") != _reference_cache_signature():
        return None

    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        return None
    equilibrium = bundle.get("equilibrium")
    if not _is_reference_equilibrium_cache_compatible(equilibrium):
        return None

    rho_axis = _as_float64_array(bundle["rho_axis"])
    psin_axis = _as_float64_array(bundle["psin_axis"])
    return ReferenceBundle(
        result=bundle["result"],
        equilibrium=equilibrium,
        ref_profiles=bundle["ref_profiles"],
        reference_shape_x=_as_float64_array(bundle["reference_shape_x"]),
        rho_axis=rho_axis,
        psin_axis=psin_axis,
        rho_interp_axis=_prepare_interp_axis(rho_axis),
        psin_interp_axis=_prepare_interp_axis(psin_axis),
    )


def _write_reference_cache(reference: ReferenceBundle) -> None:
    path = _reference_cache_path()
    payload = {
        "signature": _reference_cache_signature(),
        "bundle": {
            "result": reference.result,
            "equilibrium": reference.equilibrium,
            "ref_profiles": reference.ref_profiles,
            "reference_shape_x": reference.reference_shape_x,
            "rho_axis": reference.rho_axis,
            "psin_axis": reference.psin_axis,
        },
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _solve_reference(*, show_progress: bool = False) -> ReferenceBundle:
    """Solve or load the high-resolution reference equilibrium."""
    cached = _load_reference_cache()
    if cached is not None:
        if show_progress:
            print(f"reference cache hit: {_reference_cache_path().name}")
        return cached

    solver = Solver(operator=Operator(REFERENCE_GRID, _reference_pf_case()), config=CONFIG)
    solver.solve(
        method=CONFIG.method,
        max_residual=CONFIG.max_residual,
        max_evaluations=CONFIG.max_evaluations,
        enable_verbose=False,
        enable_history=False,
    )
    result = solver.result
    equilibrium = solver.build_equilibrium()
    rho_axis = _as_float64_array(equilibrium.rho)
    psin_axis = _as_float64_array(equilibrium.psin)
    reference = ReferenceBundle(
        result=result,
        equilibrium=equilibrium,
        ref_profiles=build_pf_reference_profiles(equilibrium),
        reference_shape_x=_extract_shape_x(solver.operator.case.profile_coeffs, result.x),
        rho_axis=rho_axis,
        psin_axis=psin_axis,
        rho_interp_axis=_prepare_interp_axis(rho_axis),
        psin_interp_axis=_prepare_interp_axis(psin_axis),
    )
    _write_reference_cache(reference)
    if show_progress:
        print(f"reference cache saved: {_reference_cache_path().name}")
    return reference


def _constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    return "physical", "physical"


def _pressure_keys_for_coordinate(coordinate: str) -> tuple[str, str]:
    if coordinate == "rho":
        return "setup_Pn_r", "P_r"
    return "setup_Pn_psin", "P_psi"


def _pick_ref_profile(
    ref: dict[str, np.ndarray | float],
    normalized_key: str,
    physical_key: str,
    normalized: bool,
) -> np.ndarray:
    key = normalized_key if normalized else physical_key
    return ref[key]


def _build_mode_init_kwargs(
    mode: str,
    coordinate: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    pressure_keys = _pressure_keys_for_coordinate(coordinate)

    if mode == "PF":
        use_normalized = constraint in {"Ip", "beta"}
        driver_keys = ("FFn_r", "FF_r") if coordinate == "rho" else ("FFn_psin", "FF_psi")
        return {
            "current_input": _pick_ref_profile(ref, driver_keys[0], driver_keys[1], use_normalized),
            "heat_input": _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], use_normalized
            ),
        }

    if mode == "PP":
        driver_normalized = constraint in {"Ip_beta", "Ip"}
        pressure_normalized = constraint in {"Ip_beta", "beta"}
        return {
            "current_input": _pick_ref_profile(ref, "psin_r", "psi_r", driver_normalized),
            "heat_input": _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], pressure_normalized
            ),
        }

    driver_domain, pressure_domain = _constraint_route_domains(constraint)
    driver_keys = {
        "PI": ("Itor", "Itor"),
        "PJ1": ("jtor", "jtor"),
        "PJ2": ("jpara", "jpara"),
        "PQ": ("qn", "q"),
    }[mode]
    driver_normalized = driver_domain == "normalized"
    pressure_normalized = pressure_domain == "normalized"
    return {
        "current_input": _pick_ref_profile(ref, driver_keys[0], driver_keys[1], driver_normalized),
        "heat_input": _pick_ref_profile(
            ref, pressure_keys[0], pressure_keys[1], pressure_normalized
        ),
    }


def _uniform_source_axis(spec: BenchmarkCaseSpec) -> np.ndarray:
    route_key = (str(spec.mode).upper(), str(spec.coordinate).lower(), str(spec.input_kind).lower())
    if source_parameterization_for_route_key(route_key) == "sqrt_psin":
        return _UNIFORM_SOURCE_AXIS_SQRT_PSIN
    return _UNIFORM_SOURCE_AXIS


def _resample_reference_input(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    spec: BenchmarkCaseSpec,
) -> np.ndarray:
    uniform_axis = _uniform_source_axis(spec)
    return _profile_interp(source_axis, values, uniform_axis)


def _sample_reference_input_on_grid(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    grid_axis: np.ndarray,
) -> np.ndarray:
    return _profile_interp(source_axis, values, grid_axis)


def _profile_coeffs_for_case(
    mode: str,
    coordinate: str,
    input_kind: str,
    *,
    constraint: str | None = None,
) -> dict[str, list[float] | None]:
    """Choose a conservative coefficient seed for one benchmark case."""
    route_key = (str(mode).upper(), str(coordinate).lower(), str(input_kind).lower())
    if route_key in backend_abi.PROFILE_OWNED_PSIN_ROUTE_KEYS:
        coeffs = {name: list(values) for name, values in PSIN_ROBUST_COEFFS.items()}
    else:
        coeffs = {name: list(values) for name, values in BASE_COEFFS.items()}
    if mode in {"PJ2"}:
        f_order = 5
        coeffs["F"] = [0.0] * f_order
    return coeffs


def _make_benchmark_case(spec: BenchmarkCaseSpec, reference: ReferenceBundle) -> OperatorCase:
    """Project the reference solution onto one route/constraint test case."""
    init_kwargs = _build_mode_init_kwargs(
        spec.mode, spec.coordinate, spec.constraint, reference.ref_profiles
    )
    heat_profile = init_kwargs["heat_input"]
    current_profile = init_kwargs["current_input"]
    if spec.input_kind == "grid":
        if spec.coordinate == "rho":
            grid_axis = _TEST_GRID_RHO_AXIS
            source_axis = reference.rho_interp_axis
        else:
            grid_axis = _profile_interp(
                reference.rho_interp_axis, reference.psin_axis, _TEST_GRID_RHO_AXIS
            )
            source_axis = reference.psin_interp_axis
        heat_input = _sample_reference_input_on_grid(heat_profile, source_axis, grid_axis)
        current_input = _sample_reference_input_on_grid(current_profile, source_axis, grid_axis)
        nodes = "grid"
    else:
        source_axis = (
            reference.rho_interp_axis if spec.coordinate == "rho" else reference.psin_interp_axis
        )
        heat_input = _resample_reference_input(heat_profile, source_axis, spec)
        current_input = _resample_reference_input(current_profile, source_axis, spec)
        nodes = "uniform"
    Ip = float(reference.equilibrium.Ip) if spec.constraint in {"Ip", "Ip_beta"} else None
    beta = (
        float(reference.ref_profiles["beta_constraint"])
        if spec.constraint in {"beta", "Ip_beta"}
        else None
    )
    return OperatorCase(
        route=spec.mode,
        profile_coeffs=_profile_coeffs_for_case(
            spec.mode,
            spec.coordinate,
            spec.input_kind,
            constraint=spec.constraint,
        ),
        boundary=BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        coordinate=spec.coordinate,
        nodes=nodes,
        Ip=Ip,
        beta=beta,
    )


def _iter_benchmark_specs():
    for mode in BENCHMARK_MODES:
        for coordinate in ("rho", "psin"):
            for input_kind in BENCHMARK_INPUT_KINDS:
                for constraint in BENCHMARK_MODE_CONSTRAINTS[mode]:
                    yield BenchmarkCaseSpec(
                        mode=mode,
                        coordinate=coordinate,
                        constraint=constraint,
                        input_kind=input_kind,
                    )


def _solve_once(case: OperatorCase) -> tuple[object, object, np.ndarray]:
    solver = Solver(operator=Operator(TEST_GRID, case), config=CONFIG)
    solver.solve(
        method=CONFIG.method,
        max_residual=CONFIG.max_residual,
        max_evaluations=CONFIG.max_evaluations,
        enable_verbose=False,
        enable_history=False,
    )
    result = solver.result
    if result is None:
        raise RuntimeError("benchmark solve produced no result")
    equilibrium = solver.build_equilibrium()
    shape_x = _extract_shape_x(case.profile_coeffs, result.x)
    return result, equilibrium, shape_x


def _solve_with_timing(case: OperatorCase) -> tuple[object, object, np.ndarray, float, float]:
    solver = Solver(operator=Operator(TEST_GRID, case), config=CONFIG)
    solver.solve(
        method=CONFIG.method,
        max_residual=CONFIG.max_residual,
        max_evaluations=CONFIG.max_evaluations,
        enable_verbose=False,
        enable_history=False,
    )

    elapsed_ms_samples = np.empty(BENCHMARK_REPEAT_COUNT, dtype=np.float64)
    result = None
    for index in range(BENCHMARK_REPEAT_COUNT):
        solver.solve(
            method=CONFIG.method,
            max_residual=CONFIG.max_residual,
            max_evaluations=CONFIG.max_evaluations,
            enable_verbose=False,
            enable_history=False,
        )
        result = solver.result
        elapsed_ms_samples[index] = float(result.elapsed) / 1000.0

    if result is None:
        raise RuntimeError("benchmark solve produced no result")

    equilibrium = solver.build_equilibrium()
    shape_x = _extract_shape_x(case.profile_coeffs, result.x)
    return (
        result,
        equilibrium,
        shape_x,
        float(np.mean(elapsed_ms_samples)),
        float(np.std(elapsed_ms_samples)),
    )


def _shape_error(reference_x: np.ndarray, current_x: np.ndarray) -> float:
    n = min(reference_x.shape[0], current_x.shape[0])
    if n == 0:
        return 0.0
    return float(np.max(np.abs(current_x[:n] - reference_x[:n])))


def _relative_profile_errors(
    reference_values: np.ndarray, current_values: np.ndarray
) -> tuple[float, float]:
    reference_values = _as_float64_array(reference_values)
    current_values = _as_float64_array(current_values)
    n = min(reference_values.shape[0], current_values.shape[0])
    if n == 0:
        return 0.0, 0.0
    reference_values = reference_values[:n]
    current_values = current_values[:n]
    diff = current_values - reference_values
    scale = max(float(np.max(np.abs(reference_values))), 1.0e-12)
    rel_rms = float(np.sqrt(np.mean(diff * diff)) / scale)
    rel_max = float(np.max(np.abs(diff)) / scale)
    return rel_rms, rel_max


def _window_derivative_sign_changes(
    values: np.ndarray, *, side: str, window: int = DIAGNOSTIC_SIGN_CHANGE_WINDOW
) -> int:
    values = _as_float64_array(values)
    count = min(int(window), values.shape[0])
    if side == "head":
        sample = values[:count]
    elif side == "tail":
        sample = values[-count:]
    else:
        raise ValueError(f"Unsupported side {side!r}")
    delta = np.diff(sample)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _diagnostic_profile_metrics(
    reference_axis: PreparedInterpAxis,
    reference_values: np.ndarray,
    current_axis: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float, int, int]:
    current_axis = _as_float64_array(current_axis)
    current_values = _as_float64_array(current_values)
    reference_on_current = _profile_interp(reference_axis, reference_values, current_axis)
    rel_rms, rel_max = _relative_profile_errors(reference_on_current, current_values)
    return (
        rel_rms,
        rel_max,
        _window_derivative_sign_changes(current_values, side="head"),
        _window_derivative_sign_changes(current_values, side="tail"),
    )


def _benchmark_case_diagnostics(
    reference: ReferenceBundle, equilibrium: object, shape_x: np.ndarray
) -> dict[str, float | int]:
    psi_r_rel_rms_error, psi_r_rel_max_error, psi_r_head_sign_changes, psi_r_tail_sign_changes = (
        _diagnostic_profile_metrics(
            reference.rho_interp_axis,
            reference.ref_profiles["psi_r"],
            equilibrium.rho,
            equilibrium.alpha2 * equilibrium.psin_r,
        )
    )
    (
        ff_psi_rel_rms_error,
        ff_psi_rel_max_error,
        ff_psi_head_sign_changes,
        ff_psi_tail_sign_changes,
    ) = _diagnostic_profile_metrics(
        reference.rho_interp_axis,
        reference.ref_profiles["FF_psi"],
        equilibrium.rho,
        equilibrium.alpha1 * equilibrium.FFn_psin,
    )
    (
        mu0_p_psi_rel_rms_error,
        mu0_p_psi_rel_max_error,
        mu0_p_psi_head_sign_changes,
        mu0_p_psi_tail_sign_changes,
    ) = _diagnostic_profile_metrics(
        reference.rho_interp_axis,
        reference.ref_profiles["mu0_P_psi"],
        equilibrium.rho,
        equilibrium.alpha1 * equilibrium.Pn_psin,
    )
    return {
        "shape_error": _shape_error(reference.reference_shape_x, shape_x),
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


def _benchmark_case_result(
    spec: BenchmarkCaseSpec, reference: ReferenceBundle
) -> BenchmarkCaseResult:
    case = _make_benchmark_case(spec, reference)
    result, equilibrium, shape_x, avg_ms, std_ms = _solve_with_timing(case)
    metrics = _benchmark_case_diagnostics(reference, equilibrium, shape_x)
    return BenchmarkCaseResult(
        spec=spec,
        result=result,
        equilibrium=equilibrium,
        avg_ms=avg_ms,
        std_ms=std_ms,
        shape_error=float(metrics["shape_error"]),
        psi_r_rel_rms_error=float(metrics["psi_r_rel_rms_error"]),
        psi_r_rel_max_error=float(metrics["psi_r_rel_max_error"]),
        psi_r_head_sign_changes=int(metrics["psi_r_head_sign_changes"]),
        psi_r_tail_sign_changes=int(metrics["psi_r_tail_sign_changes"]),
        ff_psi_rel_rms_error=float(metrics["ff_psi_rel_rms_error"]),
        ff_psi_rel_max_error=float(metrics["ff_psi_rel_max_error"]),
        ff_psi_head_sign_changes=int(metrics["ff_psi_head_sign_changes"]),
        ff_psi_tail_sign_changes=int(metrics["ff_psi_tail_sign_changes"]),
        mu0_p_psi_rel_rms_error=float(metrics["mu0_p_psi_rel_rms_error"]),
        mu0_p_psi_rel_max_error=float(metrics["mu0_p_psi_rel_max_error"]),
        mu0_p_psi_head_sign_changes=int(metrics["mu0_p_psi_head_sign_changes"]),
        mu0_p_psi_tail_sign_changes=int(metrics["mu0_p_psi_tail_sign_changes"]),
    )


def _grid_metadata(grid: Grid) -> dict[str, object]:
    return {
        "Nr": int(grid.Nr),
        "Nt": int(grid.Nt),
        "L_max": int(grid.L_max),
        "M_max": int(grid.M_max),
        "K_max": None if grid.K_max is None else int(grid.K_max),
        "quadrature_scheme": grid.quadrature_scheme,
        "calculus_scheme": grid.calculus_scheme,
    }


def _benchmark_baseline_case_row(
    spec: BenchmarkCaseSpec, reference: ReferenceBundle
) -> dict[str, object]:
    case = _make_benchmark_case(spec, reference)
    result, equilibrium, shape_x = _solve_once(case)
    metrics = _benchmark_case_diagnostics(reference, equilibrium, shape_x)
    return {
        "case_name": spec.case_name,
        "mode": spec.mode,
        "coordinate": spec.coordinate,
        "constraint": spec.constraint,
        "input_kind": spec.input_kind,
        "success": bool(result.success),
        "function_evaluations": int(result.function_evaluations),
        "jacobian_evaluations": int(result.jacobian_evaluations),
        "iterations": int(result.iterations),
        "residual_norm_final": float(result.residual_norm_final),
        "state_size": int(result.x.size),
        "state_norm": float(np.linalg.norm(result.x)),
        "state_min": float(np.min(result.x)),
        "state_max": float(np.max(result.x)),
        **metrics,
    }


def build_benchmark_baseline_payload(*, show_progress: bool = False) -> dict[str, object]:
    """Build a deterministic, non-timing benchmark regression payload."""
    reference = _solve_reference(show_progress=show_progress)
    specs = list(_iter_benchmark_specs())
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        row = _benchmark_baseline_case_row(spec, reference)
        rows.append(row)
        if show_progress:
            print(
                f"[{index:02d}/{len(specs)}] {spec.case_name}: "
                f"residual={float(row['residual_norm_final']):.3e} | "
                f"shape={float(row['shape_error']):.3e}"
            )

    return {
        "schema_version": BENCHMARK_BASELINE_SCHEMA_VERSION,
        "reference_cache_version": REFERENCE_CACHE_VERSION,
        "case_count": len(rows),
        "modes": list(BENCHMARK_MODES),
        "input_kinds": list(BENCHMARK_INPUT_KINDS),
        "mode_constraints": {
            mode: list(constraints) for mode, constraints in BENCHMARK_MODE_CONSTRAINTS.items()
        },
        "shape_match_tol": float(SHAPE_MATCH_TOL),
        "reference": {
            "case": "PF_rho_uniform_Ip",
            "grid": _grid_metadata(REFERENCE_GRID),
            "source_sample_count": int(REFERENCE_SOURCE_SAMPLE_COUNT),
            "Ip": float(REFERENCE_IP),
            "mu0_Ip": float(REFERENCE_MU0_IP),
            "residual_norm_final": float(reference.result.residual_norm_final),
            "function_evaluations": int(reference.result.function_evaluations),
            "jacobian_evaluations": int(reference.result.jacobian_evaluations),
            "iterations": int(reference.result.iterations),
        },
        "test": {
            "grid": _grid_metadata(TEST_GRID),
            "source_sample_count": int(TEST_SOURCE_SAMPLE_COUNT),
            "solver": {
                "method": CONFIG.method,
                "max_residual": float(CONFIG.max_residual),
                "max_evaluations": int(CONFIG.max_evaluations),
            },
        },
        "cases": rows,
    }


def write_benchmark_baseline(*, show_progress: bool = False) -> Path:
    """Write the deterministic non-timing benchmark regression baseline."""
    payload = build_benchmark_baseline_payload(show_progress=show_progress)
    BENCHMARK_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BENCHMARK_BASELINE_PATH


def _write_report(
    reference: ReferenceBundle,
    rows: list[BenchmarkCaseResult],
    plot_failures: list[str] | None = None,
) -> None:
    worst_shape = max(rows, key=lambda row: row.shape_error)
    slowest_case = max(rows, key=lambda row: row.avg_ms)
    largest_function_evaluations_case = max(
        rows, key=lambda row: int(row.result.function_evaluations)
    )
    worst_psi_r_case = max(rows, key=lambda row: row.psi_r_rel_rms_error)
    worst_ff_psi_case = max(rows, key=lambda row: row.ff_psi_rel_rms_error)
    worst_mu0_p_psi_case = max(rows, key=lambda row: row.mu0_p_psi_rel_rms_error)
    most_oscillatory_psi_r_case = max(
        rows, key=lambda row: row.psi_r_head_sign_changes + row.psi_r_tail_sign_changes
    )
    most_oscillatory_ff_psi_case = max(
        rows, key=lambda row: row.ff_psi_head_sign_changes + row.ff_psi_tail_sign_changes
    )
    most_oscillatory_mu0_p_psi_case = max(
        rows, key=lambda row: row.mu0_p_psi_head_sign_changes + row.mu0_p_psi_tail_sign_changes
    )
    failing_rows = [row for row in rows if row.shape_error > SHAPE_MATCH_TOL]
    rows_by_error = _sort_rows_desc(rows, lambda row: row.shape_error)
    rows_by_time = _sort_rows_desc(rows, lambda row: row.avg_ms)
    rows_by_function_evaluations = _sort_rows_desc(
        rows, lambda row: int(row.result.function_evaluations)
    )
    rows_by_psi_r_rms = _sort_rows_desc(rows, lambda row: row.psi_r_rel_rms_error)
    rows_by_ff_psi_rms = _sort_rows_desc(rows, lambda row: row.ff_psi_rel_rms_error)
    rows_by_mu0_p_psi_rms = _sort_rows_desc(rows, lambda row: row.mu0_p_psi_rel_rms_error)
    rows_by_psi_r_oscillation = _sort_rows_desc(
        rows, lambda row: row.psi_r_head_sign_changes + row.psi_r_tail_sign_changes
    )
    rows_by_ff_psi_oscillation = _sort_rows_desc(
        rows, lambda row: row.ff_psi_head_sign_changes + row.ff_psi_tail_sign_changes
    )
    rows_by_mu0_p_psi_oscillation = _sort_rows_desc(
        rows, lambda row: row.mu0_p_psi_head_sign_changes + row.mu0_p_psi_tail_sign_changes
    )

    lines = [f"PF-rho-Ip reference vs {len(rows)} low-resolution route-specific cases", ""]
    lines.extend(
        _render_pairs(
            [
                ("reference_case", "PF_RHO + Ip"),
                (
                    "reference_grid",
                    f"{REFERENCE_GRID.Nr}x{REFERENCE_GRID.Nt} ({REFERENCE_GRID.quadrature_scheme})",
                ),
                ("reference_source_samples", str(REFERENCE_SOURCE_SAMPLE_COUNT)),
                ("test_grid", f"{TEST_GRID.Nr}x{TEST_GRID.Nt} ({TEST_GRID.quadrature_scheme})"),
                ("test_source_samples", str(TEST_SOURCE_SAMPLE_COUNT)),
                ("repeat_count", str(BENCHMARK_REPEAT_COUNT)),
                ("shape_tol", f"{SHAPE_MATCH_TOL:.3e}"),
                ("failure_count", f"{len(failing_rows)}/{len(rows)}"),
                ("worst_shape_case", f"{worst_shape.case_name} ({worst_shape.shape_error:.6e})"),
                (
                    "worst_psi_r_rel_rms_case",
                    f"{worst_psi_r_case.case_name} ({worst_psi_r_case.psi_r_rel_rms_error:.6e})",
                ),
                (
                    "worst_ff_psi_rel_rms_case",
                    f"{worst_ff_psi_case.case_name} ({worst_ff_psi_case.ff_psi_rel_rms_error:.6e})",
                ),
                (
                    "worst_mu0_p_psi_rel_rms_case",
                    f"{worst_mu0_p_psi_case.case_name} "
                    f"({worst_mu0_p_psi_case.mu0_p_psi_rel_rms_error:.6e})",
                ),
                (
                    "most_oscillatory_psi_r_case",
                    f"{most_oscillatory_psi_r_case.case_name} "
                    f"(h/t={most_oscillatory_psi_r_case.psi_r_head_sign_changes}/{most_oscillatory_psi_r_case.psi_r_tail_sign_changes})",
                ),
                (
                    "most_oscillatory_ff_psi_case",
                    f"{most_oscillatory_ff_psi_case.case_name} "
                    f"(h/t={most_oscillatory_ff_psi_case.ff_psi_head_sign_changes}/{most_oscillatory_ff_psi_case.ff_psi_tail_sign_changes})",
                ),
                (
                    "most_oscillatory_mu0_p_psi_case",
                    f"{most_oscillatory_mu0_p_psi_case.case_name} "
                    f"(h/t={most_oscillatory_mu0_p_psi_case.mu0_p_psi_head_sign_changes}/{most_oscillatory_mu0_p_psi_case.mu0_p_psi_tail_sign_changes})",
                ),
                ("slowest_case", f"{slowest_case.case_name} ({slowest_case.avg_ms:.3f} ms)"),
                (
                    "largest_function_evaluations_case",
                    f"{largest_function_evaluations_case.case_name} "
                    f"({int(largest_function_evaluations_case.result.function_evaluations)})",
                ),
            ]
        )
    )

    lines.extend(["", "Case results", ""])
    lines.append(
        "case".ljust(24)
        + " | "
        + "shape_error".rjust(12)
        + " | "
        + "avg_ms".rjust(12)
        + " | "
        + "std_ms".rjust(12)
        + " | "
        + "evaluations".rjust(6)
        + " | "
        + "iterations".rjust(6)
        + " | "
        + "residual".rjust(12)
        + " | "
        + "ok".rjust(4)
    )
    lines.append("-" * 114)
    for row in rows:
        ok = "yes" if row.shape_error <= SHAPE_MATCH_TOL else "no"
        lines.append(
            f"{row.case_name:<24} | "
            f"{row.shape_error:>12.6e} | "
            f"{row.avg_ms:>12.3f} | "
            f"{row.std_ms:>12.3f} | "
            f"{int(row.result.function_evaluations):>6d} | "
            f"{int(row.result.iterations):>6d} | "
            f"{float(row.result.residual_norm_final):>12.6e} | "
            f"{ok:>4}"
        )

    lines.extend(["", "psi_r / FF_psi / mu0P_psi diagnostics", ""])
    lines.append(
        "case".ljust(24)
        + " | "
        + "psi_r_rms".rjust(10)
        + " | "
        + "psi_r_max".rjust(10)
        + " | "
        + "psi_r_h/t".rjust(9)
        + " | "
        + "FF_psi_rms".rjust(10)
        + " | "
        + "FF_psi_max".rjust(10)
        + " | "
        + "FF_psi_h/t".rjust(10)
        + " | "
        + "mu0P_rms".rjust(10)
        + " | "
        + "mu0P_max".rjust(10)
        + " | "
        + "mu0P_h/t".rjust(9)
    )
    lines.append("-" * 132)
    for row in rows:
        lines.append(
            f"{row.case_name:<24} | "
            f"{row.psi_r_rel_rms_error:>10.3e} | "
            f"{row.psi_r_rel_max_error:>10.3e} | "
            f"{f'{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}':>9} | "
            f"{row.ff_psi_rel_rms_error:>10.3e} | "
            f"{row.ff_psi_rel_max_error:>10.3e} | "
            f"{f'{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}':>10} | "
            f"{row.mu0_p_psi_rel_rms_error:>10.3e} | "
            f"{row.mu0_p_psi_rel_max_error:>10.3e} | "
            f"{f'{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}':>9}"
        )

    lines.extend(
        _render_ranking_section(
            "Largest shape_error ranking",
            rows_by_error,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                (
                    "right",
                    "evaluations",
                    6,
                    lambda index, row: int(row.result.function_evaluations),
                ),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest psi_r relative RMS error ranking",
            rows_by_psi_r_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "psi_r_rms", 10, lambda index, row: f"{row.psi_r_rel_rms_error:.3e}"),
                ("right", "psi_r_max", 10, lambda index, row: f"{row.psi_r_rel_max_error:.3e}"),
                (
                    "right",
                    "psi_r_h/t",
                    9,
                    lambda index, row: (
                        f"{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}"
                    ),
                ),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest FF_psi relative RMS error ranking",
            rows_by_ff_psi_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "FF_psi_rms", 10, lambda index, row: f"{row.ff_psi_rel_rms_error:.3e}"),
                ("right", "FF_psi_max", 10, lambda index, row: f"{row.ff_psi_rel_max_error:.3e}"),
                (
                    "right",
                    "FF_psi_h/t",
                    10,
                    lambda index, row: (
                        f"{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}"
                    ),
                ),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest mu0P_psi relative RMS error ranking",
            rows_by_mu0_p_psi_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "mu0P_h/t",
                    9,
                    lambda index, row: (
                        f"{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}"
                    ),
                ),
                ("right", "mu0P_rms", 10, lambda index, row: f"{row.mu0_p_psi_rel_rms_error:.3e}"),
                ("right", "mu0P_max", 10, lambda index, row: f"{row.mu0_p_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory psi_r ranking",
            rows_by_psi_r_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "psi_r_h/t",
                    9,
                    lambda index, row: (
                        f"{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}"
                    ),
                ),
                ("right", "psi_r_rms", 10, lambda index, row: f"{row.psi_r_rel_rms_error:.3e}"),
                ("right", "psi_r_max", 10, lambda index, row: f"{row.psi_r_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory FF_psi ranking",
            rows_by_ff_psi_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "FF_psi_h/t",
                    10,
                    lambda index, row: (
                        f"{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}"
                    ),
                ),
                ("right", "FF_psi_rms", 10, lambda index, row: f"{row.ff_psi_rel_rms_error:.3e}"),
                ("right", "FF_psi_max", 10, lambda index, row: f"{row.ff_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory mu0P_psi ranking",
            rows_by_mu0_p_psi_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "mu0P_h/t",
                    9,
                    lambda index, row: (
                        f"{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}"
                    ),
                ),
                ("right", "mu0P_rms", 10, lambda index, row: f"{row.mu0_p_psi_rel_rms_error:.3e}"),
                ("right", "mu0P_max", 10, lambda index, row: f"{row.mu0_p_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Slowest avg_ms ranking",
            rows_by_time,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
                (
                    "right",
                    "evaluations",
                    6,
                    lambda index, row: int(row.result.function_evaluations),
                ),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest function_evaluations ranking",
            rows_by_function_evaluations,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "evaluations",
                    6,
                    lambda index, row: int(row.result.function_evaluations),
                ),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )

    if plot_failures:
        lines.extend(["", "Plot failures", ""])
        lines.extend(plot_failures)

    (_artifact_dir() / "benchmark_compare.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_reference_summary_json(reference: ReferenceBundle) -> None:
    native_eq = reference.equilibrium
    summary_eq = native_eq.resample(REFERENCE_SUMMARY_GRID)

    boundary_R = _as_float64_array(summary_eq.R[-1])
    boundary_Z = _as_float64_array(summary_eq.Z[-1])
    boundary_R_closed = np.concatenate([boundary_R, boundary_R[:1]])
    boundary_Z_closed = np.concatenate([boundary_Z, boundary_Z[:1]])
    R_in = float(np.min(boundary_R))
    R_out = float(np.max(boundary_R))
    Z_top = float(np.max(boundary_Z))
    Z_bottom = float(np.min(boundary_Z))
    a_lcfs = 0.5 * (R_out - R_in)
    if a_lcfs <= 1.0e-14:
        raise ValueError("LCFS minor radius is too small to compute delta/elongation")
    R_top = float(boundary_R[int(np.argmax(boundary_Z))])
    R_bottom = float(boundary_R[int(np.argmin(boundary_Z))])
    elongation = 0.5 * (Z_top - Z_bottom) / a_lcfs
    delta_top = (float(native_eq.R0) - R_top) / a_lcfs
    delta_bottom = (float(native_eq.R0) - R_bottom) / a_lcfs
    delta_average = 0.5 * (delta_top + delta_bottom)

    rho = _REFERENCE_SUMMARY_RHO_AXIS
    native_rho_axis = _prepare_interp_axis(native_eq.rho)
    psin = _profile_interp(native_rho_axis, native_eq.psin, rho)
    np.maximum(psin, 0.0, out=psin)
    if psin.size:
        psin[0] = 0.0
        psin[-1] = 1.0

    mu0 = 4.0e-7 * np.pi
    native_P_psi = _as_float64_array(native_eq.alpha1 * native_eq.Pn_psin / mu0)
    P_psi = _profile_interp(native_rho_axis, native_P_psi, rho)
    q = _profile_interp(native_rho_axis, native_eq.q, rho)

    payload = {
        "sampling": {
            "Nr": int(REFERENCE_SUMMARY_GRID.Nr),
            "Nt": int(REFERENCE_SUMMARY_GRID.Nt),
            "quadrature_scheme": REFERENCE_SUMMARY_GRID.quadrature_scheme,
        },
        "geometry": {
            "R0": float(native_eq.R0),
            "Z0": float(native_eq.Z0),
            "a": float(native_eq.a),
            "B0": float(native_eq.B0),
            "aspect_ratio": float(native_eq.R0 / native_eq.a),
            "Ip": float(native_eq.Ip),
        },
        "outer_closed_surface": {
            "R": boundary_R_closed.tolist(),
            "Z": boundary_Z_closed.tolist(),
            "R_in": R_in,
            "R_out": R_out,
            "Z_top": Z_top,
            "Z_bottom": Z_bottom,
            "a_from_lcfs": a_lcfs,
            "elongation": float(elongation),
            "delta_top": float(delta_top),
            "delta_bottom": float(delta_bottom),
            "delta_average": float(delta_average),
        },
        "profiles": {
            "rho": rho.tolist(),
            "psin": psin.tolist(),
            "P_psi": P_psi.tolist(),
            "q": q.tolist(),
        },
    }

    _reference_summary_json_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compare_power_terms(rho: np.ndarray, power: int) -> np.ndarray:
    power = int(power)
    out = np.empty((3, rho.shape[0]), dtype=np.float64)
    if power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return out
    out[0] = rho**power
    out[1] = power * rho ** (power - 1)
    if power == 1:
        out[2].fill(0.0)
    else:
        out[2] = power * (power - 1) * rho ** (power - 2)
    return out


def _compare_envelope_terms(grid: Grid, envelope_power: int) -> np.ndarray:
    envelope_power = int(envelope_power)
    rho = np.asarray(grid.rho, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    out = np.empty((3, rho.shape[0]), dtype=np.float64)
    if envelope_power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return out
    if envelope_power == 1:
        out[0] = y
        out[1] = -2.0 * rho
        out[2].fill(-2.0)
        return out
    rho2 = np.asarray(grid.rho_powers[2], dtype=np.float64)
    out[0] = y**envelope_power
    out[1] = -2.0 * envelope_power * rho * y ** (envelope_power - 1)
    out[2] = -2.0 * envelope_power * y ** (envelope_power - 1) + 4.0 * envelope_power * (
        envelope_power - 1
    ) * rho2 * y ** (envelope_power - 2)
    return out


def _compare_shape_profile_values(equilibrium: object, name: str) -> np.ndarray:
    profile = getattr(equilibrium, "shape_profiles", {}).get(name)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    if profile is None or profile.coeff is None:
        return np.zeros_like(rho, dtype=np.float64)
    grid = equilibrium.grid
    fields = np.empty((3, grid.Nr), dtype=np.float64)
    update_profile(
        fields,
        grid.T,
        grid.T_r,
        grid.T_rr,
        _compare_power_terms(rho, int(profile.power)),
        _compare_envelope_terms(grid, int(profile.envelope_power)),
        float(profile.offset),
        np.asarray(profile.coeff, dtype=np.float64),
    )
    scale = float(profile.scale)
    return fields[0] if scale == 1.0 else fields[0] * scale


def _comparison_profile_data(equilibrium: object) -> dict[str, np.ndarray]:
    data = {
        "rho": np.asarray(equilibrium.rho, dtype=np.float64),
        "psi_r": np.asarray(equilibrium.alpha2 * equilibrium.psin_r, dtype=np.float64),
        "FF_psi": np.asarray(equilibrium.alpha1 * equilibrium.FFn_psin, dtype=np.float64),
        "mu0_P_psi": np.asarray(equilibrium.alpha1 * equilibrium.Pn_psin, dtype=np.float64),
    }
    for key in COMPARE_SHAPE_KEYS:
        data[key] = _compare_shape_profile_values(equilibrium, key)
    return data


def _comparison_profile_errors(
    reference_rho: np.ndarray,
    reference_values: np.ndarray,
    current_rho: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float]:
    reference_rho = np.asarray(reference_rho, dtype=np.float64)
    current_rho = np.asarray(current_rho, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    current_values = np.asarray(current_values, dtype=np.float64)
    if reference_rho.size <= current_rho.size:
        target_rho = reference_rho
        reference_on_target = reference_values
        current_on_target = np.interp(target_rho, current_rho, current_values)
    else:
        target_rho = current_rho
        reference_on_target = np.interp(target_rho, reference_rho, reference_values)
        current_on_target = current_values
    scale = max(float(np.max(np.abs(reference_on_target))), 1.0e-12)
    diff = current_on_target - reference_on_target
    return (
        float(np.max(np.abs(diff)) / scale),
        float(np.sqrt(np.mean(diff * diff)) / scale),
    )


def _comparison_profile_error_label(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1e}"


def _comparison_surface_curve(equilibrium: object, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(float(level), psin_unique, rho[order][unique_idx]))
    R = np.array(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.array(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def _comparison_close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return np.vstack((arr, arr[:1]))


def _comparison_rz_limits(
    curves: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([np.asarray(curve, dtype=np.float64) for curve in curves if curve.size])
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * 0.08, 1.0e-3)
    z_pad = max((z_max - z_min) * 0.08, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def _plot_benchmark_comparison(
    reference_eq: object,
    current_eq: object,
    outpath: Path,
    *,
    label_ref: str,
    label_other: str,
) -> dict[str, float]:
    compare_grid = Grid(
        Nr=64,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=max(reference_eq.grid.L_max, current_eq.grid.L_max),
        M_max=max(reference_eq.grid.M_max, current_eq.grid.M_max),
        K_max=reference_eq.grid.K_max if reference_eq.grid.K_max == current_eq.grid.K_max else None,
    )
    ref_surface = reference_eq.resample(compare_grid)
    cur_surface = current_eq.resample(compare_grid)
    ref_data = _comparison_profile_data(reference_eq)
    cur_data = _comparison_profile_data(current_eq)
    errors: dict[str, float] = {}

    fig = plt.figure(figsize=(14.0, 8.0))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.2, 0.9, 0.9),
        hspace=0.25,
        wspace=0.3,
        top=0.95,
        bottom=0.1,
        left=0.05,
        right=0.98,
    )

    surface_ax = fig.add_subplot(gs[:, 0])
    all_surface_curves: list[np.ndarray] = []
    for index, level in enumerate(COMPARE_SURFACE_LEVELS):
        ref_curve = _comparison_close_curve(_comparison_surface_curve(ref_surface, float(level)))
        cur_curve = _comparison_close_curve(_comparison_surface_curve(cur_surface, float(level)))
        all_surface_curves.extend([ref_curve, cur_curve])
        linewidth = 1.6 if index == len(COMPARE_SURFACE_LEVELS) - 1 else 1.15
        surface_ax.plot(
            ref_curve[:, 0],
            ref_curve[:, 1],
            color="black",
            linewidth=linewidth,
            alpha=0.85,
            label=label_ref if index == 0 else None,
        )
        surface_ax.plot(
            cur_curve[:, 0],
            cur_curve[:, 1],
            color="#d62728",
            linewidth=linewidth,
            alpha=0.75,
            linestyle="-",
            label=label_other if index == 0 else None,
        )
    xlim, ylim = _comparison_rz_limits(all_surface_curves)
    surface_ax.set_title("(a) Flux Surfaces")
    surface_ax.set_xlabel("R [m]")
    surface_ax.set_ylabel("Z [m]")
    surface_ax.set_xlim(*xlim)
    surface_ax.set_ylim(*ylim)
    surface_ax.set_aspect("equal", adjustable="box")
    surface_ax.grid(True, linestyle=":", alpha=0.35)
    surface_ax.legend(loc="upper right", frameon=False)

    for row, key in enumerate(COMPARE_SHAPE_KEYS):
        ax = fig.add_subplot(gs[row, 1])
        rel_max, rel_rms = _comparison_profile_errors(
            ref_data["rho"],
            ref_data[key],
            cur_data["rho"],
            cur_data[key],
        )
        errors[f"rel_{key}_max"] = rel_max
        errors[f"rel_{key}_rms"] = rel_rms
        ax.plot(ref_data["rho"], ref_data[key], color="black", linestyle="-", label=label_ref)
        ax.plot(cur_data["rho"], cur_data[key], color="#d62728", linestyle="--", label=label_other)
        ax.set_ylabel(rf"${key}$" if key != "s1" else r"$s_1$")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(
            0.03,
            0.97,
            f"err = {_comparison_profile_error_label(rel_max)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        if row == 0:
            ax.set_title("(b) Shape Parameters")
            ax.legend(loc="best", frameon=False)
        if row == len(COMPARE_SHAPE_KEYS) - 1:
            ax.set_xlabel(r"$\rho$")
        else:
            ax.tick_params(labelbottom=False)

    for row, (key, ylabel) in enumerate(COMPARE_SOURCE_KEYS):
        ax = fig.add_subplot(gs[row, 2])
        rel_max, rel_rms = _comparison_profile_errors(
            ref_data["rho"],
            ref_data[key],
            cur_data["rho"],
            cur_data[key],
        )
        errors[f"rel_{key}_max"] = rel_max
        errors[f"rel_{key}_rms"] = rel_rms
        ax.plot(ref_data["rho"], ref_data[key], color="black", linestyle="-", label=label_ref)
        ax.plot(cur_data["rho"], cur_data[key], color="#d62728", linestyle="--", label=label_other)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(
            0.03,
            0.97,
            f"err = {_comparison_profile_error_label(rel_max)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        if row == 0:
            ax.set_title("(c) Source Profiles")
        if row == len(COMPARE_SOURCE_KEYS) - 1:
            ax.set_xlabel(r"$\rho$")
        else:
            ax.tick_params(labelbottom=False)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    return errors


def run_full_benchmark(
    *, show_progress: bool = SHOW_PROGRESS
) -> tuple[ReferenceBundle, list[BenchmarkCaseResult]]:
    """Main script entry: sweep the benchmark matrix and write reports."""
    reference = _solve_reference(show_progress=show_progress)
    rows: list[BenchmarkCaseResult] = []
    plot_failures: list[str] = []
    specs = list(_iter_benchmark_specs())
    plot_dir = _plot_dir() if PLOT else None

    for index, spec in enumerate(specs, start=1):
        row = _benchmark_case_result(spec, reference)
        rows.append(row)
        if show_progress:
            print(
                f"[{index:02d}/{len(specs)}] {row.case_name}: "
                f"time={row.avg_ms:.3f}+/-{row.std_ms:.3f} ms | "
                f"shape={row.shape_error:.3e} | "
                f"psi_r={row.psi_r_rel_rms_error:.2e}"
            )
        if plot_dir is not None:
            try:
                _plot_benchmark_comparison(
                    reference.equilibrium,
                    row.equilibrium,
                    plot_dir / f"{row.case_name}_compare.png",
                    label_ref="PF_RHO_ref",
                    label_other=row.case_name,
                )
                # row.equilibrium.plot(plot_dir / f"{row.case_name}_summary.png")
            except Exception as exc:
                message = f"{row.case_name}: {type(exc).__name__}: {exc}"
                plot_failures.append(message)
                if show_progress:
                    print(f"plot warning: {message}")

    _write_report(reference, rows, plot_failures)
    _write_reference_summary_json(reference)

    if plot_dir is not None:
        try:
            reference.equilibrium.plot(outpath=_artifact_dir() / "reference_summary.png")
        except Exception as exc:
            message = f"reference_summary: {type(exc).__name__}: {exc}"
            plot_failures.append(message)
            if show_progress:
                print(f"plot warning: {message}")
            _write_report(reference, rows, plot_failures)

    return reference, rows


def _run_as_script(argv: list[str] | None = None) -> int:
    """Console entry used by ``python tests/benchmark.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="write tests/baselines/benchmark_non_timing.json without running timing reports",
    )
    args = parser.parse_args(argv)

    if args.write_baseline:
        print(write_benchmark_baseline(show_progress=SHOW_PROGRESS))
        return 0

    run_full_benchmark(show_progress=SHOW_PROGRESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
