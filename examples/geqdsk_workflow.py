"""Standalone single-GEQDSK workflow demo.

This no-argument demo keeps the public workflow small:
1. choose one GEQDSK file with ``VEQPY_GEQDSK`` or ``data/EFIT.geqdsk``,
2. fit the fixed boundary,
3. solve the PF(psin) and PQ(psin) VEQPy reconstructions with the manuscript
   benchmark settings,
4. write a two-column route diagnostic figure and serialized converged
   equilibria.

The script is intentionally self-contained.  It does not import anything from
``scripts/`` because those manuscript helpers are not part of the public test
surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from veqpy.model import Boundary, Geqdsk, Grid, Problem, Profile
from veqpy.model.boundary import _fit_boundary_params
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

DEFAULT_GEQDSK = Path("data") / "SOLOVEV.geqdsk"

MU0 = 4.0e-7 * np.pi

PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_BASE_FONT_SIZE = 10
PLOT_MATH_FONTSET = "dejavusans"
FONT_SCALE = 1.0
TITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
SAVE_DPI = 330
SAVE_TRANSPARENT = False
FIXED_DECIMALS = 2
SCIENTIFIC_DECIMALS = 2
SINGLE_COLUMN_WIDTH = 4.75
DOUBLE_COLUMN_WIDTH = 9.5

SINGLE_FIGURE_SIZE = (SINGLE_COLUMN_WIDTH, 6.5)
FIGURE_SIZE = (DOUBLE_COLUMN_WIDTH, 6.5)
PANEL_GRID_NROWS = 5
PANEL_GRID_HEIGHT_RATIOS = (3.5, 0.60, 1.25, 0.20, 1.25)
PANEL_GRID_HSPACE = 0.0
PANEL_GRID_WSPACE = 0.18

TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
GRID_ALPHA = 0.35
GRID_LINESTYLE = ":"
GRID_LINE_WIDTH = 0.8
LOG_Y_FLOOR = 1.0e-8
PSI_ERROR_YMIN = 1.0e-6
PSI_ERROR_YMAX = 5.0

REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_STYLE = "--"
VEQPY_LINE_COLOR = "#d62728"
VEQPY_LINE_STYLE = "-"
SOURCE_FF_COLOR = "#1f77b4"
SOURCE_P_COLOR = "#ff7f0e"
PROFILE_LINE_WIDTH = 1.25
THIRD_ROW_PSI_COLOR = "#9467bd"
THIRD_ROW_SHAPE_COLOR = VEQPY_LINE_COLOR
THIRD_ROW_PSI_STYLE = "-"
THIRD_ROW_SHAPE_STYLE = "-"
THIRD_ROW_PSI_MARKER = "x"
THIRD_ROW_PSI_MARKER_SIZE = 4
THIRD_ROW_PSI_MARKER_EDGE_WIDTH = 1.2
THIRD_ROW_SHAPE_MARKER = "o"
THIRD_ROW_SHAPE_MARKER_SIZE = 3.5
SURFACE_LINE_WIDTH = 1.0
BOUNDARY_SURFACE_LINE_WIDTH = 1.35
REFERENCE_SURFACE_SCALE = 1.5
REFERENCE_AXIS_COLOR = "black"
REFERENCE_AXIS_MARKER = "x"
REFERENCE_AXIS_MARKER_SIZE = 30
VEQPY_AXIS_COLOR = VEQPY_LINE_COLOR
VEQPY_AXIS_MARKER = "x"
VEQPY_AXIS_MARKER_SIZE = 15
LEGEND_FRAME_ON = False
SURFACE_LEGEND_LOC = "upper right"
SOURCE_LEGEND_LOC = "upper right"
THIRD_ROW_LEGEND_LOC = "upper right"
SURFACE_PAD_FRACTION = 0.09
SURFACE_Y_TICK_INTERVAL = 1.0
SOURCE_TOP_HEADROOM = 0.30
SOURCE_BOTTOM_HEADROOM = 0.08
LEGEND_COLUMN_SPACING = 0.8
LEGEND_LABEL_SPACING = 0.15

SOLVE_NR = 32
SOLVE_NT = 32
PLOT_NR = 128
PLOT_NT = 256
SURFACE_COUNT = 10
PSIN_ERROR_RHO_LEVELS = tuple(np.linspace(0.1, 0.9, 9, dtype=np.float64))
SHAPE_RMS_PSIN_LEVELS = tuple(np.linspace(0.0, 1.0, 11, dtype=np.float64))
SHAPE_RMS_THETA_SAMPLE_COUNT = 16
BOUNDARY_MAXTOL = 1.0
SOLVER_METHOD = "hybr"
SOLVER_MAXFEV = 2000
SOLVER_INITIAL_POLICY = "homothetic"
SOLVER_WARMUP_RUNS = 1
SOLVER_TIMING_REPEATS = 5

BOUNDARY_FIT_M = 10
BOUNDARY_FIT_N = 10
SOURCE_ROUTES = ("PF", "PQ")

D_SHAPE_PROFILE_COEFFS = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
}

GENERAL_GEQDSK_PROFILE_COEFFS = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "c3": [0.0] * 5,
    "c4": [0.0] * 5,
    "c5": [0.0] * 5,
    "c6": [0.0] * 5,
    "c7": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
}

CASE_DISPLAY_NAMES = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
    "geqdsk": "GEQDSK",
}


@dataclass(frozen=True)
class CaseSpec:
    title: str
    reference_label: str
    case_key: str
    gfile_path: Path
    boundary_fit_m: int
    boundary_fit_n: int
    profile_coeffs: dict[str, list[float]]
    solve_nr: int = SOLVE_NR
    solve_nt: int = SOLVE_NT


@dataclass(frozen=True)
class SolveTimingStats:
    warmup_runs: int
    repeat_runs: int
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class CaseResult:
    route: str
    title: str
    reference_label: str
    case_spec: CaseSpec
    geqdsk: Geqdsk
    equilibrium: Any
    plot_equilibrium: Any
    reference_surfaces: dict[float, np.ndarray]
    veqpy_surfaces: dict[float, np.ndarray]
    reference_psin_profile: tuple[np.ndarray, np.ndarray]
    veqpy_psin_profile: tuple[np.ndarray, np.ndarray]
    psin_error_profile: tuple[np.ndarray, np.ndarray]
    shape_rms_profile: tuple[np.ndarray, np.ndarray]
    parameter_count: int
    boundary_fit_rms: float
    solver_residual: float
    solver_success: bool
    solver_message: str
    solve_timing: SolveTimingStats


@dataclass(frozen=True)
class RouteFailure:
    route: str
    case_spec: CaseSpec
    error_type: str
    message: str


def scaled_font_size(size: float) -> float:
    return float(size) * FONT_SCALE


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": PLOT_BASE_FONT_SIZE,
            "mathtext.fontset": PLOT_MATH_FONTSET,
            "axes.unicode_minus": False,
        }
    )


def ensure_output_dir() -> Path:
    env_out = os.environ.get("VEQPY_OUTPUT_DIR")
    outdir = Path(env_out) if env_out else Path.cwd() / "outputs" / "geqdsk_workflow"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def target_geqdsk_path() -> Path:
    path = Path(os.environ.get("VEQPY_GEQDSK", DEFAULT_GEQDSK))
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(
            f"GEQDSK file not found: {path}. Set VEQPY_GEQDSK to a GEQDSK file path."
        )
    return path


def profile_parameter_count(profile_coeffs: dict[str, list[float]]) -> int:
    return int(sum(len(values) for values in profile_coeffs.values() if values is not None))


def profiles_from_coeffs(profile_coeffs: dict[str, list[float]]) -> dict[str, Profile]:
    return {
        name: Profile(coeff=np.asarray(values, dtype=np.float64))
        for name, values in profile_coeffs.items()
    }


def infer_case_spec(gfile_path: Path) -> CaseSpec:
    stem = gfile_path.stem.upper()
    if "SOLOVEV" in stem:
        case_key = "solovev"
        reference_label = "Solov'ev"
        title = r"$\bf{(a)}$ D-shaped Equilibrium"
        profile_coeffs = D_SHAPE_PROFILE_COEFFS
    elif "CHEASE" in stem:
        case_key = "chease"
        reference_label = "CHEASE"
        title = r"$\bf{(a)}$ H-mode Equilibrium"
        profile_coeffs = GENERAL_GEQDSK_PROFILE_COEFFS
    elif "EFIT" in stem:
        case_key = "efit"
        reference_label = "EFIT"
        title = r"$\bf{(a)}$ X-point Equilibrium"
        profile_coeffs = GENERAL_GEQDSK_PROFILE_COEFFS
    else:
        case_key = "geqdsk"
        reference_label = gfile_path.stem
        title = r"$\bf{(a)}$ GEQDSK Equilibrium"
        profile_coeffs = GENERAL_GEQDSK_PROFILE_COEFFS

    return CaseSpec(
        title=title,
        reference_label=reference_label,
        case_key=case_key,
        gfile_path=gfile_path,
        boundary_fit_m=BOUNDARY_FIT_M,
        boundary_fit_n=BOUNDARY_FIT_N,
        profile_coeffs=profile_coeffs,
    )


def read_geqdsk(path: Path) -> Geqdsk:
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def build_boundary(
    geqdsk: Geqdsk, *, fit_m: int, fit_n: int
) -> tuple[Boundary, dict[str, float | np.ndarray]]:
    fit = _fit_boundary_params(
        geqdsk,
        M=fit_m,
        N=fit_n,
        maxtol=BOUNDARY_MAXTOL,
        R0=None,
        Z0=None,
        a=None,
        ka=None,
    )
    normalized = {
        "rms": float(fit["rms"]),
        "a": float(fit["a"]),
        "R0": float(fit["R0"]),
        "Z0": float(fit["Z0"]),
        "ka": float(fit["ka"]),
        "c_offsets": np.asarray(fit["c_offsets"], dtype=np.float64),
        "s_offsets": np.asarray(fit["s_offsets"], dtype=np.float64),
    }
    boundary = Boundary(
        a=normalized["a"],
        R0=normalized["R0"],
        Z0=normalized["Z0"],
        B0=float(geqdsk.Bt0),
        ka=normalized["ka"],
        c_offsets=normalized["c_offsets"],
        s_offsets=normalized["s_offsets"],
    )
    return boundary, normalized


def build_solver_case(
    boundary: Boundary,
    geqdsk: Geqdsk,
    *,
    route: str,
    profile_coeffs: dict[str, list[float]],
) -> Problem:
    route = str(route).upper()
    if route == "PF":
        current_input = np.asarray(geqdsk.FF_psi, dtype=np.float64)
    elif route == "PQ":
        current_input = np.asarray(geqdsk.q, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported source route {route!r}; expected one of {SOURCE_ROUTES}")

    return Problem(
        route=route,
        coordinate="psin",
        nodes="uniform",
        profiles=profiles_from_coeffs(profile_coeffs),
        boundary=boundary,
        heat_input=np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=current_input,
        Ip=float(geqdsk.Ip),
    )


def build_solver(case: Problem, solve_grid: Grid) -> Solver:
    return Solver(
        operator=Operator(solve_grid, case.copy()),
        config=SolverConfig(
            method=SOLVER_METHOD,
            max_evaluations=SOLVER_MAXFEV,
            initial_policy=SOLVER_INITIAL_POLICY,
            enable_fallback=False,
            enable_verbose=False,
            enable_history=False,
        ),
    )


def solve_existing_solver_once(solver: Solver) -> tuple[Solver, float]:
    solver.solve(
        enable_verbose=False,
        enable_history=False,
        initial_policy=SOLVER_INITIAL_POLICY,
        enable_fallback=False,
    )
    if solver.result is None:
        raise RuntimeError("solver completed without a SolverResult")
    return solver, float(solver.result.elapsed) / 1000.0


def solve_once(case: Problem, solve_grid: Grid) -> tuple[Solver, float]:
    solver = build_solver(case, solve_grid)
    return solve_existing_solver_once(solver)


def build_timing_stats(samples_ms: list[float], *, warmup_runs: int) -> SolveTimingStats:
    samples = tuple(float(sample) for sample in samples_ms)
    if not samples:
        raise ValueError("at least one timing sample is required")
    sample_array = np.asarray(samples, dtype=np.float64)
    std_ms = float(np.std(sample_array, ddof=1)) if sample_array.size > 1 else 0.0
    return SolveTimingStats(
        warmup_runs=int(warmup_runs),
        repeat_runs=int(sample_array.size),
        samples_ms=samples,
        median_ms=float(np.median(sample_array)),
        mean_ms=float(np.mean(sample_array)),
        std_ms=std_ms,
        min_ms=float(np.min(sample_array)),
        max_ms=float(np.max(sample_array)),
    )


def solve_equilibrium(
    case: Problem,
    *,
    solve_nr: int = SOLVE_NR,
    solve_nt: int = SOLVE_NT,
) -> tuple[Solver, Any, Any, SolveTimingStats]:
    solve_grid = Grid(Nr=int(solve_nr), Nt=int(solve_nt), quadrature_scheme="legendre")
    plot_grid = Grid(
        Nr=max(PLOT_NR, int(solve_nr)),
        Nt=max(PLOT_NT, int(solve_nt)),
        quadrature_scheme="uniform",
        L_max=solve_grid.L_max,
        M_max=solve_grid.M_max,
    )

    for _ in range(SOLVER_WARMUP_RUNS):
        solve_once(case, solve_grid)

    timed_samples_ms: list[float] = []
    solver: Solver | None = None
    for _ in range(SOLVER_TIMING_REPEATS):
        solver, elapsed_ms = solve_once(case, solve_grid)
        timed_samples_ms.append(elapsed_ms)

    if solver is None:
        raise RuntimeError("no timed solver run was executed")

    timing = build_timing_stats(timed_samples_ms, warmup_runs=SOLVER_WARMUP_RUNS)
    equilibrium = solver.build_equilibrium()
    return solver, equilibrium, equilibrium.resample(grid=plot_grid), timing


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return np.vstack((arr, arr[:1]))


def polygon_area(points: np.ndarray) -> float:
    curve = close_curve(points)
    x = curve[:, 0]
    y = curve[:, 1]
    return float(0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def point_in_polygon(points: np.ndarray, R: float, Z: float) -> bool:
    vertices = np.asarray(points, dtype=np.float64)
    inside = False
    j = vertices.shape[0] - 1
    for i in range(vertices.shape[0]):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > Z) != (yj > Z):
            dy = yj - yi
            if abs(dy) < 1.0e-14:
                dy = 1.0e-14 if dy >= 0.0 else -1.0e-14
            x_cross = (xj - xi) * (Z - yi) / dy + xi
            if R < x_cross:
                inside = not inside
        j = i
    return inside


def select_contour(
    candidates: list[np.ndarray], *, axis_center: tuple[float, float]
) -> np.ndarray | None:
    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not point_in_polygon(arr, axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(curve, dtype=np.float64) for curve in candidates), key=len)
    return None


def extract_gfile_surfaces(geqdsk: Geqdsk, levels: list[float]) -> dict[float, np.ndarray]:
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    psin_grid = (np.asarray(geqdsk.psi.T, dtype=np.float64) - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(geqdsk.Rmin, geqdsk.Rmax, geqdsk.NR, dtype=np.float64)
    Z = np.linspace(geqdsk.Zmin, geqdsk.Zmax, geqdsk.NZ, dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}
    contour_levels = [float(level) for level in levels if level < 1.0 - 1.0e-12]
    if contour_levels:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=contour_levels)
        plt.close(fig)
        for idx, level in enumerate(contour_levels):
            selected = select_contour(contour.allsegs[idx], axis_center=axis_center)
            if selected is not None:
                surfaces[level] = selected
    if any(abs(level - 1.0) <= 1.0e-12 for level in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def build_surface_from_psin(equilibrium: Any, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(level, psin_unique, rho[order][unique_idx]))
    R = np.array(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.array(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def curve_distance_metrics(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    dist = np.sqrt(
        np.sum((np.asarray(points_a)[:, None, :] - np.asarray(points_b)[None, :, :]) ** 2, axis=2)
    )
    nearest_a = dist.min(axis=1)
    nearest_b = dist.min(axis=0)
    return {
        "hausdorff": float(max(nearest_a.max(), nearest_b.max())),
        "rms": float(np.sqrt(0.5 * (np.mean(nearest_a**2) + np.mean(nearest_b**2)))),
    }


def collect_surface_metrics(
    reference_surfaces: dict[float, np.ndarray], equilibrium: Any, levels: list[float]
) -> tuple[dict[float, np.ndarray], dict[float, dict[str, float]]]:
    veqpy_surfaces: dict[float, np.ndarray] = {}
    metrics: dict[float, dict[str, float]] = {}
    for level in levels:
        if level not in reference_surfaces:
            continue
        veqpy_surface = build_surface_from_psin(equilibrium, level)
        if (
            np.asarray(reference_surfaces[level], dtype=np.float64).size == 0
            or np.asarray(veqpy_surface, dtype=np.float64).size == 0
        ):
            continue
        veqpy_surfaces[level] = veqpy_surface
        metrics[level] = curve_distance_metrics(reference_surfaces[level], veqpy_surface)
    return veqpy_surfaces, metrics


def sample_curve_at_theta(
    points: np.ndarray, *, center: tuple[float, float], theta_eval: np.ndarray
) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    theta = np.mod(np.arctan2(curve[:, 1] - center[1], curve[:, 0] - center[0]), 2.0 * np.pi)
    order = np.argsort(theta, kind="mergesort")
    theta_sorted = theta[order]
    curve_sorted = curve[order]
    theta_periodic = np.concatenate((theta_sorted, [theta_sorted[0] + 2.0 * np.pi]))
    R_periodic = np.concatenate((curve_sorted[:, 0], [curve_sorted[0, 0]]))
    Z_periodic = np.concatenate((curve_sorted[:, 1], [curve_sorted[0, 1]]))
    theta_target = np.mod(np.asarray(theta_eval, dtype=np.float64), 2.0 * np.pi)
    return np.column_stack(
        (
            np.interp(theta_target, theta_periodic, R_periodic),
            np.interp(theta_target, theta_periodic, Z_periodic),
        )
    )


def radial_profile_from_surface(
    points: np.ndarray, *, center: tuple[float, float], theta_eval: np.ndarray
) -> np.ndarray:
    sampled = sample_curve_at_theta(points, center=center, theta_eval=theta_eval)
    center_arr = np.asarray(center, dtype=np.float64)
    return np.sqrt(np.sum((sampled - center_arr[None, :]) ** 2, axis=1))


def axis_position_error(
    reference_axis: tuple[float, float], veqpy_axis: tuple[float, float]
) -> float:
    ref_axis = np.asarray(reference_axis, dtype=np.float64)
    vq_axis = np.asarray(veqpy_axis, dtype=np.float64)
    return float(np.linalg.norm(vq_axis - ref_axis))


def build_shape_rms_profile(geqdsk: Geqdsk, plot_equilibrium: Any) -> tuple[np.ndarray, np.ndarray]:
    profile_levels = [float(level) for level in SHAPE_RMS_PSIN_LEVELS]
    if not profile_levels:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    theta_eval = np.linspace(
        0.0,
        2.0 * np.pi,
        SHAPE_RMS_THETA_SAMPLE_COUNT,
        endpoint=False,
        dtype=np.float64,
    )
    ref_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    veqpy_center = (
        float(plot_equilibrium.R[0, 0]),
        float(plot_equilibrium.Z[0, 0]),
    )
    reference_surfaces = extract_gfile_surfaces(
        geqdsk,
        [level for level in profile_levels if level > 1.0e-12],
    )
    psin_axis: list[float] = []
    rms_values: list[float] = []
    for level in profile_levels:
        if abs(level) <= 1.0e-12:
            psin_axis.append(level)
            rms_values.append(axis_position_error(ref_center, veqpy_center))
            continue
        if level not in reference_surfaces:
            continue
        reference_surface = np.asarray(reference_surfaces[level], dtype=np.float64)
        veqpy_surface = np.asarray(
            build_surface_from_psin(plot_equilibrium, level), dtype=np.float64
        )
        if reference_surface.size == 0 or veqpy_surface.size == 0:
            continue
        ref_r = radial_profile_from_surface(
            reference_surface,
            center=ref_center,
            theta_eval=theta_eval,
        )
        veqpy_r = radial_profile_from_surface(
            veqpy_surface,
            center=veqpy_center,
            theta_eval=theta_eval,
        )
        rms_values.append(float(np.sqrt(np.mean((veqpy_r - ref_r) ** 2))))
        psin_axis.append(level)
    return np.asarray(psin_axis, dtype=np.float64), np.asarray(rms_values, dtype=np.float64)


def build_profile_from_equilibrium(equilibrium: Any) -> tuple[np.ndarray, np.ndarray]:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    curves = [
        np.column_stack((equilibrium.R[idx], equilibrium.Z[idx]))
        for idx in range(1, equilibrium.grid.Nr)
    ]
    edge_area = polygon_area(curves[-1])
    rho_geom = [0.0]
    psin_values = [0.0]
    for idx, curve in enumerate(curves, start=1):
        rho_geom.append(float((polygon_area(curve) / edge_area) ** 0.5))
        psin_values.append(float(psin[idx]))
    return np.asarray(rho_geom, dtype=np.float64), np.asarray(psin_values, dtype=np.float64)


def build_profile_from_gfile_levels(
    geqdsk: Geqdsk, psin_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    levels = sorted(
        {
            float(level)
            for level in np.asarray(psin_values, dtype=np.float64)
            if 0.0 < float(level) <= 1.0
        }
        | {1.0}
    )
    surfaces = extract_gfile_surfaces(geqdsk, levels)
    edge_area = polygon_area(surfaces[1.0])
    rho_geom = [0.0]
    profile_psin = [0.0]
    for level in np.asarray(psin_values[1:], dtype=np.float64):
        nearest = min(
            (key for key in surfaces if key > 0.0), key=lambda key: abs(key - float(level))
        )
        rho_geom.append(float((polygon_area(surfaces[nearest]) / edge_area) ** 0.5))
        profile_psin.append(float(level))
    return np.asarray(rho_geom, dtype=np.float64), np.asarray(profile_psin, dtype=np.float64)


def interpolate_unique(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    xq = np.asarray(x_eval, dtype=np.float64)
    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return np.interp(xq, x_unique, y_unique)


def build_psin_error_profile(
    reference_psin_profile: tuple[np.ndarray, np.ndarray],
    veqpy_psin_profile: tuple[np.ndarray, np.ndarray],
    *,
    rho_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rho_eval = np.asarray(
        PSIN_ERROR_RHO_LEVELS if rho_samples is None else rho_samples, dtype=np.float64
    )
    if rho_eval.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    rho_ref, psin_ref = reference_psin_profile
    rho_vq, psin_vq = veqpy_psin_profile
    psin_axis = interpolate_unique(rho_ref, psin_ref, rho_eval)
    ref_psin_samples = interpolate_unique(rho_ref, psin_ref, rho_eval)
    vq_psin_samples = interpolate_unique(rho_vq, psin_vq, rho_eval)
    return psin_axis, np.abs(vq_psin_samples - ref_psin_samples)


def default_levels_from_count(surface_count: int) -> list[float]:
    return [level / surface_count for level in range(1, surface_count + 1)] + [1.0]


def compute_rz_limits(
    curves: list[np.ndarray], *, pad_fraction: float = SURFACE_PAD_FRACTION
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack(
        [np.asarray(curve, dtype=np.float64) for curve in curves if np.asarray(curve).size]
    )
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * pad_fraction, 1.0e-3)
    z_pad = max((z_max - z_min) * pad_fraction, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def set_symmetric_ylim(ax: plt.Axes, *, headroom_ratio: float = 0.0) -> None:
    y0, y1 = ax.get_ylim()
    bound = max(abs(y0), abs(y1))
    if bound <= 0.0:
        bound = 1.0
    ax.set_ylim(-bound * (1.0 + headroom_ratio), bound * (1.0 + headroom_ratio))


def style_axis(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(ylabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        labeltop=False,
        labelright=False,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)
    ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)


def aggregate_surface_rms_error(profile: tuple[np.ndarray, np.ndarray]) -> float:
    values = np.asarray(profile[1], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def reference_a(case_result: CaseResult) -> float:
    return float(case_result.equilibrium.a)


def format_tex_float(value: float, *, precision: int = SCIENTIFIC_DECIMALS) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent <= 3:
        return f"{value:.{precision}f}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{precision}f}\times 10^{{{exponent}}}"


def format_normalized_error(error_m: float, normalizer_m: float) -> str:
    if not np.isfinite(error_m) or not np.isfinite(normalizer_m) or normalizer_m <= 0.0:
        return "--"
    return "$" + format_tex_float(float(error_m) / float(normalizer_m)) + "$"


def format_case_error_metric(case_result: CaseResult) -> str:
    return format_normalized_error(
        aggregate_surface_rms_error(case_result.shape_rms_profile),
        reference_a(case_result),
    )


def format_boundary_fit_metric(case_result: CaseResult) -> str:
    return format_normalized_error(
        float(case_result.boundary_fit_rms),
        reference_a(case_result),
    )


def _coeff_length(profile_coeffs: dict[str, list[float]], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(len(values))


def format_core_psin_tuple(case_spec: CaseSpec) -> str:
    profile_coeffs = case_spec.profile_coeffs
    values = (
        _coeff_length(profile_coeffs, "h"),
        _coeff_length(profile_coeffs, "v"),
        _coeff_length(profile_coeffs, "k"),
        _coeff_length(profile_coeffs, "psin"),
    )
    return "$(" + ", ".join(str(value) for value in values) + ")$"


def format_high_order_family_column(case_spec: CaseSpec, prefix: str) -> str:
    if prefix == "c" and case_spec.case_key == "solovev":
        return "--"
    return r"$(10, 5^{\times 7})$"


def build_case_summary_latex_table(case_result: CaseResult) -> str:
    indent = "              "
    header = [
        "Case",
        "Route",
        "Status",
        "Params",
        "Time [ms]",
        r"$E_{\mathrm{gqdsk}}/a$",
        r"$E_{\mathrm{lcfs}}/a$",
        "Core",
        "Cos",
        "Sin",
    ]
    label = CASE_DISPLAY_NAMES.get(case_result.case_spec.case_key, case_result.reference_label)
    row = [
        label,
        f"{case_result.route}(psin)",
        "ok" if case_result.solver_success else "not converged",
        f"${int(case_result.parameter_count)}$",
        f"${float(case_result.solve_timing.median_ms):.{FIXED_DECIMALS}f}$",
        format_case_error_metric(case_result),
        format_boundary_fit_metric(case_result),
        format_core_psin_tuple(case_result.case_spec),
        format_high_order_family_column(case_result.case_spec, "c"),
        format_high_order_family_column(case_result.case_spec, "s"),
    ]
    column_widths = [max(len(header[index]), len(row[index])) for index in range(len(header))]

    def format_row(cells: list[str]) -> str:
        return (
            " & ".join(cell.ljust(column_widths[index]) for index, cell in enumerate(cells))
            + r" \\"
        )

    return "\n".join(
        indent + line
        for line in [
            r"\hline",
            format_row(header),
            r"\hline",
            format_row(row),
            r"\hline",
        ]
    )


def build_case_result(case_spec: CaseSpec, *, route: str) -> CaseResult:
    route = str(route).upper()
    geqdsk = read_geqdsk(case_spec.gfile_path)
    boundary, fit = build_boundary(
        geqdsk,
        fit_m=case_spec.boundary_fit_m,
        fit_n=case_spec.boundary_fit_n,
    )
    case = build_solver_case(
        boundary,
        geqdsk,
        route=route,
        profile_coeffs=case_spec.profile_coeffs,
    )
    solver, equilibrium, plot_equilibrium, timing = solve_equilibrium(
        case,
        solve_nr=case_spec.solve_nr,
        solve_nt=case_spec.solve_nt,
    )

    levels = default_levels_from_count(SURFACE_COUNT)
    reference_surfaces = extract_gfile_surfaces(geqdsk, levels)
    veqpy_surfaces, surface_metrics = collect_surface_metrics(
        reference_surfaces, plot_equilibrium, levels
    )
    reference_psin_profile = build_profile_from_gfile_levels(geqdsk, equilibrium.psin)
    veqpy_psin_profile = build_profile_from_equilibrium(equilibrium)
    psin_error_profile = build_psin_error_profile(
        reference_psin_profile,
        veqpy_psin_profile,
    )
    shape_rms_profile = build_shape_rms_profile(geqdsk, plot_equilibrium)

    result = solver.result
    if result is not None:
        print(f"{case_spec.title} {route}(psin)")
        print(f"boundary fit rms: {float(fit['rms']):.{SCIENTIFIC_DECIMALS}e}")
        print(f"solver residual: {float(result.residual_norm_final):.{SCIENTIFIC_DECIMALS}e}")
        print(f"solver success: {bool(result.success)}")
        if not bool(result.success):
            print(f"solver message: {result.message}")
        print(
            "solve timing: "
            f"median={timing.median_ms:.{FIXED_DECIMALS}f} ms, "
            f"mean={timing.mean_ms:.{FIXED_DECIMALS}f}+/-"
            f"{timing.std_ms:.{FIXED_DECIMALS}f} ms, "
            f"range=[{timing.min_ms:.{FIXED_DECIMALS}f}, "
            f"{timing.max_ms:.{FIXED_DECIMALS}f}] ms, "
            f"n={timing.repeat_runs}, warmup={timing.warmup_runs}"
        )
        print(f"target Ip: {float(geqdsk.Ip):.{SCIENTIFIC_DECIMALS}e}")
        print(f"solved Ip: {float(equilibrium.Ip):.{SCIENTIFIC_DECIMALS}e}")
        if 1.0 in surface_metrics:
            print(f"boundary rms distance: {surface_metrics[1.0]['rms']:.{SCIENTIFIC_DECIMALS}e}")

    return CaseResult(
        route=route,
        title=f"{case_spec.title} {route}(psin)",
        reference_label=case_spec.reference_label,
        case_spec=case_spec,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        plot_equilibrium=plot_equilibrium,
        reference_surfaces=reference_surfaces,
        veqpy_surfaces=veqpy_surfaces,
        reference_psin_profile=reference_psin_profile,
        veqpy_psin_profile=veqpy_psin_profile,
        psin_error_profile=psin_error_profile,
        shape_rms_profile=shape_rms_profile,
        parameter_count=profile_parameter_count(case_spec.profile_coeffs),
        boundary_fit_rms=float(fit["rms"]),
        solver_residual=float("nan") if result is None else float(result.residual_norm_final),
        solver_success=False if result is None else bool(result.success),
        solver_message="solver produced no result" if result is None else str(result.message),
        solve_timing=timing,
    )


def route_panel_title(case_result: CaseResult) -> str:
    label = CASE_DISPLAY_NAMES.get(case_result.case_spec.case_key, case_result.reference_label)
    status = "" if case_result.solver_success else "\nnot converged"
    return f"{label} {case_result.route}(psin){status}"


def plot_source_profiles(ax: plt.Axes, case_result: CaseResult) -> None:
    psin_axis = np.linspace(0.0, 1.0, case_result.geqdsk.NR, dtype=np.float64)
    if case_result.route == "PQ":
        ax.plot(
            psin_axis,
            np.asarray(case_result.geqdsk.q, dtype=np.float64),
            color=SOURCE_FF_COLOR,
            linewidth=PROFILE_LINE_WIDTH,
            label=r"$q$",
        )
    else:
        ax.plot(
            psin_axis,
            np.asarray(case_result.geqdsk.FF_psi, dtype=np.float64),
            color=SOURCE_FF_COLOR,
            linewidth=PROFILE_LINE_WIDTH,
            label=r"$FF_\psi$",
        )
    ax.plot(
        psin_axis,
        MU0 * np.asarray(case_result.geqdsk.P_psi, dtype=np.float64),
        color=SOURCE_P_COLOR,
        linestyle="--",
        linewidth=PROFILE_LINE_WIDTH,
        label=r"$\mu_0 P_\psi$",
    )


def plot_case_result_column(
    case_result: CaseResult,
    *,
    ax_surfaces: plt.Axes,
    ax_source: plt.Axes,
    ax_errors: plt.Axes,
    rz_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_ylabels: bool = True,
) -> None:
    for idx, level in enumerate(sorted(case_result.reference_surfaces)):
        linewidth = SURFACE_LINE_WIDTH if level < 1.0 - 1.0e-12 else BOUNDARY_SURFACE_LINE_WIDTH
        reference_curve = close_curve(case_result.reference_surfaces[level])
        ax_surfaces.plot(
            reference_curve[:, 0],
            reference_curve[:, 1],
            linestyle=REFERENCE_LINE_STYLE,
            color=REFERENCE_LINE_COLOR,
            linewidth=linewidth * REFERENCE_SURFACE_SCALE,
            label=(case_result.reference_label if idx == 0 else None),
        )
        if level in case_result.veqpy_surfaces:
            veqpy_curve = close_curve(case_result.veqpy_surfaces[level])
            ax_surfaces.plot(
                veqpy_curve[:, 0],
                veqpy_curve[:, 1],
                linestyle=VEQPY_LINE_STYLE,
                color=VEQPY_LINE_COLOR,
                linewidth=linewidth,
                label=("VEQ" if idx == 0 else None),
            )

    style_axis(
        ax_surfaces,
        title=route_panel_title(case_result),
        xlabel="R [m]",
        ylabel="Z [m]" if show_ylabels else "",
    )
    if rz_limits is None:
        surface_curves = list(case_result.reference_surfaces.values()) + list(
            case_result.veqpy_surfaces.values()
        )
        limits = compute_rz_limits(surface_curves)
    else:
        limits = rz_limits
    ax_surfaces.set_xlim(*limits[0])
    ax_surfaces.set_ylim(*limits[1])
    ax_surfaces.yaxis.set_major_locator(MultipleLocator(SURFACE_Y_TICK_INTERVAL))
    ax_surfaces.set_aspect("equal")
    ax_surfaces.scatter(
        [float(case_result.geqdsk.Raxis)],
        [float(case_result.geqdsk.Zaxis)],
        color=REFERENCE_AXIS_COLOR,
        marker=REFERENCE_AXIS_MARKER,
        s=REFERENCE_AXIS_MARKER_SIZE,
        zorder=5,
        label="_nolegend_",
    )
    ax_surfaces.scatter(
        [float(case_result.equilibrium.R[0, 0])],
        [float(case_result.equilibrium.Z[0, 0])],
        color=VEQPY_AXIS_COLOR,
        marker=VEQPY_AXIS_MARKER,
        s=VEQPY_AXIS_MARKER_SIZE,
        zorder=6,
        label="_nolegend_",
    )
    ax_surfaces.legend(
        loc=SURFACE_LEGEND_LOC,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        frameon=LEGEND_FRAME_ON,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )

    plot_source_profiles(ax_source, case_result)
    style_axis(ax_source, title="", xlabel="", ylabel="value" if show_ylabels else "")
    ax_source.tick_params(labelbottom=False)
    set_symmetric_ylim(
        ax_source,
        headroom_ratio=max(SOURCE_BOTTOM_HEADROOM, SOURCE_TOP_HEADROOM),
    )
    ax_source.legend(
        loc=SOURCE_LEGEND_LOC,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        frameon=LEGEND_FRAME_ON,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )

    psi_psin, psi_error = case_result.psin_error_profile
    shape_psin, shape_rms = case_result.shape_rms_profile
    shape_rms_normalized = shape_rms / max(reference_a(case_result), 1.0e-12)
    if shape_psin.size and shape_rms.size:
        ax_errors.semilogy(
            shape_psin,
            np.maximum(shape_rms_normalized, LOG_Y_FLOOR),
            color=THIRD_ROW_SHAPE_COLOR,
            linestyle=THIRD_ROW_SHAPE_STYLE,
            linewidth=PROFILE_LINE_WIDTH,
            marker=THIRD_ROW_SHAPE_MARKER,
            markersize=THIRD_ROW_SHAPE_MARKER_SIZE,
            label=r"${R}_{\mathrm{rms}}(\hat{\psi})/a$",
        )
    if psi_psin.size and psi_error.size:
        ax_errors.semilogy(
            psi_psin,
            np.maximum(psi_error, LOG_Y_FLOOR),
            color=THIRD_ROW_PSI_COLOR,
            linestyle=THIRD_ROW_PSI_STYLE,
            linewidth=PROFILE_LINE_WIDTH,
            marker=THIRD_ROW_PSI_MARKER,
            markersize=THIRD_ROW_PSI_MARKER_SIZE,
            markeredgewidth=THIRD_ROW_PSI_MARKER_EDGE_WIDTH,
            label=r"$|\Delta\hat{\psi}(\rho)|$",
        )
    style_axis(
        ax_errors,
        title="",
        xlabel=r"$\hat{\psi}$",
        ylabel="error" if show_ylabels else "",
    )
    ax_errors.set_ylim(PSI_ERROR_YMIN, PSI_ERROR_YMAX)
    ax_errors.legend(
        loc=THIRD_ROW_LEGEND_LOC,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        frameon=LEGEND_FRAME_ON,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )


def plot_route_failure(
    failure: RouteFailure,
    *,
    ax_surfaces: plt.Axes,
    ax_source: plt.Axes,
    ax_errors: plt.Axes,
) -> None:
    for ax in (ax_surfaces, ax_source, ax_errors):
        ax.set_axis_off()
    label = CASE_DISPLAY_NAMES.get(failure.case_spec.case_key, failure.case_spec.reference_label)
    ax_surfaces.set_title(f"{label} {failure.route}(psin)\nfailed", fontsize=TITLE_FONT_SIZE)
    ax_surfaces.text(
        0.5,
        0.5,
        f"{failure.error_type}\n{failure.message}",
        ha="center",
        va="center",
        transform=ax_surfaces.transAxes,
        fontsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
        wrap=True,
    )


def route_compare_limits(
    outcomes: dict[str, CaseResult | RouteFailure],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    curves: list[np.ndarray] = []
    for outcome in outcomes.values():
        if isinstance(outcome, RouteFailure):
            continue
        curves.extend(outcome.reference_surfaces.values())
        curves.extend(outcome.veqpy_surfaces.values())
    if not curves:
        return None
    return compute_rz_limits(curves)


def build_route_compare_figure(
    outcomes: dict[str, CaseResult | RouteFailure],
) -> plt.Figure:
    apply_plot_style()
    fig = plt.figure(figsize=FIGURE_SIZE)
    panel_grid = fig.add_gridspec(
        PANEL_GRID_NROWS,
        len(SOURCE_ROUTES),
        height_ratios=PANEL_GRID_HEIGHT_RATIOS,
        hspace=PANEL_GRID_HSPACE,
        wspace=PANEL_GRID_WSPACE,
    )
    rz_limits = route_compare_limits(outcomes)
    for col, route in enumerate(SOURCE_ROUTES):
        ax_surfaces = fig.add_subplot(panel_grid[0, col])
        ax_source = fig.add_subplot(panel_grid[2, col])
        ax_errors = fig.add_subplot(panel_grid[4, col], sharex=ax_source)
        outcome = outcomes[route]
        if isinstance(outcome, RouteFailure):
            plot_route_failure(
                outcome,
                ax_surfaces=ax_surfaces,
                ax_source=ax_source,
                ax_errors=ax_errors,
            )
        else:
            plot_case_result_column(
                outcome,
                ax_surfaces=ax_surfaces,
                ax_source=ax_source,
                ax_errors=ax_errors,
                rz_limits=rz_limits,
                show_ylabels=(col == 0),
            )

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.94)
    return fig


def build_single_case_figure(case_result: CaseResult) -> plt.Figure:
    apply_plot_style()
    fig = plt.figure(figsize=SINGLE_FIGURE_SIZE)
    panel_grid = fig.add_gridspec(
        PANEL_GRID_NROWS,
        1,
        height_ratios=PANEL_GRID_HEIGHT_RATIOS,
        hspace=PANEL_GRID_HSPACE,
    )
    ax_surfaces = fig.add_subplot(panel_grid[0, 0])
    ax_source = fig.add_subplot(panel_grid[2, 0])
    ax_errors = fig.add_subplot(panel_grid[4, 0], sharex=ax_source)
    plot_case_result_column(
        case_result,
        ax_surfaces=ax_surfaces,
        ax_source=ax_source,
        ax_errors=ax_errors,
    )
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.08, top=0.94)
    return fig


def build_route_outcomes(case_spec: CaseSpec) -> dict[str, CaseResult | RouteFailure]:
    outcomes: dict[str, CaseResult | RouteFailure] = {}
    for route in SOURCE_ROUTES:
        try:
            outcomes[route] = build_case_result(case_spec, route=route)
        except Exception as exc:  # noqa: BLE001 - public example should show both route outcomes.
            message = str(exc).splitlines()[-1] if str(exc).strip() else type(exc).__name__
            outcomes[route] = RouteFailure(
                route=route,
                case_spec=case_spec,
                error_type=type(exc).__name__,
                message=message,
            )
            print(f"{case_spec.title} {route}(psin)")
            print(f"solver failed: {type(exc).__name__}: {message}")
    return outcomes


def write_equilibrium_outputs(
    outcomes: dict[str, CaseResult | RouteFailure],
    *,
    output_dir: Path,
) -> list[Path]:
    saved_paths: list[Path] = []
    for route in SOURCE_ROUTES:
        outcome = outcomes[route]
        if isinstance(outcome, RouteFailure) or not outcome.solver_success:
            continue
        route_path = output_dir / f"demo_geqdsk_{route.lower()}_equilibrium.json"
        outcome.equilibrium.write_json(str(route_path))
        saved_paths.append(route_path)
        if route == "PF":
            compatibility_path = output_dir / "demo_geqdsk_equilibrium.json"
            outcome.equilibrium.write_json(str(compatibility_path))
            saved_paths.append(compatibility_path)
    return saved_paths


def print_outcome_summary(outcomes: dict[str, CaseResult | RouteFailure]) -> None:
    for route in SOURCE_ROUTES:
        outcome = outcomes[route]
        if isinstance(outcome, RouteFailure):
            print(f"{route}(psin) failed: {outcome.error_type}: {outcome.message}")
            continue
        print(build_case_summary_latex_table(outcome))


def main() -> None:
    target_geqdsk = target_geqdsk_path()
    case_spec = infer_case_spec(target_geqdsk)
    outcomes = build_route_outcomes(case_spec)

    output_dir = ensure_output_dir()
    output_figure = output_dir / "demo_geqdsk_workflow.png"

    fig = build_route_compare_figure(outcomes)
    fig.savefig(
        output_figure,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    plt.close(fig)
    output_equilibria = write_equilibrium_outputs(outcomes, output_dir=output_dir)

    print_outcome_summary(outcomes)
    print(f"Read GEQDSK  : {target_geqdsk}")
    print(f"Saved figure : {output_figure}")
    for path in output_equilibria:
        print(f"Saved equilibrium : {path}")


if __name__ == "__main__":
    main()
